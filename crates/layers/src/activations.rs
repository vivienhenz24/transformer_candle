//! Activation catalogue for transformer feed-forward stacks.
//!
//! Activations consume tensors shaped as `(batch, seq, hidden)` and return
//! tensors with identical layout. Each implementation promotes inputs to the
//! compute dtype requested by [`PrecisionPolicy`] before evaluating the
//! non-linearity, then casts the result back to the storage dtype so callers can
//! chain additional mixed-precision aware operations.
//!
//! # Built-in formulas
//!
//! * **GELU** uses the erf-based approximation
//!   `0.5 * x * (1 + erf(x / sqrt(2)))`.
//! * **SiLU / Swish** computes `x * sigmoid(x)` via the fused kernel exposed by
//!   Candle.
//! * **SwiGLU** re-uses the SiLU kernel for the gating branch of gated MLPs.
//!
//! All implementations operate in at least `f32` precision to avoid precision
//! loss when inputs are stored in reduced formats such as `f16`/`bf16`.

use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::dtypes::PrecisionPolicy;

/// Identifies which non-linearity is implemented by an [`Activation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// Identity function, useful for debugging or wiring custom stacks.
    Identity,
    /// GeLU with the tanh approximation used by GPT style models.
    Gelu,
    /// SwiGLU-style activation where gating is fused with linear projections.
    SwiGlu,
    /// Standard SiLU (a.k.a. swish) activation.
    Silu,
    /// ReLU or variants that retain shape but zero out negative values.
    Relu,
}

/// Common interface shared by transformer-friendly activation functions.
pub trait Activation: Send + Sync {
    /// Returns the [`ActivationKind`] for introspection when wiring composite blocks.
    fn kind(&self) -> ActivationKind;

    /// Applies the activation to `input` using the precision rules in `policy`.
    ///
    /// Implementations should cast to `policy.compute()` when computing the
    /// activation and return a tensor compatible with
    /// `policy.cast_to_storage`.
    fn forward(&self, input: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor>;

    /// Indicates if the activation can safely operate in-place on `input`.
    fn allows_inplace(&self) -> bool {
        false
    }
}

/// Built-in activation backed by Candle kernels.
struct BuiltinActivation {
    kind: ActivationKind,
}

impl BuiltinActivation {
    fn new(kind: ActivationKind) -> Self {
        Self { kind }
    }
}

impl Activation for BuiltinActivation {
    fn kind(&self) -> ActivationKind {
        self.kind
    }

    fn forward(&self, input: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        match self.kind {
            ActivationKind::Identity => policy.cast_to_storage(input),
            ActivationKind::Relu => {
                let compute = policy.cast_for_matmul(input)?;
                policy.cast_to_storage(&compute.relu()?)
            }
            ActivationKind::Gelu => {
                let compute = policy.cast_for_matmul(input)?;
                policy.cast_to_storage(&compute.gelu_erf()?)
            }
            ActivationKind::Silu | ActivationKind::SwiGlu => {
                let compute = policy.cast_for_matmul(input)?;
                policy.cast_to_storage(&compute.silu()?)
            }
        }
    }

    fn allows_inplace(&self) -> bool {
        matches!(self.kind, ActivationKind::Relu)
    }
}

/// Returns a shared built-in activation implementation.
pub fn builtin(kind: ActivationKind) -> Arc<dyn Activation> {
    Arc::new(BuiltinActivation::new(kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::PrecisionPolicy;
    use candle_core::{DType, Device};
    use std::f64::consts::SQRT_2;

    #[test]
    fn gelu_matches_reference_formula() -> Result<()> {
        let device = Device::Cpu;
        let activation = builtin(ActivationKind::Gelu);
        let input = Tensor::from_slice(&[-2.5f32, -0.5, 0.0, 1.0, 3.0], (5,), &device)?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let output = activation.forward(&input, &policy)?.to_dtype(DType::F32)?;

        let reference = {
            let x = input.to_dtype(DType::F32)?;
            let scaled = x.affine(1.0 / SQRT_2, 0.0)?;
            let term = scaled.erf()?;
            let one = Tensor::ones_like(&term)?;
            let inner = (one + term)?;
            x.affine(0.5, 0.0)?.broadcast_mul(&inner)?
        };

        let diff = output.sub(&reference)?.abs()?.max_all()?.to_vec0::<f32>()?;
        assert!(diff < 5e-6);
        Ok(())
    }

    #[test]
    fn silu_matches_swish_reference() -> Result<()> {
        let device = Device::Cpu;
        let activation = builtin(ActivationKind::Silu);
        let input = Tensor::from_slice(&[-3.0f32, -1.0, 0.0, 0.5, 2.0], (5,), &device)?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let output = activation.forward(&input, &policy)?.to_dtype(DType::F32)?;

        let one = Tensor::ones_like(&input)?;
        let neg = input.affine(-1.0, 0.0)?;
        let exp = neg.exp()?;
        let denom = (one.clone() + exp)?;
        let sigmoid = one.broadcast_div(&denom)?;
        let reference = input.mul(&sigmoid)?;
        let diff = output.sub(&reference)?.abs()?.max_all()?.to_vec0::<f32>()?;
        assert!(diff < 5e-6);
        Ok(())
    }
}
