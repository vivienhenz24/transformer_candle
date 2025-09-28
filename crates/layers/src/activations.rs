//! Activation catalogue for transformer feed-forward stacks.
//!
//! Activations consume tensors shaped as `(batch, seq, hidden)` and return
//! tensors with identical layout. Modules are expected to apply the activation
//! after casting to the compute dtype described by [`PrecisionPolicy`], then
//! optionally cast back to the requested output dtype.

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
