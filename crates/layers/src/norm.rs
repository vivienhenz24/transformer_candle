//! Normalisation layers bundled with unified shape and dtype handling.
//!
//! Inputs follow the `(batch, seq, hidden)` convention. Normalisation happens
//! along the last axis while preserving the original layout. Implementations are
//! expected to promote intermediate statistics (mean, variance) to
//! [`PrecisionPolicy::reduction`] before casting the output back.

use candle_core::{Error, Result, Tensor, D};

use crate::{checks, dtypes::PrecisionPolicy};

/// Available normalisation strategies for transformer blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NormKind {
    /// Standard LayerNorm with learnable affine parameters.
    LayerNorm,
    /// RMSNorm variant that matches LLaMA style blocks.
    RmsNorm,
}

/// Configuration shared by all normalisation layers.
#[derive(Debug, Clone, PartialEq)]
pub struct NormConfig {
    /// Size of the hidden dimension being normalised.
    pub hidden_size: usize,
    /// Numeric stabiliser applied to variance or RMS computations.
    pub epsilon: f64,
    /// Desired variant of the normalisation routine.
    pub kind: NormKind,
    /// Whether post-normalisation affine parameters should be trained.
    pub elementwise_affine: bool,
}

impl NormConfig {
    /// Creates a configuration using defaults aligned with transformer blocks.
    pub fn new(hidden_size: usize, kind: NormKind) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
            kind,
            elementwise_affine: true,
        }
    }
}

/// Shared interface for normalisation layers used inside decoder blocks.
pub trait NormalizationLayer: Send + Sync {
    /// Returns the configuration so callers can check shape compatibility.
    fn config(&self) -> &NormConfig;

    /// Applies the normalisation to a hidden state tensor.
    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor>;
}

#[derive(Debug, Clone)]
struct NormImpl {
    config: NormConfig,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
}

impl NormImpl {
    fn new(config: NormConfig, weight: Option<Tensor>, bias: Option<Tensor>) -> Result<Self> {
        if matches!(config.kind, NormKind::RmsNorm) && bias.is_some() {
            return Err(Error::Msg(
                "RMSNorm does not support bias parameters".into(),
            ));
        }
        if config.elementwise_affine {
            if weight.is_none() {
                return Err(Error::Msg(
                    "elementwise affine norms must supply a scale parameter".into(),
                ));
            }
        } else if weight.is_some() || bias.is_some() {
            return Err(Error::Msg(
                "non-affine norms must not include scale or bias parameters".into(),
            ));
        }

        if let Some(weight) = &weight {
            checks::expect_shape("norm.weight", weight, &[config.hidden_size])?;
            checks::expect_dtype_in(
                "norm.weight",
                weight,
                &[candle_core::DType::F16, candle_core::DType::BF16, candle_core::DType::F32],
            )?;
            checks::expect_contiguous("norm.weight", weight)?;
        }
        if let Some(bias) = &bias {
            checks::expect_shape("norm.bias", bias, &[config.hidden_size])?;
            checks::expect_dtype_in(
                "norm.bias",
                bias,
                &[candle_core::DType::F16, candle_core::DType::BF16, candle_core::DType::F32],
            )?;
            checks::expect_contiguous("norm.bias", bias)?;
        }

        Ok(Self {
            config,
            weight,
            bias,
        })
    }

    fn config(&self) -> &NormConfig {
        &self.config
    }

    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        checks::expect_batch_seq_hidden("norm.input", hidden, self.config.hidden_size)?;

        let hidden_size = self.config.hidden_size as f64;
        let mut compute = policy.cast_for_reduction(hidden)?;

        if matches!(self.config.kind, NormKind::LayerNorm) {
            let mean = (compute.sum_keepdim(D::Minus1)? / hidden_size)?;
            compute = compute.broadcast_sub(&mean)?;
        }

        let variance = (compute.sqr()?.sum_keepdim(D::Minus1)? / hidden_size)?;
        let denom = (variance + self.config.epsilon)?.sqrt()?;
        let mut normalized = compute.broadcast_div(&denom)?;

        if normalized.dtype() != policy.compute() {
            normalized = normalized.to_dtype(policy.compute())?;
        }

        if let Some(weight) = &self.weight {
            let weight = weight.to_dtype(normalized.dtype())?;
            normalized = normalized.broadcast_mul(&weight)?;
        }
        if let Some(bias) = &self.bias {
            let bias = bias.to_dtype(normalized.dtype())?;
            normalized = normalized.broadcast_add(&bias)?;
        }

        policy.cast_to_storage(&normalized)
    }
}

/// Standard LayerNorm implementation with optional affine parameters.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    inner: NormImpl,
}

impl LayerNorm {
    /// Constructs a LayerNorm with learnable scale and bias parameters.
    pub fn new(weight: Tensor, bias: Tensor, mut config: NormConfig) -> Result<Self> {
        config.kind = NormKind::LayerNorm;
        config.elementwise_affine = true;
        Ok(Self {
            inner: NormImpl::new(config, Some(weight), Some(bias))?,
        })
    }

    /// Constructs a LayerNorm with an affine scale but no bias.
    pub fn with_scale(weight: Tensor, mut config: NormConfig) -> Result<Self> {
        config.kind = NormKind::LayerNorm;
        config.elementwise_affine = true;
        Ok(Self {
            inner: NormImpl::new(config, Some(weight), None)?,
        })
    }

    /// Constructs a LayerNorm without affine parameters (scale = 1, bias = 0).
    pub fn without_affine(mut config: NormConfig) -> Result<Self> {
        config.kind = NormKind::LayerNorm;
        config.elementwise_affine = false;
        Ok(Self {
            inner: NormImpl::new(config, None, None)?,
        })
    }
}

impl NormalizationLayer for LayerNorm {
    fn config(&self) -> &NormConfig {
        self.inner.config()
    }

    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        self.inner.forward(hidden, policy)
    }
}

/// Root mean square norm that mirrors LLaMA-style transformer blocks.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: NormImpl,
}

impl RmsNorm {
    /// Constructs a RMSNorm with learnable scale.
    pub fn new(weight: Tensor, mut config: NormConfig) -> Result<Self> {
        config.kind = NormKind::RmsNorm;
        config.elementwise_affine = true;
        Ok(Self {
            inner: NormImpl::new(config, Some(weight), None)?,
        })
    }

    /// Constructs a RMSNorm without affine parameters (pure normalization).
    pub fn without_scale(mut config: NormConfig) -> Result<Self> {
        config.kind = NormKind::RmsNorm;
        config.elementwise_affine = false;
        Ok(Self {
            inner: NormImpl::new(config, None, None)?,
        })
    }
}

impl NormalizationLayer for RmsNorm {
    fn config(&self) -> &NormConfig {
        self.inner.config()
    }

    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        self.inner.forward(hidden, policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::PrecisionPolicy;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ops;

    fn build_input(
        device: &Device,
        dtype: DType,
        batch: usize,
        seq: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let total = batch * seq * hidden;
        let data = (0..total)
            .map(|i| (i as f32 * 0.25_f32) - 1.5_f32)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (batch, seq, hidden), device)?.to_dtype(dtype)
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        a.to_dtype(DType::F32)?
            .sub(&b.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()
    }

    #[test]
    fn layer_norm_matches_reference_across_dtypes() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq = 3;
        let hidden = 4;
        let mut base_config = NormConfig::new(hidden, NormKind::LayerNorm);
        base_config.epsilon = 1e-5;

        let weight_f32 = Tensor::from_vec(vec![1.0f32, 0.5, -0.25, 1.5], (hidden,), &device)?;
        let bias_f32 = Tensor::from_vec(vec![0.1f32, -0.2, 0.05, 0.0], (hidden,), &device)?;

        for &dtype in &[DType::F32, DType::F16, DType::BF16] {
            let input = build_input(&device, dtype, batch, seq, hidden)?;
            let weight = weight_f32.to_dtype(dtype)?;
            let bias = bias_f32.to_dtype(dtype)?;
            let layer = LayerNorm::new(weight.clone(), bias.clone(), base_config.clone())?;
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);
            let output = layer.forward(&input, &policy)?;

            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.dtype(), dtype);

            let reference = ops::layer_norm(&input, &weight, &bias, base_config.epsilon as f32)?;
            let tol = match dtype {
                DType::F16 => 1e-3,
                DType::BF16 => 1e-2,
                _ => 5e-4,
            };
            let diff = max_diff(&output, &reference)?;
            assert!(diff < tol, "max diff {} for dtype {:?}", diff, dtype);
        }

        Ok(())
    }

    #[test]
    fn layer_norm_without_affine_equals_normalised_reference() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq = 3;
        let hidden = 2;
        let mut config = NormConfig::new(hidden, NormKind::LayerNorm);
        config.epsilon = 1e-5;

        let input = build_input(&device, DType::F32, batch, seq, hidden)?;
        let layer = LayerNorm::without_affine(config.clone())?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let output = layer.forward(&input, &policy)?;

        let weight = Tensor::ones((hidden,), DType::F32, &device)?;
        let bias = Tensor::zeros((hidden,), DType::F32, &device)?;
        let reference = ops::layer_norm(&input, &weight, &bias, config.epsilon as f32)?;

        let diff = max_diff(&output, &reference)?;
        assert!(diff < 5e-4);
        Ok(())
    }

    #[test]
    fn rms_norm_matches_reference_across_dtypes() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq = 4;
        let hidden = 6;
        let mut config = NormConfig::new(hidden, NormKind::RmsNorm);
        config.epsilon = 1e-6;

        let weight_f32 = Tensor::from_vec(
            (0..hidden)
                .map(|i| 1.0_f32 + (i as f32) * 0.01)
                .collect::<Vec<_>>(),
            (hidden,),
            &device,
        )?;

        for &dtype in &[DType::F32, DType::F16, DType::BF16] {
            let input = build_input(&device, dtype, batch, seq, hidden)?;
            let weight = weight_f32.to_dtype(dtype)?;
            let norm = RmsNorm::new(weight.clone(), config.clone())?;
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);
            let output = norm.forward(&input, &policy)?;

            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.dtype(), dtype);

            let reference = ops::rms_norm(&input, &weight, config.epsilon as f32)?;
            let tol = match dtype {
                DType::F16 => 1e-3,
                DType::BF16 => 1e-2,
                _ => 5e-4,
            };
            let diff = max_diff(&output, &reference)?;
            assert!(diff < tol, "max diff {} for dtype {:?}", diff, dtype);
        }
        Ok(())
    }

    #[test]
    fn rms_norm_without_scale_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq = 5;
        let hidden = 3;
        let mut config = NormConfig::new(hidden, NormKind::RmsNorm);
        config.epsilon = 1e-6;

        let input = build_input(&device, DType::F32, batch, seq, hidden)?;
        let norm = RmsNorm::without_scale(config.clone())?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let output = norm.forward(&input, &policy)?;

        let weight = Tensor::ones((hidden,), DType::F32, &device)?;
        let reference = ops::rms_norm(&input, &weight, config.epsilon as f32)?;
        let diff = max_diff(&output, &reference)?;
        assert!(diff < 5e-4);
        Ok(())
    }

    #[test]
    fn layer_norm_handles_edge_shapes() -> Result<()> {
        let device = Device::Cpu;
        let shapes = [(1, 1, 1), (2, 1, 1), (1, 64, 8), (2, 3, 256), (1, 4, 2048)];
        for &(batch, seq, hidden) in &shapes {
            let mut config = NormConfig::new(hidden, NormKind::LayerNorm);
            config.epsilon = 1e-5;
            let input = build_input(&device, DType::F32, batch, seq, hidden)?;
            let weight = Tensor::ones((hidden,), DType::F32, &device)?;
            let bias = Tensor::zeros((hidden,), DType::F32, &device)?;
            let layer = LayerNorm::new(weight.clone(), bias.clone(), config.clone())?;
            let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
            let output = layer.forward(&input, &policy)?;
            let reference = ops::layer_norm(&input, &weight, &bias, config.epsilon as f32)?;
            let diff = max_diff(&output, &reference)?;
            assert!(
                diff < 5e-4,
                "shape {:?} diff {}",
                (batch, seq, hidden),
                diff
            );
        }
        Ok(())
    }

    #[test]
    fn rms_norm_handles_long_sequences() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq = 128;
        let hidden = 16;
        let mut config = NormConfig::new(hidden, NormKind::RmsNorm);
        config.epsilon = 1e-6;

        let input = build_input(&device, DType::F32, batch, seq, hidden)?;
        let weight = Tensor::from_vec(vec![1.0f32; hidden], (hidden,), &device)?;
        let norm = RmsNorm::new(weight.clone(), config.clone())?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let output = norm.forward(&input, &policy)?;
        let reference = ops::rms_norm(&input, &weight, config.epsilon as f32)?;
        let diff = max_diff(&output, &reference)?;
        assert!(diff < 5e-4);
        Ok(())
    }

    #[test]
    fn gradient_check_layer_norm_sum_output() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq = 2;
        let hidden = 3;
        let total = batch * seq * hidden;
        let mut config = NormConfig::new(hidden, NormKind::LayerNorm);
        config.epsilon = 1e-5;
        let layer = LayerNorm::without_affine(config.clone())?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);

        let base = (0..total)
            .map(|i| (i as f32) * 0.1 - 0.2)
            .collect::<Vec<_>>();

        let eval_layer = |values: &[f32]| -> Result<f32> {
            let tensor = Tensor::from_vec(values.to_vec(), (batch, seq, hidden), &device)?;
            let out = layer.forward(&tensor, &policy)?;
            out.sum_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()
        };

        let weight = Tensor::ones((hidden,), DType::F32, &device)?;
        let bias = Tensor::zeros((hidden,), DType::F32, &device)?;
        let eval_reference = |values: &[f32]| -> Result<f32> {
            let tensor = Tensor::from_vec(values.to_vec(), (batch, seq, hidden), &device)?;
            let out = ops::layer_norm(&tensor, &weight, &bias, config.epsilon as f32)?;
            out.sum_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()
        };

        let eps = 1e-3f32;
        let mut our_grad = Vec::with_capacity(total);
        let mut ref_grad = Vec::with_capacity(total);
        for idx in 0..total {
            let mut plus = base.clone();
            plus[idx] += eps;
            let mut minus = base.clone();
            minus[idx] -= eps;
            let our_plus = eval_layer(&plus)?;
            let our_minus = eval_layer(&minus)?;
            our_grad.push((our_plus - our_minus) / (2.0 * eps));

            let ref_plus = eval_reference(&plus)?;
            let ref_minus = eval_reference(&minus)?;
            ref_grad.push((ref_plus - ref_minus) / (2.0 * eps));
        }

        for (idx, (ours, reference)) in our_grad.iter().zip(ref_grad.iter()).enumerate() {
            let diff = (ours - reference).abs();
            assert!(
                diff < 1e-3,
                "grad mismatch at {}: {} vs {}",
                idx,
                ours,
                reference
            );
        }

        Ok(())
    }

    #[test]
    fn gradient_check_rms_norm_sum_output() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq = 2;
        let hidden = 4;
        let total = batch * seq * hidden;
        let mut config = NormConfig::new(hidden, NormKind::RmsNorm);
        config.epsilon = 1e-6;
        let weight = Tensor::from_vec(vec![1.0f32; hidden], (hidden,), &device)?;
        let norm = RmsNorm::new(weight.clone(), config.clone())?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);

        let base = (0..total)
            .map(|i| (i as f32) * 0.2 - 0.5)
            .collect::<Vec<_>>();

        let eval_norm = |values: &[f32]| -> Result<f32> {
            let tensor = Tensor::from_vec(values.to_vec(), (batch, seq, hidden), &device)?;
            let out = norm.forward(&tensor, &policy)?;
            out.sum_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()
        };

        let eval_reference = |values: &[f32]| -> Result<f32> {
            let tensor = Tensor::from_vec(values.to_vec(), (batch, seq, hidden), &device)?;
            let out = ops::rms_norm(&tensor, &weight, config.epsilon as f32)?;
            out.sum_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()
        };

        let eps = 1e-3f32;
        for idx in 0..total {
            let mut plus = base.clone();
            plus[idx] += eps;
            let mut minus = base.clone();
            minus[idx] -= eps;
            let grad = (eval_norm(&plus)? - eval_norm(&minus)?) / (2.0 * eps);
            let ref_grad = (eval_reference(&plus)? - eval_reference(&minus)?) / (2.0 * eps);
            assert!(
                (grad - ref_grad).abs() < 1e-3,
                "grad mismatch at {}: {} vs {}",
                idx,
                grad,
                ref_grad
            );
        }

        Ok(())
    }
}
