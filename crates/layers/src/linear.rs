//! Linear and affine projection helpers.
//!
//! Linear layers expect inputs shaped `(batch, seq, in_dim)` and return tensors
//! with `(batch, seq, out_dim)`. Multi-projection variants pack the output as
//! `(batch, seq, num_projections * head_dim)` so that caller controlled reshapes
//! can split them for attention or gated activations. We consistently cast
//! weights and activations to [`PrecisionPolicy::compute`] for matmuls and rely
//! on [`PrecisionPolicy::cast_to_storage`](crate::dtypes::PrecisionPolicy::cast_to_storage)
//! for the final dtype. Initialisation policies mirror common transformer
//! recipes (Glorot, Kaiming, scaled variants) so downstream crates can share a
//! single implementation.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Error, Result, Tensor};

use crate::{checks, dtypes::PrecisionPolicy};

/// Configuration shared by dense projection layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearConfig {
    /// Incoming feature dimension.
    pub input_dim: usize,
    /// Output feature dimension per projection shard.
    pub output_dim: usize,
    /// Whether a learnable bias vector should be applied.
    pub bias: bool,
    /// Number of projections fused together (1 for standard linear).
    pub fused_projections: usize,
}

impl LinearConfig {
    /// Creates a configuration for a single projection layer.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            bias: true,
            fused_projections: 1,
        }
    }

    /// Indicates whether outputs are packed for multi-way projections.
    pub fn is_fused(&self) -> bool {
        self.fused_projections > 1
    }

    /// Total number of output features produced by the layer.
    pub fn total_output_dim(&self) -> usize {
        self.output_dim * self.fused_projections
    }
}

/// Shared interface for affine projections.
pub trait LinearLayer: Send + Sync {
    /// Returns the static configuration used to validate inputs.
    fn config(&self) -> &LinearConfig;

    /// Applies the linear projection, promoting to the compute dtype when needed.
    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor>;
}

/// Supported weight initialisation policies for transformer projections.
#[derive(Debug, Clone)]
pub enum LinearInit {
    /// Xavier/Glorot uniform initialisation.
    XavierUniform,
    /// Xavier/Glorot normal initialisation.
    XavierNormal,
    /// Kaiming/He uniform initialisation (defaults to ReLU gain).
    KaimingUniform { negative_slope: f64 },
    /// Kaiming/He normal initialisation (defaults to ReLU gain).
    KaimingNormal { negative_slope: f64 },
    /// Scales another policy to support deep network stabilisation.
    Scaled { base: Box<LinearInit>, scale: f64 },
}

impl LinearInit {
    /// Convenience helper to scale an existing policy.
    pub fn scaled(base: LinearInit, scale: f64) -> Self {
        Self::Scaled {
            base: Box::new(base),
            scale,
        }
    }

    fn sample(&self, shape: (usize, usize), device: &Device, dtype: DType) -> Result<Tensor> {
        let (out_dim, in_dim) = shape;
        let (fan_in, fan_out) = (in_dim as f64, out_dim as f64);
        let weight_f32 = match self {
            LinearInit::XavierUniform => {
                let bound = (6.0f64 / (fan_in + fan_out)).sqrt();
                Tensor::rand(-bound as f32, bound as f32, shape, device)?
            }
            LinearInit::XavierNormal => {
                let std = (2.0f64 / (fan_in + fan_out)).sqrt();
                Tensor::randn(0f32, std as f32, shape, device)?
            }
            LinearInit::KaimingUniform { negative_slope } => {
                let gain = (2.0f64 / (1.0 + negative_slope.powi(2))).sqrt();
                let bound = (3.0f64).sqrt() * gain / fan_in.sqrt();
                Tensor::rand(-bound as f32, bound as f32, shape, device)?
            }
            LinearInit::KaimingNormal { negative_slope } => {
                let gain = (2.0f64 / (1.0 + negative_slope.powi(2))).sqrt();
                let std = gain / fan_in.sqrt();
                Tensor::randn(0f32, std as f32, shape, device)?
            }
            LinearInit::Scaled { base, scale } => {
                let sampled = base.sample(shape, device, DType::F32)?;
                let scale_tensor = Tensor::full(*scale as f32, (), device)?;
                sampled.broadcast_mul(&scale_tensor)?
            }
        };
        if dtype == DType::F32 {
            Ok(weight_f32)
        } else {
            checks::ensure_cast_supported("linear.init", DType::F32, dtype)?;
            weight_f32.to_dtype(dtype)
        }
    }
}

/// Dense affine projection with optional bias and mixed-precision aware forward pass.
#[derive(Debug, Clone)]
pub struct Linear {
    config: LinearConfig,
    weight: Arc<Mutex<Tensor>>,
    bias: Option<Arc<Mutex<Tensor>>>,
}

impl Linear {
    /// Constructs a linear layer from pre-existing parameters.
    pub fn new(config: LinearConfig, weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::validate_weight(&config, &weight)?;
        Self::validate_bias(&config, bias.as_ref())?;
        Ok(Self {
            config,
            weight: Arc::new(Mutex::new(weight)),
            bias: bias.map(|b| Arc::new(Mutex::new(b))),
        })
    }

    /// Builds a linear layer with randomly initialised weights following `init`.
    pub fn with_init(
        config: LinearConfig,
        init: &LinearInit,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let weight = init.sample((config.total_output_dim(), config.input_dim), device, dtype)?;
        let bias = if config.bias {
            Some(Tensor::zeros(config.total_output_dim(), dtype, device)?)
        } else {
            None
        };
        Self::new(config, weight, bias)
    }

    /// Creates a new `Linear` that shares the weight (and optional bias) storage with `source`.
    pub fn tied(config: LinearConfig, source: &Linear) -> Result<Self> {
        if config.input_dim != source.config.input_dim
            || config.total_output_dim() != source.config.total_output_dim()
        {
            return Err(Error::Msg(
                "tied linear requires matching input/output dimensions".into(),
            ));
        }
        if config.bias != source.config.bias {
            return Err(Error::Msg(
                "tied linears must agree on bias usage to share storage".into(),
            ));
        }
        Ok(Self {
            config,
            weight: source.weight.clone(),
            bias: source.bias.as_ref().map(Arc::clone),
        })
    }

    /// Returns a clone of the underlying weight tensor.
    pub fn weight(&self) -> Tensor {
        self.weight.lock().unwrap().clone()
    }

    /// Returns a clone of the bias tensor if present.
    pub fn bias(&self) -> Option<Tensor> {
        self.bias.as_ref().map(|bias| bias.lock().unwrap().clone())
    }

    /// Copies `value` into the shared weight storage (useful for weight tying scenarios).
    pub fn copy_weight_from(&mut self, value: &Tensor) -> Result<()> {
        Self::validate_weight(&self.config, value)?;
        let mut weight = self.weight.lock().unwrap();
        checks::expect_same_dtype("linear.weight", value, "linear.weight", &*weight)?;
        let cast = value.to_dtype(weight.dtype())?;
        *weight = cast;
        Ok(())
    }

    /// Copies `value` into the shared bias storage.
    pub fn copy_bias_from(&mut self, value: &Tensor) -> Result<()> {
        match (&self.bias, value) {
            (Some(existing), input) => {
                Self::validate_bias(&self.config, Some(input))?;
                let mut bias = existing.lock().unwrap();
                checks::expect_same_dtype("linear.bias", input, "linear.bias", &*bias)?;
                let cast = input.to_dtype(bias.dtype())?;
                *bias = cast;
                Ok(())
            }
            (None, _) => Err(Error::Msg("layer has no bias to copy into".into())),
        }
    }

    fn validate_weight(config: &LinearConfig, weight: &Tensor) -> Result<()> {
        checks::expect_rank("linear.weight", weight, 2)?;
        checks::expect_shape(
            "linear.weight",
            weight,
            &[config.total_output_dim(), config.input_dim],
        )?;
        checks::expect_dtype_in(
            "linear.weight",
            weight,
            &[DType::F16, DType::BF16, DType::F32],
        )?;
        checks::expect_contiguous("linear.weight", weight)?;
        Ok(())
    }

    fn validate_bias(config: &LinearConfig, bias: Option<&Tensor>) -> Result<()> {
        match (config.bias, bias) {
            (true, Some(tensor)) => {
                checks::expect_rank("linear.bias", tensor, 1)?;
                checks::expect_shape("linear.bias", tensor, &[config.total_output_dim()])?;
                checks::expect_dtype_in(
                    "linear.bias",
                    tensor,
                    &[DType::F16, DType::BF16, DType::F32],
                )?;
                checks::expect_contiguous("linear.bias", tensor)?;
                Ok(())
            }
            (false, Some(_)) => Err(Error::Msg("bias provided but config disables bias".into())),
            (true, None) => Err(Error::Msg("config expects bias but none supplied".into())),
            (false, None) => Ok(()),
        }
    }

    fn validate_input(&self, hidden: &Tensor) -> Result<()> {
        let dims = hidden.dims();
        match dims {
            [batch, seq, hidden_dim] => {
                checks::expect_batch_seq_hidden("linear.input", hidden, self.config.input_dim)?;
                if *hidden_dim != self.config.input_dim {
                    Err(Error::Msg(format!(
                        "expected last dim {} but received {}",
                        self.config.input_dim, hidden_dim
                    )))
                } else if *batch == 0 || *seq == 0 {
                    Err(Error::Msg("batch/seq dimensions must be non-zero".into()))
                } else {
                    Ok(())
                }
            }
            [_, hidden_dim] => {
                if *hidden_dim != self.config.input_dim {
                    Err(Error::Msg(format!(
                        "expected last dim {} but received {}",
                        self.config.input_dim, hidden_dim
                    )))
                } else {
                    Ok(())
                }
            }
            _ => Err(Error::Msg(
                "linear expects input shaped [B, T, H_in] or [T, H_in]".into(),
            )),
        }
    }
}

impl LinearLayer for Linear {
    fn config(&self) -> &LinearConfig {
        &self.config
    }

    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        self.validate_input(hidden)?;

        let input = policy.cast_for_matmul(hidden)?;
        let weight = {
            let guard = self.weight.lock().unwrap();
            policy.cast_for_matmul(&*guard)?
        };
        let weight_t = weight.t()?;
        let dims = input.dims();

        let mut output = match dims {
            [batch, seq, _] => {
                let flat = input.reshape((*batch * *seq, self.config.input_dim))?;
                let proj = flat.matmul(&weight_t)?;
                proj.reshape((*batch, *seq, self.config.total_output_dim()))?
            }
            [rows, _] => input
                .matmul(&weight_t)?
                .reshape((*rows, self.config.total_output_dim()))?,
            _ => unreachable!("validated above"),
        };

        if let Some(bias) = &self.bias {
            let bias = {
                let guard = bias.lock().unwrap();
                policy.cast_for_matmul(&*guard)?
            };
            output = output.broadcast_add(&bias)?;
        }

        policy.cast_to_storage(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::PrecisionPolicy;
    use candle_core::{DType, Device};

    fn reference_linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let weight_t = weight.t()?;
        let dims = input.dims();
        let mut out = match dims {
            [batch, seq, hidden] => {
                let flat = input.reshape((*batch * *seq, *hidden))?;
                flat.matmul(&weight_t)?
                    .reshape((*batch, *seq, weight.dims()[0]))?
            }
            [rows, _hidden] => input
                .matmul(&weight_t)?
                .reshape((*rows, weight.dims()[0]))?,
            _ => unreachable!(),
        };
        if let Some(bias) = bias {
            out = out.broadcast_add(bias)?;
        }
        Ok(out)
    }

    fn tensor_stats(tensor: &Tensor) -> Result<(f64, f64)> {
        let values = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mean = values.iter().copied().map(f64::from).sum::<f64>() / values.len() as f64;
        let var = values
            .iter()
            .copied()
            .map(|v| {
                let diff = f64::from(v) - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;
        Ok((mean, var.sqrt()))
    }

    #[test]
    fn forward_matches_reference_across_dtypes() -> Result<()> {
        let device = Device::Cpu;
        let config = LinearConfig {
            input_dim: 8,
            output_dim: 4,
            bias: true,
            fused_projections: 3,
        };
        let weight = Tensor::randn(
            0f32,
            0.05,
            (config.total_output_dim(), config.input_dim),
            &device,
        )?;
        let bias = Tensor::randn(0f32, 0.02, config.total_output_dim(), &device)?;

        for &dtype in &[DType::F32, DType::F16, DType::BF16] {
            let linear = Linear::new(
                config.clone(),
                weight.to_dtype(dtype)?,
                Some(bias.to_dtype(dtype)?),
            )?;
            let input =
                Tensor::randn(0f32, 1.0, (2, 5, config.input_dim), &device)?.to_dtype(dtype)?;
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);
            let output = linear.forward(&input, &policy)?;

            assert_eq!(output.dims(), &[2, 5, config.total_output_dim()]);
            assert_eq!(output.dtype(), dtype);

            let reference = reference_linear(&input.to_dtype(DType::F32)?, &weight, Some(&bias))?;
            let diff = output
                .to_dtype(DType::F32)?
                .sub(&reference)?
                .abs()?
                .max_all()?;
            let tol = match dtype {
                DType::F16 => 1e-2,
                DType::BF16 => 2e-2,
                _ => 1e-4,
            };
            let max = diff.to_vec0::<f32>()?;
            assert!(max <= tol, "max diff {} for {:?}", max, dtype);
        }

        Ok(())
    }

    #[test]
    fn glorot_normal_stats_are_reasonable() -> Result<()> {
        let device = Device::Cpu;
        let config = LinearConfig::new(128, 64);
        let linear = Linear::with_init(config, &LinearInit::XavierNormal, &device, DType::F32)?;
        let (mean, std) = tensor_stats(&linear.weight())?;
        let expected = (2.0f64 / (128.0f64 + 64.0f64)).sqrt();
        assert!(mean.abs() < 5e-3);
        assert!((std - expected).abs() < expected * 0.25);
        Ok(())
    }

    #[test]
    fn kaiming_uniform_respects_scale() -> Result<()> {
        let device = Device::Cpu;
        let config = LinearConfig::new(256, 256);
        let init = LinearInit::scaled(
            LinearInit::KaimingUniform {
                negative_slope: 0.0,
            },
            0.5,
        );
        let linear = Linear::with_init(config, &init, &device, DType::F32)?;
        let (_, std) = tensor_stats(&linear.weight())?;
        let base_std = (2.0f64 / 256.0f64).sqrt();
        let expected = base_std * 0.5;
        assert!((std - expected).abs() < expected * 0.25);
        Ok(())
    }

    #[test]
    fn weight_tying_reflects_updates() -> Result<()> {
        let device = Device::Cpu;
        let config = LinearConfig::new(16, 16);
        let init = LinearInit::XavierUniform;
        let mut base = Linear::with_init(config.clone(), &init, &device, DType::F32)?;
        let tied = Linear::tied(config.clone(), &base)?;

        let new_weight = Tensor::full(
            0.25f32,
            (config.total_output_dim(), config.input_dim),
            &device,
        )?;
        base.copy_weight_from(&new_weight)?;

        let input = Tensor::randn(0f32, 1.0, (1, 3, config.input_dim), &device)?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let out_base = base.forward(&input, &policy)?;
        let out_tied = tied.forward(&input, &policy)?;
        let diff = out_base
            .to_dtype(DType::F32)?
            .sub(&out_tied.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?;
        assert!(diff.to_vec0::<f32>()? <= 1e-6);
        Ok(())
    }
}
