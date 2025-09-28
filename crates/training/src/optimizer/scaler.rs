use candle_core::{DType, Tensor};

use crate::{config::Precision, TrainingError};

#[derive(Debug, Clone)]
pub struct LossScaleConfig {
    pub initial_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
    pub min_scale: f32,
    pub max_scale: f32,
}

impl Default for LossScaleConfig {
    fn default() -> Self {
        Self {
            initial_scale: 2f32.powi(15),
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 200,
            min_scale: 1.0,
            max_scale: 2f32.powi(24),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradientScaler {
    state: ScalerState,
}

#[derive(Debug, Clone)]
enum ScalerState {
    Disabled,
    Enabled(EnabledState),
}

#[derive(Debug, Clone)]
struct EnabledState {
    loss_scale: f32,
    stable_steps: usize,
    config: LossScaleConfig,
}

impl GradientScaler {
    pub fn new(precision: Precision) -> Self {
        Self::with_config(LossScaleConfig::default(), precision)
    }

    pub fn with_config(config: LossScaleConfig, precision: Precision) -> Self {
        let enabled = matches!(
            precision,
            Precision::Fp16 | Precision::Bf16 | Precision::Mixed
        );
        if !enabled {
            return Self {
                state: ScalerState::Disabled,
            };
        }

        let cfg = sanitize_config(config);
        let state = EnabledState {
            loss_scale: cfg.initial_scale,
            stable_steps: 0,
            config: cfg,
        };
        Self {
            state: ScalerState::Enabled(state),
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self.state, ScalerState::Enabled(_))
    }

    pub fn loss_scale(&self) -> f32 {
        match &self.state {
            ScalerState::Disabled => 1.0,
            ScalerState::Enabled(state) => state.loss_scale,
        }
    }

    pub fn scale(&self, tensor: &Tensor) -> Result<Tensor, TrainingError> {
        match &self.state {
            ScalerState::Disabled => Ok(tensor.clone()),
            ScalerState::Enabled(state) => tensor
                .affine(state.loss_scale as f64, 0.0)
                .map_err(to_runtime_error),
        }
    }

    pub fn unscale(&self, tensor: &Tensor) -> Result<Tensor, TrainingError> {
        match &self.state {
            ScalerState::Disabled => Ok(tensor.clone()),
            ScalerState::Enabled(state) => {
                let scale = 1.0 / state.loss_scale;
                tensor.affine(scale as f64, 0.0).map_err(to_runtime_error)
            }
        }
    }

    pub fn has_overflow<I>(&self, tensors: I) -> Result<bool, TrainingError>
    where
        I: IntoIterator,
        I::Item: AsRef<Tensor>,
    {
        match &self.state {
            ScalerState::Disabled => Ok(false),
            ScalerState::Enabled(_) => {
                for tensor in tensors {
                    if contains_non_finite(tensor.as_ref())? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    pub fn update(&mut self, found_inf: bool) {
        if let ScalerState::Enabled(state) = &mut self.state {
            if found_inf {
                state.loss_scale =
                    (state.loss_scale * state.config.backoff_factor).max(state.config.min_scale);
                state.stable_steps = 0;
            } else {
                state.stable_steps += 1;
                if state.stable_steps >= state.config.growth_interval {
                    state.loss_scale =
                        (state.loss_scale * state.config.growth_factor).min(state.config.max_scale);
                    state.stable_steps = 0;
                }
            }
        }
    }
}

fn contains_non_finite(tensor: &Tensor) -> Result<bool, TrainingError> {
    if tensor.elem_count() == 0 {
        return Ok(false);
    }
    let sum = tensor
        .to_dtype(DType::F32)
        .map_err(to_runtime_error)?
        .sqr()
        .map_err(to_runtime_error)?
        .sum_all()
        .map_err(to_runtime_error)?
        .to_vec0::<f32>()
        .map_err(to_runtime_error)?;
    Ok(!sum.is_finite())
}

fn sanitize_config(mut config: LossScaleConfig) -> LossScaleConfig {
    if config.growth_factor < 1.0 {
        config.growth_factor = 1.0;
    }
    if !(0.0..1.0).contains(&config.backoff_factor) {
        config.backoff_factor = 0.5;
    }
    if config.growth_interval == 0 {
        config.growth_interval = 1;
    }
    if config.min_scale <= 0.0 {
        config.min_scale = 1.0;
    }
    if config.max_scale < config.min_scale {
        config.max_scale = config.min_scale;
    }
    config.initial_scale = config
        .initial_scale
        .clamp(config.min_scale, config.max_scale);
    config
}

fn to_runtime_error(err: candle_core::Error) -> TrainingError {
    TrainingError::runtime(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn tensor_from(data: &[f32]) -> Tensor {
        Tensor::from_slice(data, (data.len(),), &Device::Cpu).unwrap()
    }

    #[test]
    fn grows_after_interval() {
        let mut scaler = GradientScaler::with_config(
            LossScaleConfig {
                initial_scale: 512.0,
                growth_interval: 2,
                ..LossScaleConfig::default()
            },
            Precision::Bf16,
        );

        assert!(scaler.is_enabled());
        assert_eq!(scaler.loss_scale(), 512.0);
        scaler.update(false);
        assert_eq!(scaler.loss_scale(), 512.0);
        scaler.update(false);
        assert_eq!(scaler.loss_scale(), 1024.0);
    }

    #[test]
    fn backs_off_on_infinite() {
        let mut scaler = GradientScaler::with_config(
            LossScaleConfig {
                initial_scale: 1024.0,
                backoff_factor: 0.25,
                ..LossScaleConfig::default()
            },
            Precision::Fp16,
        );

        scaler.update(true);
        assert_eq!(scaler.loss_scale(), 256.0);
    }

    #[test]
    fn detects_non_finite_gradients() {
        let scaler = GradientScaler::new(Precision::Mixed);
        let finite = tensor_from(&[1.0, -3.0]);
        let overflow = tensor_from(&[f32::INFINITY]);
        assert!(!scaler.has_overflow([&finite]).unwrap());
        assert!(scaler.has_overflow([&overflow]).unwrap());
    }

    #[test]
    fn no_op_for_fp32() {
        let scaler = GradientScaler::new(Precision::Fp32);
        assert!(!scaler.is_enabled());
        assert_eq!(scaler.loss_scale(), 1.0);

        let tensor = tensor_from(&[2.0, 4.0]);
        let scaled = scaler.scale(&tensor).unwrap();
        let unscaled = scaler.unscale(&tensor).unwrap();

        assert_eq!(scaled.to_vec1::<f32>().unwrap(), vec![2.0, 4.0]);
        assert_eq!(unscaled.to_vec1::<f32>().unwrap(), vec![2.0, 4.0]);
    }
}
