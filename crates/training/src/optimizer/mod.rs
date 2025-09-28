use std::collections::HashMap;

pub mod scaler;

pub use scaler::{GradientScaler, GradientScalerState, LossScaleConfig};

use candle_core::{backprop::GradStore, DType, Tensor, Var};
use serde::{Deserialize, Serialize};

use crate::{config, TrainingError};

const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    AdamW(AdamWConfig),
}

#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl TryFrom<&config::OptimizerConfig> for OptimizerConfig {
    type Error = TrainingError;

    fn try_from(value: &config::OptimizerConfig) -> Result<Self, Self::Error> {
        match value.algorithm {
            config::OptimizerType::AdamW => Ok(OptimizerConfig::AdamW(AdamWConfig {
                learning_rate: value.learning_rate as f64,
                beta1: value.beta1 as f64,
                beta2: value.beta2 as f64,
                epsilon: value.epsilon as f64,
                weight_decay: value.weight_decay as f64,
            })),
            other => Err(TrainingError::initialization(format!(
                "unsupported optimizer algorithm {:?}",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainerOptimizerOptions {
    pub use_master_weights: bool,
    pub clip_global_norm: Option<f64>,
    pub clip_per_parameter: Option<f64>,
    pub weight_decay_exclude: Vec<String>,
}

impl Default for TrainerOptimizerOptions {
    fn default() -> Self {
        Self {
            use_master_weights: true,
            clip_global_norm: None,
            clip_per_parameter: None,
            weight_decay_exclude: vec![],
        }
    }
}

#[derive(Debug)]
pub struct TrainerOptimizer {
    config: OptimizerConfig,
    params: Vec<ParameterSlot>,
    step: usize,
    clip_global_norm: Option<f64>,
    clip_per_parameter: Option<f64>,
}

#[derive(Debug)]
struct ParameterSlot {
    name: String,
    param: Var,
    dtype: DType,
    master: Option<Var>,
    first_moment: Tensor,
    second_moment: Tensor,
    apply_weight_decay: bool,
}

impl TrainerOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        named_parameters: Vec<(String, Var)>,
        config: OptimizerConfig,
        options: TrainerOptimizerOptions,
    ) -> Result<Self, TrainingError> {
        if named_parameters.is_empty() {
            return Err(TrainingError::initialization(
                "optimizer requires at least one parameter",
            ));
        }

        let mut params = Vec::with_capacity(named_parameters.len());
        for (name, var) in named_parameters {
            let tensor = var.as_tensor();
            if !tensor.dtype().is_float() {
                return Err(TrainingError::initialization(format!(
                    "optimizer received non-floating parameter '{}'",
                    name
                )));
            }
            let device = tensor.device();
            let shape = tensor.dims().to_vec();
            let dtype = tensor.dtype();

            let first_moment =
                Tensor::zeros(shape.as_slice(), DType::F32, device).map_err(to_runtime_error)?;
            let second_moment =
                Tensor::zeros(shape.as_slice(), DType::F32, device).map_err(to_runtime_error)?;

            let apply_weight_decay =
                should_apply_weight_decay(&name, &options.weight_decay_exclude);

            let master = if options.use_master_weights && dtype != DType::F32 {
                let fp32 = tensor.to_dtype(DType::F32).map_err(to_runtime_error)?;
                Some(Var::from_tensor(&fp32).map_err(to_runtime_error)?)
            } else {
                None
            };

            params.push(ParameterSlot {
                name,
                param: var,
                dtype,
                master,
                first_moment,
                second_moment,
                apply_weight_decay,
            });
        }

        Ok(Self {
            config,
            params,
            step: 0,
            clip_global_norm: options.clip_global_norm,
            clip_per_parameter: options.clip_per_parameter,
        })
    }

    pub fn learning_rate(&self) -> f64 {
        match self.config {
            OptimizerConfig::AdamW(cfg) => cfg.learning_rate,
        }
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        match &mut self.config {
            OptimizerConfig::AdamW(cfg) => cfg.learning_rate = lr,
        }
    }

    pub fn step(&mut self, grads: &mut GradStore) -> Result<(), TrainingError> {
        let mut processed = Vec::new();

        for (idx, slot) in self.params.iter().enumerate() {
            let tensor = slot.param.as_tensor();
            let grad = match grads.remove(tensor) {
                Some(grad) => grad,
                None => continue,
            };

            let mut grad = grad.to_dtype(DType::F32).map_err(to_runtime_error)?;

            let mut norm = tensor_l2_norm(&grad)?;

            if let Some(max_norm) = self.clip_per_parameter {
                if norm > max_norm {
                    let scale = (max_norm / (norm + EPS)) as f64;
                    grad = grad.affine(scale, 0.0).map_err(to_runtime_error)?;
                    norm = max_norm;
                }
            }

            processed.push(ProcessedGradient {
                index: idx,
                grad,
                norm,
            });
        }

        if processed.is_empty() {
            return Ok(());
        }

        if let Some(max_norm) = self.clip_global_norm {
            let total_norm_sq: f64 = processed.iter().map(|g| g.norm * g.norm).sum();
            let total_norm = total_norm_sq.sqrt();
            if total_norm > max_norm {
                let scale = max_norm / (total_norm + EPS);
                for item in &mut processed {
                    item.grad = item
                        .grad
                        .affine(scale as f64, 0.0)
                        .map_err(to_runtime_error)?;
                    item.norm *= scale;
                }
            }
        }

        self.step += 1;
        match self.config {
            OptimizerConfig::AdamW(cfg) => {
                self.step_adamw(cfg, processed)?;
            }
        }

        Ok(())
    }

    fn step_adamw(
        &mut self,
        cfg: AdamWConfig,
        processed: Vec<ProcessedGradient>,
    ) -> Result<(), TrainingError> {
        let bias_correction1 = 1.0 - cfg.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - cfg.beta2.powi(self.step as i32);
        let scale_m = if bias_correction1.abs() < EPS {
            1.0
        } else {
            1.0 / bias_correction1
        };
        let scale_v = if bias_correction2.abs() < EPS {
            1.0
        } else {
            1.0 / bias_correction2
        };

        for item in processed {
            let slot = &mut self.params[item.index];
            let beta1 = cfg.beta1;
            let beta2 = cfg.beta2;

            let prev_m = slot
                .first_moment
                .affine(beta1, 0.0)
                .map_err(to_runtime_error)?;
            let grad_term = item
                .grad
                .clone()
                .affine(1.0 - beta1, 0.0)
                .map_err(to_runtime_error)?;
            let new_m = prev_m.add(&grad_term).map_err(to_runtime_error)?;

            let grad_sq = item.grad.sqr().map_err(to_runtime_error)?;
            let prev_v = slot
                .second_moment
                .affine(beta2, 0.0)
                .map_err(to_runtime_error)?;
            let grad_sq_term = grad_sq.affine(1.0 - beta2, 0.0).map_err(to_runtime_error)?;
            let new_v = prev_v.add(&grad_sq_term).map_err(to_runtime_error)?;

            let m_hat = new_m.affine(scale_m, 0.0).map_err(to_runtime_error)?;
            let v_hat = new_v.affine(scale_v, 0.0).map_err(to_runtime_error)?;
            let denom = v_hat
                .sqrt()
                .map_err(to_runtime_error)?
                .affine(1.0, cfg.epsilon)
                .map_err(to_runtime_error)?;
            let update = m_hat
                .div(&denom)
                .map_err(to_runtime_error)?
                .affine(cfg.learning_rate, 0.0)
                .map_err(to_runtime_error)?;

            let base = if let Some(master) = slot.master.as_ref() {
                master.as_tensor().clone()
            } else {
                slot.param
                    .as_tensor()
                    .to_dtype(DType::F32)
                    .map_err(to_runtime_error)?
            };

            let decayed = if slot.apply_weight_decay && cfg.weight_decay != 0.0 {
                base.affine(1.0 - cfg.learning_rate * cfg.weight_decay, 0.0)
                    .map_err(to_runtime_error)?
            } else {
                base
            };

            let next = decayed.sub(&update).map_err(to_runtime_error)?;

            if let Some(master) = slot.master.as_ref() {
                master.set(&next).map_err(to_runtime_error)?;
                let cast = if slot.dtype == DType::F32 {
                    next
                } else {
                    next.to_dtype(slot.dtype).map_err(to_runtime_error)?
                };
                slot.param.set(&cast).map_err(to_runtime_error)?;
            } else {
                let cast = if slot.dtype == DType::F32 {
                    next
                } else {
                    next.to_dtype(slot.dtype).map_err(to_runtime_error)?
                };
                slot.param.set(&cast).map_err(to_runtime_error)?;
            }

            slot.first_moment = new_m;
            slot.second_moment = new_v;
        }

        Ok(())
    }

    pub fn zero_grad(&self, grads: &mut GradStore) {
        for slot in &self.params {
            let _ = grads.remove(slot.param.as_tensor());
        }
    }

    pub fn state(&self) -> Result<OptimizerState, TrainingError> {
        let mut parameters = Vec::with_capacity(self.params.len());
        for slot in &self.params {
            let shape = slot.param.as_tensor().dims().to_vec();
            let numel = numel(&shape);
            let first = flatten_to_vec(&slot.first_moment, numel)?;
            let second = flatten_to_vec(&slot.second_moment, numel)?;
            let master = if let Some(master) = &slot.master {
                Some(flatten_to_vec(master.as_tensor(), numel)?)
            } else {
                None
            };
            parameters.push(ParameterState {
                name: slot.name.clone(),
                shape,
                first_moment: first,
                second_moment: second,
                master,
            });
        }

        Ok(OptimizerState {
            step: self.step,
            parameters,
        })
    }

    pub fn load_state(&mut self, state: OptimizerState) -> Result<(), TrainingError> {
        self.step = state.step;
        let mut by_name: HashMap<_, _> = state
            .parameters
            .into_iter()
            .map(|param| (param.name.clone(), param))
            .collect();

        for slot in &mut self.params {
            let state = by_name.remove(&slot.name).ok_or_else(|| {
                TrainingError::runtime(format!("optimizer state missing parameter '{}'", slot.name))
            })?;

            let expected = numel(&slot.param.as_tensor().dims().to_vec());
            if slot.param.as_tensor().dims() != state.shape.as_slice() {
                return Err(TrainingError::runtime(format!(
                    "optimizer state shape mismatch for '{}'",
                    slot.name
                )));
            }
            if expected != state.first_moment.len()
                || expected != state.second_moment.len()
                || state.master.as_ref().map_or(false, |m| m.len() != expected)
            {
                return Err(TrainingError::runtime(format!(
                    "optimizer state size mismatch for '{}'",
                    slot.name
                )));
            }

            let device = slot.param.as_tensor().device();
            slot.first_moment = Tensor::from_vec(state.first_moment, expected, device)
                .map_err(to_runtime_error)?
                .reshape(slot.param.as_tensor().dims())
                .map_err(to_runtime_error)?;
            slot.second_moment = Tensor::from_vec(state.second_moment, expected, device)
                .map_err(to_runtime_error)?
                .reshape(slot.param.as_tensor().dims())
                .map_err(to_runtime_error)?;

            match (&mut slot.master, state.master) {
                (Some(master), Some(values)) => {
                    let tensor = Tensor::from_vec(values, expected, device)
                        .map_err(to_runtime_error)?
                        .reshape(master.as_tensor().dims())
                        .map_err(to_runtime_error)?;
                    master.set(&tensor).map_err(to_runtime_error)?;
                    let cast = if slot.dtype == DType::F32 {
                        tensor
                    } else {
                        tensor.to_dtype(slot.dtype).map_err(to_runtime_error)?
                    };
                    slot.param.set(&cast).map_err(to_runtime_error)?;
                }
                (None, None) => {}
                (Some(_), None) => {
                    return Err(TrainingError::runtime(format!(
                        "optimizer state missing master weights for '{}'",
                        slot.name
                    )))
                }
                (None, Some(_)) => {
                    return Err(TrainingError::runtime(format!(
                        "optimizer state contains master weights for '{}' but optimizer is not using them",
                        slot.name
                    )))
                }
            }
        }

        if !by_name.is_empty() {
            return Err(TrainingError::runtime(
                "optimizer state has extra parameters not present in the model",
            ));
        }

        Ok(())
    }
}

struct ProcessedGradient {
    index: usize,
    grad: Tensor,
    norm: f64,
}

fn should_apply_weight_decay(name: &str, exclusions: &[String]) -> bool {
    !exclusions
        .iter()
        .any(|pattern| matches_pattern(name, pattern))
}

fn matches_pattern(name: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    name.ends_with(pattern) || name.contains(pattern)
}

fn tensor_l2_norm(tensor: &Tensor) -> Result<f64, TrainingError> {
    let squared = tensor
        .sqr()
        .map_err(to_runtime_error)?
        .sum_all()
        .map_err(to_runtime_error)?;
    let value = squared.to_vec0::<f32>().map_err(to_runtime_error)?;
    Ok((value as f64).sqrt())
}

fn flatten_to_vec(tensor: &Tensor, expected: usize) -> Result<Vec<f32>, TrainingError> {
    let flat = tensor
        .flatten_all()
        .map_err(to_runtime_error)?
        .to_vec1::<f32>()
        .map_err(to_runtime_error)?;
    if flat.len() != expected {
        return Err(TrainingError::runtime(
            "unexpected element count during serialization",
        ));
    }
    Ok(flat)
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub parameters: Vec<ParameterState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterState {
    pub name: String,
    pub shape: Vec<usize>,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
    pub master: Option<Vec<f32>>,
}

fn to_runtime_error(err: candle_core::Error) -> TrainingError {
    TrainingError::runtime(err.to_string())
}
