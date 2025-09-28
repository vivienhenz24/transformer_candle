use candle_core::{DType, Tensor, D};
use candle_nn::ops;

use crate::TrainingError;

/// Cross entropy loss with optional label smoothing and ignore-index handling.
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    label_smoothing: f32,
    ignore_index: Option<u32>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_label_smoothing(mut self, smoothing: f32) -> Result<Self, TrainingError> {
        if !(0.0..1.0).contains(&smoothing) {
            return Err(TrainingError::initialization(
                "label smoothing must be in [0, 1) for cross entropy",
            ));
        }
        self.label_smoothing = smoothing;
        Ok(self)
    }

    pub fn with_ignore_index(mut self, ignore_index: Option<u32>) -> Self {
        self.ignore_index = ignore_index;
        self
    }

    pub fn compute(&self, logits: &Tensor, targets: &Tensor) -> Result<LossOutput, TrainingError> {
        let dims = logits.dims();
        if dims.len() < 2 {
            return Err(TrainingError::runtime(
                "cross entropy expects logits with at least two dimensions",
            ));
        }

        let vocab_size = *dims
            .last()
            .ok_or_else(|| TrainingError::runtime("logits tensor missing vocabulary dimension"))?;
        if vocab_size == 0 {
            return Err(TrainingError::runtime(
                "logits vocabulary dimension must be greater than zero",
            ));
        }

        let token_dims = &dims[..dims.len() - 1];
        if targets.dims() != token_dims {
            return Err(TrainingError::runtime(
                "target tensor must match logits batch/sequence dimensions",
            ));
        }

        let token_count: usize = token_dims.iter().copied().product();
        if token_count == 0 {
            return Err(TrainingError::runtime(
                "no tokens available for loss computation",
            ));
        }

        let device = logits.device();
        let logits_flat = logits
            .reshape((token_count, vocab_size))
            .map_err(to_runtime_error)?;

        let log_probs = ops::log_softmax(&logits_flat, D::Minus1).map_err(to_runtime_error)?;

        let targets_on_device = targets.to_device(device).map_err(to_runtime_error)?;
        let targets_flat = targets_on_device
            .reshape((token_count,))
            .map_err(to_runtime_error)?;
        let targets_flat = match targets_flat.dtype() {
            DType::U32 => targets_flat,
            DType::I64 | DType::U8 => targets_flat
                .to_dtype(DType::U32)
                .map_err(to_runtime_error)?,
            dtype => {
                return Err(TrainingError::runtime(format!(
                    "unsupported target dtype {:?} for cross entropy",
                    dtype
                )))
            }
        };

        let valid_mask = if let Some(ignore_index) = self.ignore_index {
            targets_flat
                .ne(ignore_index)
                .map_err(to_runtime_error)?
                .to_dtype(DType::F32)
                .map_err(to_runtime_error)?
        } else {
            Tensor::ones((token_count,), DType::F32, device).map_err(to_runtime_error)?
        };

        let total_tokens_scalar = valid_mask
            .sum_all()
            .map_err(to_runtime_error)?
            .to_vec0::<f32>()
            .map_err(to_runtime_error)?;
        let total_tokens = total_tokens_scalar.round() as usize;
        if total_tokens == 0 {
            return Err(TrainingError::runtime(
                "no valid tokens remain after applying ignore_index",
            ));
        }

        let target_indices = targets_flat.unsqueeze(1).map_err(to_runtime_error)?;
        let nll = log_probs
            .gather(&target_indices, 1)
            .map_err(to_runtime_error)?
            .neg()
            .map_err(to_runtime_error)?
            .squeeze(1)
            .map_err(to_runtime_error)?;

        let per_token_loss = if self.label_smoothing > 0.0 {
            let smoothing = self.label_smoothing;
            let uniform = log_probs
                .mean(1)
                .map_err(to_runtime_error)?
                .neg()
                .map_err(to_runtime_error)?;
            let smoothed = nll
                .affine((1.0 - smoothing) as f64, 0.0)
                .map_err(to_runtime_error)?;
            let uniform_term = uniform
                .affine(smoothing as f64, 0.0)
                .map_err(to_runtime_error)?;
            (smoothed + uniform_term).map_err(to_runtime_error)?
        } else {
            nll
        };

        let weighted_loss = (&per_token_loss * &valid_mask).map_err(to_runtime_error)?;
        let loss_sum = weighted_loss.sum_all().map_err(to_runtime_error)?;
        let average_loss = loss_sum
            .affine(1f64 / total_tokens as f64, 0.0)
            .map_err(to_runtime_error)?;

        let average_loss_value = average_loss.to_vec0::<f32>().map_err(to_runtime_error)?;

        let predictions = logits_flat.argmax(D::Minus1).map_err(to_runtime_error)?;
        let correct = predictions
            .eq(&targets_flat)
            .map_err(to_runtime_error)?
            .to_dtype(DType::F32)
            .map_err(to_runtime_error)?;
        let correct = (&correct * &valid_mask).map_err(to_runtime_error)?;
        let correct_tokens = correct
            .sum_all()
            .map_err(to_runtime_error)?
            .to_vec0::<f32>()
            .map_err(to_runtime_error)?
            .round() as usize;

        Ok(LossOutput {
            loss: average_loss,
            metrics: LossMetrics {
                average_loss: average_loss_value,
                total_tokens,
                correct_tokens,
            },
        })
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {
            label_smoothing: 0.0,
            ignore_index: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LossOutput {
    pub loss: Tensor,
    pub metrics: LossMetrics,
}

#[derive(Debug, Clone)]
pub struct LossMetrics {
    average_loss: f32,
    total_tokens: usize,
    correct_tokens: usize,
}

impl LossMetrics {
    pub fn average_loss(&self) -> f32 {
        self.average_loss
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn correct_tokens(&self) -> usize {
        self.correct_tokens
    }

    pub fn accuracy(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.correct_tokens as f32 / self.total_tokens as f32
        }
    }

    pub fn perplexity(&self) -> f32 {
        self.average_loss.exp()
    }
}

fn to_runtime_error(err: candle_core::Error) -> TrainingError {
    TrainingError::runtime(err.to_string())
}
