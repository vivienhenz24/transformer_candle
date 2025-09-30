//! Residual connections, dropout scheduling, and block orchestration helpers.
//!
//! Residual branches combine tensors of shape `(batch, seq, hidden)` and assume
//! consistent dtype policies across inputs. Implementations should promote to
//! [`PrecisionPolicy::compute`] before addition when mixed precision is in play
//! and use [`PrecisionPolicy::reduction`] for statistics such as dropout masks.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

use candle_core::{DType, Error, Result, Tensor};

use crate::{checks, dtypes::PrecisionPolicy};

/// Configuration describing how residual blocks are wired.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualConfig {
    /// Apply layer normalisation before residual addition (pre-norm block).
    pub prenorm: bool,
    /// Dropout probability applied to the transformed branch during training.
    pub dropout_p: Option<f32>,
    /// Optional residual scaling factor (e.g. for deep residual stabilisation).
    pub residual_scale: Option<f32>,
}

impl ResidualConfig {
    /// Creates a configuration with no dropout and no scaling.
    pub fn new(prenorm: bool) -> Self {
        Self {
            prenorm,
            dropout_p: None,
            residual_scale: None,
        }
    }
}

/// Interface for modules that merge a transformed branch with a residual path.
pub trait ResidualBlock: Send + Sync {
    /// Returns the configuration describing how the block is wired.
    fn config(&self) -> &ResidualConfig;

    /// Combines `input` with `residual` using the provided precision policy.
    fn forward(
        &self,
        input: &Tensor,
        residual: &Tensor,
        policy: &PrecisionPolicy,
    ) -> Result<Tensor>;
}

/// Dropout policy used throughout the residual stack.
#[derive(Debug)]
pub enum DropoutMode {
    /// Dropout is disabled (e.g. during evaluation or when probability is zero).
    Disabled,
    /// Dropout is active and uses the supplied probability and RNG seed.
    Enabled { probability: f32, rng: Mutex<Lcg64> },
}

impl Clone for DropoutMode {
    fn clone(&self) -> Self {
        match self {
            DropoutMode::Disabled => DropoutMode::Disabled,
            DropoutMode::Enabled { probability, rng } => {
                let state = match rng.lock() {
                    Ok(guard) => guard.clone(),
                    Err(_poisoned) => {
                        // If the mutex is poisoned, we can't recover the state
                        // Create a new Lcg64 with a default seed
                        Lcg64::new(0)
                    }
                };
                DropoutMode::Enabled {
                    probability: *probability,
                    rng: Mutex::new(state),
                }
            }
        }
    }
}

impl DropoutMode {
    /// Builds a mode from an optional probability; `None` or `0.0` disables dropout.
    pub fn from_probability(probability: Option<f32>, seed: u64) -> Self {
        match probability.unwrap_or(0.0) {
            p if p <= 0.0 => DropoutMode::Disabled,
            p if p >= 1.0 => DropoutMode::Disabled,
            p => DropoutMode::Enabled {
                probability: p,
                rng: Mutex::new(Lcg64::new(seed)),
            },
        }
    }
}

fn apply_dropout(tensor: &Tensor, mode: &DropoutMode, policy: &PrecisionPolicy) -> Result<Tensor> {
    match mode {
        DropoutMode::Disabled => Ok(tensor.clone()),
        DropoutMode::Enabled { probability, rng } => {
            let keep_prob = 1.0 - probability;
            let dims = tensor.dims();
            if dims.len() != 3 {
                return Err(Error::Msg(format!(
                    "dropout expects (batch, seq, hidden) tensor, got {:?}",
                    dims
                )));
            }
            checks::expect_batch_seq_hidden("dropout.input", tensor, dims[2])?;
            let device = tensor.device();
            let dtype = policy.compute();
            let total = tensor.elem_count();
            let mut rng = rng
                .lock()
                .map_err(|_| Error::Msg("dropout RNG mutex poisoned".into()))?;
            let mut mask_data = Vec::with_capacity(total);
            for _ in 0..total {
                let sample = rng.next_f32();
                mask_data.push(if sample < keep_prob { 1.0f32 } else { 0.0f32 });
            }
            checks::ensure_cast_supported("dropout.mask", DType::F32, dtype)?;
            let mask = Tensor::from_vec(mask_data, dims.to_vec(), &device)?.to_dtype(dtype)?;
            checks::ensure_cast_supported("dropout.scale", DType::F32, dtype)?;
            let scale = Tensor::full(1.0f32 / keep_prob, (), &device)?.to_dtype(dtype)?;
            let compute = policy.cast_for_matmul(tensor)?;
            let dropped = compute.broadcast_mul(&mask)?.broadcast_mul(&scale)?;
            policy.cast_to_storage(&dropped)
        }
    }
}

/// Residual add helper with optional scaling and dropout.
pub struct Residual {
    config: ResidualConfig,
    dropout: DropoutMode,
    training: AtomicBool,
}

impl Clone for Residual {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            dropout: self.dropout.clone(),
            training: AtomicBool::new(self.training.load(Ordering::Relaxed)),
        }
    }
}

impl fmt::Debug for Residual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Residual")
            .field("config", &self.config)
            .field("dropout", &self.dropout)
            .finish()
    }
}

impl Residual {
    /// Creates a residual helper with a deterministic dropout seed (used during tests).
    pub fn new(config: ResidualConfig, seed: u64) -> Self {
        let dropout = DropoutMode::from_probability(config.dropout_p, seed);
        Self {
            config,
            dropout,
            training: AtomicBool::new(true),
        }
    }

    /// Applies dropout to `branch` using the configured policy.
    pub fn apply_dropout(&self, branch: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) {
            return Ok(branch.clone());
        }
        apply_dropout(branch, &self.dropout, policy)
    }

    /// Overrides the dropout mode (useful for evaluation or tests).
    pub fn set_dropout_mode(&mut self, mode: DropoutMode) {
        self.dropout = mode;
    }

    /// Enables or disables dropout based on training mode.
    pub fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    /// Adds `branch` to `residual`, applying scaling when requested.
    pub fn add(
        &self,
        branch: &Tensor,
        residual: &Tensor,
        policy: &PrecisionPolicy,
    ) -> Result<Tensor> {
        let expected = residual.dims();
        checks::expect_batch_seq_hidden("residual.input", residual, expected[2])?;
        checks::expect_shape("residual.branch", branch, expected)?;
        checks::expect_same_dtype("residual.branch", branch, "residual.input", residual)?;

        let branch = policy.cast_for_matmul(branch)?;
        let residual = policy.cast_for_matmul(residual)?;
        let branch = if let Some(scale) = self.config.residual_scale {
            checks::ensure_cast_supported("residual.scale", DType::F32, branch.dtype())?;
            let scale_tensor =
                Tensor::full(scale, (), branch.device())?.to_dtype(branch.dtype())?;
            branch.broadcast_mul(&scale_tensor)?
        } else {
            branch
        };
        let added = branch.add(&residual)?;
        policy.cast_to_storage(&added)
    }

    /// Pre-norm residual helper (norm -> branch -> dropout -> add).
    pub fn prenorm_step(
        &self,
        normed_input: &Tensor,
        residual_input: &Tensor,
        policy: &PrecisionPolicy,
    ) -> Result<Tensor> {
        let after_dropout = self.apply_dropout(normed_input, policy)?;
        self.add(&after_dropout, residual_input, policy)
    }

    /// Post-norm residual helper (branch -> dropout -> add -> norm).
    pub fn postnorm_step(
        &self,
        branch_output: &Tensor,
        residual_input: &Tensor,
        policy: &PrecisionPolicy,
    ) -> Result<Tensor> {
        let after_dropout = self.apply_dropout(branch_output, policy)?;
        self.add(&after_dropout, residual_input, policy)
    }
}

/// Simple 64-bit linear congruential generator for deterministic dropout masks.
#[derive(Debug, Clone)]
pub(crate) struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Parameters from Numerical Recipes.
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        let bits = self.next_u64() >> 11;
        (bits as f64 * SCALE) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn policy(dtype: DType) -> PrecisionPolicy {
        PrecisionPolicy::from_parameter_dtype(dtype)
    }

    #[test]
    fn residual_add_preserves_shape_and_dtype() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F16;
        let residual_cfg = ResidualConfig {
            prenorm: false,
            dropout_p: None,
            residual_scale: None,
        };
        let residual = Residual::new(residual_cfg, 0);
        let left = Tensor::randn(0f32, 1.0, (2, 4, 8), &device)?.to_dtype(dtype)?;
        let right = Tensor::randn(0f32, 1.0, (2, 4, 8), &device)?.to_dtype(dtype)?;
        let out = residual.add(&left, &right, &policy(dtype))?;
        assert_eq!(out.dims(), &[2, 4, 8]);
        assert_eq!(out.dtype(), dtype);
        Ok(())
    }

    #[test]
    fn residual_scaling_is_applied() -> Result<()> {
        let device = Device::Cpu;
        let config = ResidualConfig {
            prenorm: false,
            dropout_p: None,
            residual_scale: Some(0.5),
        };
        let residual = Residual::new(config, 0);
        let branch = Tensor::ones((1, 1, 4), DType::F32, &device)?;
        let parent = Tensor::zeros((1, 1, 4), DType::F32, &device)?;
        let out = residual.add(&branch, &parent, &policy(DType::F32))?;
        let values = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(values.iter().all(|v| (*v - 0.5).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn dropout_respects_probability_and_seed() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = ResidualConfig {
            prenorm: false,
            dropout_p: Some(0.25),
            residual_scale: None,
        };
        let residual = Residual::new(config, 123);
        let input = Tensor::ones((4, 8, 16), dtype, &device)?;
        let policy = policy(dtype);
        let dropped = residual.apply_dropout(&input, &policy)?;

        let values = dropped.flatten_all()?.to_vec1::<f32>()?;
        let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
        assert!((mean - 1.0).abs() < 0.1);
        Ok(())
    }

    #[test]
    fn dropout_disabled_in_inference() -> Result<()> {
        let device = Device::Cpu;
        let config = ResidualConfig {
            prenorm: false,
            dropout_p: Some(0.5),
            residual_scale: None,
        };
        let residual = Residual::new(config, 0);
        residual.set_training(false);

        let input = Tensor::randn(0f32, 1.0, (2, 2, 4), &device)?;
        let out = residual.apply_dropout(&input, &policy(DType::F32))?;
        let diff = input.sub(&out)?.abs()?.max_all()?.to_vec0::<f32>()?;
        assert!(diff < 1e-7);
        Ok(())
    }

    #[test]
    fn prenorm_and_postnorm_helpers_work() -> Result<()> {
        let device = Device::Cpu;
        let policy = policy(DType::F32);
        let cfg = ResidualConfig {
            prenorm: true,
            dropout_p: None,
            residual_scale: None,
        };
        let residual = Residual::new(cfg, 0);
        let branch = Tensor::full(0.2f32, (1, 1, 3), &device)?;
        let parent = Tensor::full(1.0f32, (1, 1, 3), &device)?;
        let pre = residual.prenorm_step(&branch, &parent, &policy)?;
        let post = residual.postnorm_step(&branch, &parent, &policy)?;
        let pre_vals = pre.flatten_all()?.to_vec1::<f32>()?;
        let post_vals = post.flatten_all()?.to_vec1::<f32>()?;
        assert!(pre_vals.iter().all(|v| (*v - 1.2).abs() < 1e-6));
        assert!(post_vals.iter().all(|v| (*v - 1.2).abs() < 1e-6));
        Ok(())
    }
}
