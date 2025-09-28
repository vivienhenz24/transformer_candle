//! Residual connections, dropout scheduling, and block orchestration helpers.
//!
//! Residual branches combine tensors of shape `(batch, seq, hidden)` and assume
//! consistent dtype policies across inputs. Implementations should promote to
//! [`PrecisionPolicy::compute`] before addition when mixed precision is in play
//! and use [`PrecisionPolicy::reduction`] for statistics such as dropout masks.

use candle_core::{Result, Tensor};

use crate::dtypes::PrecisionPolicy;

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
