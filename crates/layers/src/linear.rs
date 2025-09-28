//! Linear and affine projection helpers.
//!
//! Linear layers expect inputs shaped `(batch, seq, in_dim)` and return tensors
//! with `(batch, seq, out_dim)`. Multi-projection variants pack the output as
//! `(batch, seq, num_projections * head_dim)` so that caller controlled reshapes
//! can split them for attention or gated activations. We consistently cast
//! weights and activations to [`PrecisionPolicy::compute`] for matmuls and rely
//! on [`PrecisionPolicy::cast_to_storage`](crate::dtypes::PrecisionPolicy::cast_to_storage)
//! for the final dtype.

use candle_core::{Result, Tensor};

use crate::dtypes::PrecisionPolicy;

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
}

/// Shared interface for affine projections.
pub trait LinearLayer: Send + Sync {
    /// Returns the static configuration used to validate inputs.
    fn config(&self) -> &LinearConfig;

    /// Applies the linear projection, promoting to the compute dtype when needed.
    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor>;
}
