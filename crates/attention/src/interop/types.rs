//! Shared type aliases related to external integrations.

use candle_core::DType;

/// Logical description of attention tensor shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionShape {
    /// Batch size for the attention invocation.
    pub batch: usize,
    /// Number of heads processed in parallel.
    pub num_heads: usize,
    /// Sequence length (queries and keys) measured in tokens.
    pub seq_len: usize,
    /// Dimensionality per-head.
    pub head_dim: usize,
}

/// Permitted data types for attention tensors.
pub type AttentionDType = DType;
