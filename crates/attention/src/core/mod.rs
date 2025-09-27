//! Core traits and types shared across attention implementations.
//!
//! Implementations are expected to operate on tensors with layout
//! `[batch, n_heads, seq_len, head_dim]`. The output tensor mirrors the input
//! layout, and reductions should accumulate in `f32` regardless of the incoming
//! dtype (`bf16`, `f16`, or `f32`).

pub mod config;
pub mod errors;

use candle_core::Tensor;

pub use config::Config;
pub use errors::AttentionError;

/// Unified interface for attention kernels.
///
/// * `q`, `k`, and `v` share the layout `[batch, n_heads, seq_len, head_dim]`.
/// * The returned tensor mirrors the layout and dtype of `q`.
/// * Masks, when present, must be shaped `[batch, 1 or n_heads, q_len, k_len]`.
/// * Reductions are performed in `f32`; other dtypes are converted as needed.
/// * Dropout is controlled via [`Config::dropout_p`] and is only applied during
///   training.
pub trait Attention {
    /// Compute causal self-attention with an optional additive mask.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        config: &Config,
    ) -> Result<Tensor, AttentionError>;
}
