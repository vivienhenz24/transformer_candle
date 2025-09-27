//! Mask utilities shared by attention implementations.
//!
//! All masks produced here are additive tensors with dtype `f32`, shaped
//! `[batch, num_heads, q_len, k_len]`. Values are either `0.0` (keep) or
//! `f32::NEG_INFINITY` (discard) to align with Candle's softmax-friendly
//! masking behaviour.

pub mod blocked;
pub mod causal;
pub mod padding;

use candle_core::DType;

/// Dtype shared by all additive masks.
pub const MASK_DTYPE: DType = DType::F32;

pub use blocked::build_blocked_causal_mask;
pub use causal::build_causal_mask;
pub use padding::{padding_mask_from_booleans, padding_mask_from_lengths};

#[cfg(test)]
mod tests;
