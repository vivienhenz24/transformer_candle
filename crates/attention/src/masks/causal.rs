//! Builders for causal attention masks.
//!
//! The resulting tensors have dtype [`MASK_DTYPE`](super::MASK_DTYPE) and shape
//! `[batch, num_heads, q_len, k_len]`. Entries are `0.0` where attention is
//! permitted and `f32::NEG_INFINITY` otherwise.

use candle_core::{Device, Result, Tensor};

/// Construct a causal mask for the supplied sequence dimensions.
///
/// When `k_len > q_len`, queries are assumed to align with the most recent
/// `q_len` keys, allowing access to the extended prefix.
pub fn build_causal_mask(
    device: &Device,
    batch: usize,
    num_heads: usize,
    q_len: usize,
    k_len: usize,
) -> Result<Tensor> {
    let total = batch * num_heads * q_len * k_len;
    let mut data = vec![0f32; total];

    let offset = k_len.saturating_sub(q_len);

    for b in 0..batch {
        for h in 0..num_heads {
            for q in 0..q_len {
                let row_start = ((((b * num_heads) + h) * q_len) + q) * k_len;
                let max_k = q + offset;
                for k in 0..k_len {
                    if k > max_k {
                        data[row_start + k] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    Tensor::from_vec(data, (batch, num_heads, q_len, k_len), device)
}
