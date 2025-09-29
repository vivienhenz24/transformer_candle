//! Blocked attention mask helpers for windowed schemes.
//!
//! The blocked mask mirrors the layout rules from [`super::MASK_DTYPE`] and
//! produces step-wise causal coverage suitable for tiled kernels.

use candle_core::{Device, Result, Tensor};

/// Construct a block-structured causal mask with the provided block size.
pub fn build_blocked_causal_mask(
    device: &Device,
    batch: usize,
    num_heads: usize,
    q_len: usize,
    k_len: usize,
    block_size: usize,
) -> Result<Tensor> {
    let block = block_size.max(1);
    let total = batch * num_heads * q_len * k_len;
    let mut data = vec![0f32; total];

    let q_blocks = (q_len + block - 1) / block;
    let k_blocks = (k_len + block - 1) / block;
    let offset_blocks = if k_blocks > q_blocks {
        k_blocks - q_blocks
    } else {
        0
    };

    for b in 0..batch {
        for h in 0..num_heads {
            for q in 0..q_len {
                let q_block = q / block;
                let row_start = ((((b * num_heads) + h) * q_len) + q) * k_len;
                for k in 0..k_len {
                    let k_block = k / block;
                    if k_block > q_block + offset_blocks {
                        data[row_start + k] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    Tensor::from_vec(data, (batch, num_heads, q_len, k_len), device)
}
