//! Builders for padding masks used to drop padded keys.
//!
//! All padding masks share the dtype and layout described in
//! [`super::MASK_DTYPE`](super::MASK_DTYPE).

use candle_core::{Device, Result, Tensor};

/// Construct padding masks from per-batch valid key lengths.
pub fn padding_mask_from_lengths(
    device: &Device,
    key_lengths: &[usize],
    num_heads: usize,
    q_len: usize,
    k_len: usize,
) -> Result<Tensor> {
    let batch = key_lengths.len();
    let total = batch * num_heads * q_len * k_len;
    let mut data = vec![0f32; total];

    for (b, &valid) in key_lengths.iter().enumerate() {
        let valid = valid.min(k_len);
        for h in 0..num_heads {
            for q in 0..q_len {
                let row_start = ((((b * num_heads) + h) * q_len) + q) * k_len;
                for k in valid..k_len {
                    data[row_start + k] = f32::NEG_INFINITY;
                }
            }
        }
    }

    Tensor::from_vec(data, (batch, num_heads, q_len, k_len), device)
}

/// Construct padding masks from boolean padding indicators.
///
/// Each inner slice corresponds to a batch element and must share the same
/// length. `true` indicates a padded (masked) key position.
pub fn padding_mask_from_booleans(
    device: &Device,
    padding: &[Vec<bool>],
    num_heads: usize,
    q_len: usize,
) -> Result<Tensor> {
    if padding.is_empty() {
        return Tensor::zeros((0, num_heads, q_len, 0), super::MASK_DTYPE, device);
    }

    let k_len = padding[0].len();
    for mask in padding.iter() {
        assert_eq!(
            mask.len(),
            k_len,
            "all boolean padding masks must share k_len"
        );
    }

    let batch = padding.len();
    let total = batch * num_heads * q_len * k_len;
    let mut data = vec![0f32; total];

    for (b, mask) in padding.iter().enumerate() {
        for h in 0..num_heads {
            for q in 0..q_len {
                let row_start = ((((b * num_heads) + h) * q_len) + q) * k_len;
                for (k, &is_padding) in mask.iter().enumerate() {
                    if is_padding {
                        data[row_start + k] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    Tensor::from_vec(data, (batch, num_heads, q_len, k_len), device)
}
