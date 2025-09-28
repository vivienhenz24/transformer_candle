use super::*;
use candle_core::{Device, Result};

fn idx(
    b: usize,
    h: usize,
    q: usize,
    k: usize,
    num_heads: usize,
    q_len: usize,
    k_len: usize,
) -> usize {
    ((((b * num_heads) + h) * q_len) + q) * k_len + k
}

#[test]
fn causal_mask_respects_offsets() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let num_heads = 2;
    let q_len = 3;
    let k_len = 5;

    let mask = build_causal_mask(&device, batch, num_heads, q_len, k_len)?;
    assert_eq!(mask.dims(), &[batch, num_heads, q_len, k_len]);

    let values = mask.flatten_all()?.to_vec1::<f32>()?;

    // Earliest query can only see the prefix (offset = k_len - q_len).
    assert_eq!(values[idx(0, 0, 0, 2, num_heads, q_len, k_len)], 0.0);
    assert_eq!(
        values[idx(0, 0, 0, 3, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );

    // Later queries gain access to more keys.
    assert_eq!(values[idx(0, 1, 2, 4, num_heads, q_len, k_len)], 0.0);

    Ok(())
}

#[test]
fn causal_mask_handles_single_token_cases() -> Result<()> {
    let device = Device::Cpu;

    let mask = build_causal_mask(&device, 1, 1, 1, 1)?;
    assert_eq!(mask.flatten_all()?.to_vec1::<f32>()?, vec![0.0]);

    let mask = build_causal_mask(&device, 1, 1, 1, 4)?;
    assert_eq!(
        mask.flatten_all()?.to_vec1::<f32>()?,
        vec![0.0, 0.0, 0.0, 0.0]
    );

    let mask = build_causal_mask(&device, 1, 1, 4, 2)?;
    let values = mask.flatten_all()?.to_vec1::<f32>()?;
    let dims = [1, 1, 4, 2];
    assert_eq!(mask.dims(), &dims);
    assert_eq!(values[idx(0, 0, 0, 1, 1, 4, 2)], f32::NEG_INFINITY);
    assert_eq!(values[idx(0, 0, 3, 1, 1, 4, 2)], 0.0);

    Ok(())
}

#[test]
fn padding_mask_from_lengths_masks_tail() -> Result<()> {
    let device = Device::Cpu;
    let lengths = [2, 5];
    let num_heads = 1;
    let q_len = 3;
    let k_len = 5;

    let mask = padding_mask_from_lengths(&device, &lengths, num_heads, q_len, k_len)?;
    assert_eq!(mask.dims(), &[lengths.len(), num_heads, q_len, k_len]);

    let values = mask.flatten_all()?.to_vec1::<f32>()?;
    // Batch 0 -> keys >=2 masked.
    assert_eq!(
        values[idx(0, 0, 0, 2, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );
    assert_eq!(values[idx(0, 0, 1, 1, num_heads, q_len, k_len)], 0.0);
    // Batch 1 -> length clamped to k_len, so no padding masked.
    assert_eq!(values[idx(1, 0, 2, 4, num_heads, q_len, k_len)], 0.0);

    Ok(())
}

#[test]
fn padding_mask_from_booleans_respects_flags() -> Result<()> {
    let device = Device::Cpu;
    let padding = vec![vec![false, true, false], vec![true, true, false]];
    let num_heads = 2;
    let q_len = 2;

    let mask = padding_mask_from_booleans(&device, &padding, num_heads, q_len)?;
    let k_len = padding[0].len();
    assert_eq!(mask.dims(), &[padding.len(), num_heads, q_len, k_len]);

    let values = mask.flatten_all()?.to_vec1::<f32>()?;
    // Batch 0, head 1, q 1, k 1 is padded.
    assert_eq!(
        values[idx(0, 1, 1, 1, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );
    // Batch 1, all first two keys masked across heads and queries.
    assert_eq!(
        values[idx(1, 0, 0, 0, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );
    assert_eq!(
        values[idx(1, 1, 1, 1, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );
    // Batch 1, final key remains available.
    assert_eq!(values[idx(1, 0, 0, 2, num_heads, q_len, k_len)], 0.0);

    Ok(())
}

#[test]
fn blocked_mask_enforces_block_level_causality() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let num_heads = 1;
    let q_len = 4;
    let k_len = 6;
    let block_size = 2;

    let mask = build_blocked_causal_mask(&device, batch, num_heads, q_len, k_len, block_size)?;
    assert_eq!(mask.dims(), &[batch, num_heads, q_len, k_len]);
    let values = mask.flatten_all()?.to_vec1::<f32>()?;

    // Query in block 0 should only see first two blocks (offset derived from extra k blocks).
    assert_eq!(values[idx(0, 0, 0, 3, num_heads, q_len, k_len)], 0.0);
    assert_eq!(
        values[idx(0, 0, 0, 4, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );

    // Query in block 1 can see all blocks.
    assert_eq!(values[idx(0, 0, 3, 5, num_heads, q_len, k_len)], 0.0);

    Ok(())
}

#[test]
fn composition_of_causal_and_padding_masks_respects_both() -> Result<()> {
    let device = Device::Cpu;
    let causal = build_causal_mask(&device, 1, 1, 3, 5)?;
    let padding = padding_mask_from_lengths(&device, &[3], 1, 3, 5)?;

    let combined = causal.add(&padding)?;
    let values = combined.flatten_all()?.to_vec1::<f32>()?;
    let num_heads = 1;
    let q_len = 3;
    let k_len = 5;

    // Padding should mask final keys even if causal would allow them.
    assert_eq!(
        values[idx(0, 0, 2, 4, num_heads, q_len, k_len)],
        f32::NEG_INFINITY
    );
    // Positions allowed by both remain unmasked.
    assert_eq!(values[idx(0, 0, 1, 1, num_heads, q_len, k_len)], 0.0);

    Ok(())
}
