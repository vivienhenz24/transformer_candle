use candle_core::{DType, Device, Result, Tensor};
use embedding::positional::rope::{
    apply_rope_to_qk, effective_positions, get_sin_cos, reset_sin_cos_cache_stats,
    sin_cos_cache_counters, Rope, RopeCache, RopeConfig, RopeScaling,
};

fn assert_close(lhs: &[f32], rhs: &[f32], atol: f32, rtol: f32) {
    assert_eq!(lhs.len(), rhs.len(), "length mismatch");
    for (idx, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (l - r).abs();
        let tol = atol + r.abs() * rtol;
        assert!(
            diff <= tol,
            "values differ at index {idx}: left={l}, right={r}, diff={diff}, tol={tol}"
        );
    }
}

fn make_tensor(data: &[f32], shape: (usize, usize, usize, usize)) -> Result<Tensor> {
    Tensor::from_vec(data.to_vec(), shape, &Device::Cpu)
}

fn naive_rope_rotate(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    rotate_dim: usize,
    sin_vals: &[Vec<f32>],
    cos_vals: &[Vec<f32>],
) -> Vec<f32> {
    let mut out = data.to_vec();
    let half_dim = rotate_dim / 2;
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..seq_len {
                let base = (((b * heads) + h) * seq_len + t) * head_dim;
                for pair in 0..half_dim {
                    let even_idx = base + pair * 2;
                    let odd_idx = even_idx + 1;
                    let even = data[even_idx];
                    let odd = data[odd_idx];
                    let sin = sin_vals[t][pair];
                    let cos = cos_vals[t][pair];
                    out[even_idx] = even * cos - odd * sin;
                    out[odd_idx] = odd * cos + even * sin;
                }
            }
        }
    }
    out
}


fn tensor_to_flat_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn rope_types_are_send_sync() {
    assert_send_sync::<RopeConfig>();
    assert_send_sync::<Rope>();
    assert_send_sync::<RopeCache>();
}

#[test]
fn rope_scaling_defaults_to_none() {
    let scaling = RopeScaling::default();
    assert!(matches!(scaling, RopeScaling::None));
}

#[test]
fn naive_vs_vectorized_rotation_match() -> Result<()> {
    let device = Device::Cpu;
    let cfg = RopeConfig {
        head_dim: 8,
        rope_theta: 10_000.0,
        rotate_dim: Some(8),
        scaling: RopeScaling::None,
    };

    let batch = 1;
    let heads = 1;
    let seq_len = 4;
    let head_dim = cfg.head_dim;
    let total = batch * heads * seq_len * head_dim;

    let q_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2) - 3.0).collect();

    let q = make_tensor(&q_data, (batch, heads, seq_len, head_dim))?;
    let k = make_tensor(&k_data, (batch, heads, seq_len, head_dim))?;

    let (sin_all, cos_all) = get_sin_cos(seq_len, &cfg, &device);
    let (q_rot, k_rot) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin_all, &cos_all)?;

    assert!(q_rot.is_contiguous());
    assert!(k_rot.is_contiguous());
    assert_eq!(q_rot.dtype(), DType::F32);
    assert_eq!(k_rot.dtype(), DType::F32);

    let sin_slice = sin_all.narrow(0, 0, seq_len)?;
    let cos_slice = cos_all.narrow(0, 0, seq_len)?;
    let sin_vals = sin_slice.to_vec2::<f32>()?;
    let cos_vals = cos_slice.to_vec2::<f32>()?;

    let rotate_dim = cfg.rotate_dim.unwrap_or(cfg.head_dim);
    let q_naive = naive_rope_rotate(
        &q_data, batch, heads, seq_len, head_dim, rotate_dim, &sin_vals, &cos_vals,
    );
    let k_naive = naive_rope_rotate(
        &k_data, batch, heads, seq_len, head_dim, rotate_dim, &sin_vals, &cos_vals,
    );

    let q_vec = tensor_to_flat_vec(&q_rot)?;
    let k_vec = tensor_to_flat_vec(&k_rot)?;

    assert_close(&q_vec, &q_naive, 1e-5, 1e-5);
    assert_close(&k_vec, &k_naive, 1e-5, 1e-5);

    Ok(())
}

#[test]
fn tail_dimensions_unchanged() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 16;
    let rotate_dim = 8;
    let cfg = RopeConfig {
        head_dim,
        rope_theta: 10_000.0,
        rotate_dim: Some(rotate_dim),
        scaling: RopeScaling::None,
    };

    let batch = 1;
    let heads = 1;
    let seq_len = 4;
    let total = batch * heads * seq_len * head_dim;

    let mut q_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.01).collect();
    let mut k_data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.02) - 5.0).collect();

    // Stamp distinctive tail values to check passthrough.
    let tail_span = head_dim - rotate_dim;
    for offset in 0..(batch * heads * seq_len) {
        for t in 0..tail_span {
            let idx = offset * head_dim + rotate_dim + t;
            q_data[idx] = 1_000.0 + idx as f32;
            k_data[idx] = -1_000.0 - idx as f32;
        }
    }

    let q = make_tensor(&q_data, (batch, heads, seq_len, head_dim))?;
    let k = make_tensor(&k_data, (batch, heads, seq_len, head_dim))?;

    let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
    let (q_rot, k_rot) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos)?;

    let q_tail_before = tensor_to_flat_vec(&q.narrow(3, rotate_dim, tail_span)?)?;
    let q_tail_after = tensor_to_flat_vec(&q_rot.narrow(3, rotate_dim, tail_span)?)?;
    let k_tail_before = tensor_to_flat_vec(&k.narrow(3, rotate_dim, tail_span)?)?;
    let k_tail_after = tensor_to_flat_vec(&k_rot.narrow(3, rotate_dim, tail_span)?)?;

    assert_eq!(q_tail_before, q_tail_after);
    assert_eq!(k_tail_before, k_tail_after);

    // Ensure the rotated slice actually changed to catch degenerate behaviour.
    let q_main_before = tensor_to_flat_vec(&q.narrow(3, 0, rotate_dim)?)?;
    let q_main_after = tensor_to_flat_vec(&q_rot.narrow(3, 0, rotate_dim)?)?;
    assert!(q_main_before != q_main_after);

    Ok(())
}

#[test]
fn dtype_stability_across_inputs() -> Result<()> {
    let device = Device::Cpu;
    let cfg = RopeConfig {
        head_dim: 8,
        rope_theta: 10_000.0,
        rotate_dim: Some(8),
        scaling: RopeScaling::None,
    };

    let batch = 1;
    let heads = 1;
    let seq_len = 4;
    let head_dim = cfg.head_dim;
    let total = batch * heads * seq_len * head_dim;

    let base_q: Vec<f32> = (0..total).map(|i| (i as f32).sin()).collect();
    let base_k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.5).cos()).collect();

    let q_f32 = make_tensor(&base_q, (batch, heads, seq_len, head_dim))?;
    let k_f32 = make_tensor(&base_k, (batch, heads, seq_len, head_dim))?;

    let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
    let sin_vals = sin.narrow(0, 0, seq_len)?.to_vec2::<f32>()?;
    let cos_vals = cos.narrow(0, 0, seq_len)?.to_vec2::<f32>()?;
    let reference = naive_rope_rotate(
        &base_q,
        batch,
        heads,
        seq_len,
        head_dim,
        cfg.rotate_dim.unwrap_or(cfg.head_dim),
        &sin_vals,
        &cos_vals,
    );

    let configs = [
        (DType::F32, 1e-5f32),
        (DType::BF16, 3e-3f32),
        (DType::F16, 3e-3f32),
    ];

    for (dtype, tol) in configs {
        let q = q_f32.to_dtype(dtype)?;
        let k = k_f32.to_dtype(dtype)?;
        let (q_rot, _k_rot) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos)?;
        assert_eq!(q_rot.dtype(), dtype);

        let q_rot_f32 = q_rot.to_dtype(DType::F32)?;
        let values = tensor_to_flat_vec(&q_rot_f32)?;
        for (idx, v) in values.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite value at index {idx} for dtype {dtype:?}"
            );
        }
        assert_close(&values, &reference, tol, tol);
    }

    Ok(())
}

#[test]
fn scaling_modes_effective_positions() -> Result<()> {
    let pos_start = 1_024usize;
    let seq_len = 8usize;
    let train_ctx_len = 2_048usize;

    let base_cfg = RopeConfig {
        head_dim: 64,
        rope_theta: 10_000.0,
        rotate_dim: None,
        scaling: RopeScaling::None,
    };

    let positions_none = effective_positions(pos_start, seq_len, &base_cfg, train_ctx_len);
    let expected: Vec<u32> = (0..seq_len).map(|i| (pos_start + i) as u32).collect();
    assert_eq!(positions_none, expected);

    #[cfg(feature = "long-context")]
    {
        let pi_cfg = RopeConfig {
            scaling: RopeScaling::PositionInterpolation { scale: 8.0 },
            ..base_cfg.clone()
        };
        let positions_pi = effective_positions(pos_start, seq_len, &pi_cfg, train_ctx_len);
        let expected_pi: Vec<u32> = (0..seq_len)
            .map(|i| ((pos_start + i) as f64 / 8.0).floor() as u32)
            .collect();
        assert_eq!(positions_pi, expected_pi);
        assert!(positions_pi.windows(2).all(|w| w[0] <= w[1]));
        assert!(positions_pi.iter().all(|&p| (p as usize) < train_ctx_len));

        let ntk_cfg = RopeConfig {
            scaling: RopeScaling::NTKAware { alpha: 8.0 },
            ..base_cfg.clone()
        };
        let positions_ntk = effective_positions(pos_start, seq_len, &ntk_cfg, train_ctx_len);
        assert_eq!(positions_ntk, expected);
    }

    #[cfg(not(feature = "long-context"))]
    {
        // When long-context is disabled the additional variants are unavailable; ensure None still works.
        assert!(matches!(base_cfg.scaling, RopeScaling::None));
    }

    Ok(())
}

#[cfg(feature = "long-context")]
#[test]
fn position_interpolation_effective_positions() {
    let pos_start = 1_024usize;
    let seq_len = 8usize;
    let train_ctx_len = 2_048usize;

    let cfg = RopeConfig {
        head_dim: 64,
        rope_theta: 10_000.0,
        rotate_dim: None,
        scaling: RopeScaling::PositionInterpolation { scale: 8.0 },
    };

    let positions = effective_positions(pos_start, seq_len, &cfg, train_ctx_len);
    for (i, pos) in positions.iter().enumerate() {
        let expected = ((pos_start + i) as f64 / 8.0).floor() as u32;
        assert_eq!(
            *pos, expected,
            "position mismatch at offset {i}: got {pos}, expected {expected}"
        );
        assert!(
            (*pos as usize) < train_ctx_len,
            "position exceeds train context"
        );
    }
}

#[test]
fn cache_behavior_per_device() -> Result<()> {
    let device = Device::Cpu;
    let cfg = RopeConfig {
        head_dim: 8,
        rope_theta: 10_000.0,
        rotate_dim: Some(8),
        scaling: RopeScaling::None,
    };

    reset_sin_cos_cache_stats();

    let (hits_base, misses_base) = sin_cos_cache_counters();
    let (sin1, cos1) = get_sin_cos(8, &cfg, &device);
    let (hits_after_first, misses_after_first) = sin_cos_cache_counters();
    assert!(misses_after_first > misses_base || hits_after_first > hits_base);

    let (sin2, cos2) = get_sin_cos(8, &cfg, &device);
    let (hits_after_second, misses_after_second) = sin_cos_cache_counters();
    assert!(
        hits_after_second > hits_after_first,
        "expected cache hit on repeat access"
    );
    assert!(misses_after_second >= misses_after_first);

    // Ensure retrieved tensors carry identical data.
    assert_eq!(tensor_to_flat_vec(&sin1)?, tensor_to_flat_vec(&sin2)?);
    assert_eq!(tensor_to_flat_vec(&cos1)?, tensor_to_flat_vec(&cos2)?);

    let cfg_alt = RopeConfig {
        rotate_dim: Some(4),
        ..cfg.clone()
    };
    let _ = get_sin_cos(8, &cfg_alt, &device);
    let (hits_final, misses_final) = sin_cos_cache_counters();
    assert!(hits_final >= hits_after_second);
    assert!(
        misses_final > misses_after_second,
        "changing rotate_dim should trigger a cache miss"
    );

    Ok(())
}

#[test]
#[ignore = "long-context stress test pending implementation"]
fn plan_long_context_smoke_test() {
    // Scenario: `seq_len=128_000`, `head_dim=128`, `rotate_dim=64`, dtype=f16, executed on GPU when available.
    // Acceptance: ensure `apply_rope_to_qk` completes without allocating more than ~2x input tensor memory, produces no NaNs/Infs, and respects cache bounds.
    // Marked #[ignore] because it is heavyweight; run manually on CI with sufficient resources.
    todo!("implement long-context RoPE smoke test");
}
