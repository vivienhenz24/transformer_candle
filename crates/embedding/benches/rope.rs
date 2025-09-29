use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use embedding::positional::rope::{
    apply_rope_to_qk, get_sin_cos, reset_sin_cos_cache_stats, RopeConfig, RopeScaling,
};

fn build_inputs(
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let total = batch * heads * seq_len * head_dim;
    let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.002 - 1.0).collect();
    let q = Tensor::from_vec(q_data, (batch, heads, seq_len, head_dim), device)?;
    let k = Tensor::from_vec(k_data, (batch, heads, seq_len, head_dim), device)?;
    Ok((q, k))
}

fn rope_cold_vs_warm(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 1;
    let heads = 8;
    let seq_len = 4_096;
    let head_dim = 64;
    let rotate_dim = head_dim;
    let cfg = RopeConfig {
        head_dim,
        rope_theta: 10_000.0,
        rotate_dim: Some(rotate_dim),
        scaling: RopeScaling::None,
    };

    let (q, k) = build_inputs(batch, heads, seq_len, head_dim, &device).unwrap();

    c.bench_function("rope_cold_cache", |b| {
        b.iter(|| {
            reset_sin_cos_cache_stats();
            let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
            let (q_rot, k_rot) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos).unwrap();
            black_box(q_rot);
            black_box(k_rot);
        })
    });

    // Preload cache for warm benchmark and capture tensors once.
    reset_sin_cos_cache_stats();
    let start = Instant::now();
    let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
    let cold_warmup = start.elapsed();
    eprintln!(
        "rope warmup: seq_len={} rotate_dim={} time={:.2?}",
        seq_len, rotate_dim, cold_warmup
    );

    c.bench_function("rope_warm_cache", |b| {
        b.iter(|| {
            let (hits_before, misses_before) =
                embedding::positional::rope::sin_cos_cache_counters();
            let (q_rot, k_rot) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos).unwrap();
            black_box(&q_rot);
            black_box(&k_rot);
            let (hits_after, misses_after) = embedding::positional::rope::sin_cos_cache_counters();
            // Ensure no additional sin/cos allocations occur in the steady state.
            assert_eq!(misses_after, misses_before);
            assert_eq!(hits_after, hits_before);
        })
    });
}

criterion_group!(rope_benches, rope_cold_vs_warm);
criterion_main!(rope_benches);
