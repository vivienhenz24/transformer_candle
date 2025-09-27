use candle_core::{Device, Result, Tensor};
use embedding::positional::rope::{
    apply_rope_to_qk, get_sin_cos, reset_sin_cos_cache_stats, sin_cos_cache_counters, RopeConfig,
    RopeScaling,
};

fn main() {
    println!("Transformer started");
    pretraining_data::init();

    let _rope_config = RopeConfig::default();
    println!(
        "RoPE module placeholder ready (head_dim = {}, theta = {})",
        _rope_config.head_dim, _rope_config.rope_theta
    );

    run_rope_demo();
}

/// Minimal RoPE demo executed from the binary.
fn run_rope_demo() {
    println!("--- RoPE demo ---");

    if let Err(err) = rope_demo_inner() {
        eprintln!("RoPE demo encountered an error: {err}");
    }
}

fn rope_demo_inner() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let heads = 2;
    let seq_len = 8;
    let head_dim = 16;
    let rotate_dim = 8;

    let cfg = RopeConfig {
        head_dim,
        rope_theta: 10_000.0,
        rotate_dim: Some(rotate_dim),
        scaling: RopeScaling::None,
    };

    let total = batch * heads * seq_len * head_dim;
    let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();
    let k_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.03 - 2.0).collect();

    let q = Tensor::from_vec(q_data.clone(), (batch, heads, seq_len, head_dim), &device)?;
    let k = Tensor::from_vec(k_data, (batch, heads, seq_len, head_dim), &device)?;

    reset_sin_cos_cache_stats();

    let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
    let (q_rot, k_rot) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos)?;
    let (hits_first, _misses_first) = sin_cos_cache_counters();

    let flatten =
        |tensor: &Tensor| -> Result<Vec<f32>> { Ok(tensor.flatten_all()?.to_vec1::<f32>()?) };

    let q_main_before = flatten(&q.narrow(3, 0, rotate_dim)?)?;
    let q_main_after = flatten(&q_rot.narrow(3, 0, rotate_dim)?)?;
    let mut sum_sq = 0.0f32;
    for (before, after) in q_main_before.iter().zip(q_main_after.iter()) {
        let diff = after - before;
        sum_sq += diff * diff;
    }
    let l2_delta = sum_sq.sqrt();

    let tail_equal = if rotate_dim < head_dim {
        let tail_before = flatten(&q.narrow(3, rotate_dim, head_dim - rotate_dim)?)?;
        let tail_after = flatten(&q_rot.narrow(3, rotate_dim, head_dim - rotate_dim)?)?;
        tail_before == tail_after
    } else {
        true
    };

    // Trigger a cache hit by requesting the same geometry again.
    let (_sin_again, _cos_again) = get_sin_cos(seq_len, &cfg, &device);
    let (hits_second, misses_second) = sin_cos_cache_counters();
    let cache_hit_second_call = hits_second > hits_first;

    println!(
        "rotate_dim={} l2_delta_first_slice={:.6} tail_preserved={} cache_hits={} cache_misses={} second_call_hit={}",
        rotate_dim,
        l2_delta,
        tail_equal,
        hits_second,
        misses_second,
        cache_hit_second_call
    );

    // Prevent unused variable warnings for the rotated keys in demo context.
    let _ = k_rot;

    Ok(())
}
