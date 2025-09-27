//! Benchmark plan for RoPE rotation and cache performance.
//!
//! Matrix under test:
//! - Sequence lengths: {1, 256, 4_096, 32_768}
//! - Head dimensions: {64, 128}
//! - Number of heads: {8, 32}
//! - Scaling modes: `RopeScaling::None`, `RopeScaling::PositionInterpolation { scale: 8.0 }`,
//!   and (guarded behind the `long-context` feature) `RopeScaling::NTKAware { alpha: 8.0 }`.
//!
//! Benchmark objectives to verify once implementations exist:
//! - After cache warm-up, the steady-state path performs zero additional heap allocations
//!   (tracked via allocation counters or custom allocators).
//! - Measured throughput scales approximately linearly with `seq_len` and `n_heads`; any
//!   super-linear behaviour must be justified.
//! - Cold-start timings report cache build latency separately from rotation throughput
//!   to attribute costs accurately.
//!
//! Skeleton placeholders below keep `cargo bench` aware of the upcoming suite without providing
//! executable benchmarks yet.

#![cfg_attr(not(test), allow(dead_code))]

#[cfg(test)]
mod rope_bench_plan {
    // TODO: Implement Criterion benchmarks covering the matrix above once RoPE is wired up.
}
