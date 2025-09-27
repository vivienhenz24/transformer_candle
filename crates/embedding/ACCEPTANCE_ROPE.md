# RoPE Acceptance Criteria

This document captures the "done" conditions for the rotary positional embedding (RoPE) workstream. Once all items below are satisfied, the feature
should be ready for integration with attention and downstream consumers.

## API Stability
- `RopeConfig`, `RopeScaling`, `get_sin_cos`, `apply_rope_to_qk`, `effective_positions`, and `scaling_fingerprint` remain public and backwards
  compatible.

## Correctness
- All unit and integration tests in `crates/embedding` pass.
- Vectorized rotation matches a naïve scalar reference within abs/rel tolerance `1e-5` for small tensors.
- When `rotate_dim < head_dim`, tail dimensions are bitwise unchanged after applying RoPE.
- The rotation helpers preserve the dtype of the input queries/keys (bf16/fp16/f32) while using f32 sin/cos tables.

## Performance
- After cache warm-up, repeated `get_sin_cos` + `apply_rope_to_qk` invocations with `seq_len=4096` perform zero additional heap allocations in the
  steady state (validated via allocator hooks or instrumentation).
- The sin/cos cache is bounded via LRU or similar eviction and does not grow unbounded.

## Long-Context Behaviour
- With the `long-context` feature enabled, `PositionInterpolation { scale >= 8 }` and `NTKAware { alpha >= 4 }` yield smoke tests at `seq_len=32_768`
  without NaNs/Infs.

## Documentation & Demo
- `docs/rope.md` describes RoPE usage, scaling, caching, and environment overrides.
- Environment variable overrides (`ROPE_MODE`, `ROPE_THETA`, `ROPE_ROTATE_DIM`, `ROPE_ALPHA`, `ROPE_PI_SCALE`) are honoured and logged via
  `scaling_fingerprint`.
- The `run_rope_demo` path executes end-to-end, prints norm checks, and verifies cache reuse.

## Future Work
- YaRN-style scaling and other advanced schemes remain optional follow-ups once the core implementation ships.
