# RoPE Acceptance Criteria

This document captures the "done" conditions for the rotary positional embedding (RoPE) workstream. Once all items below are satisfied, the feature
should be ready for integration with attention and downstream consumers.

## API Stability
- [x] `RopeConfig`, `RopeScaling`, `get_sin_cos`, `apply_rope_to_qk`, `effective_positions`, and `scaling_fingerprint` remain public and backwards
  compatible.

## Correctness
- [x] All unit and integration tests in `crates/embedding` pass.
- [x] Vectorized rotation matches a naïve scalar reference within abs/rel tolerance `1e-5` for small tensors.
- [x] When `rotate_dim < head_dim`, tail dimensions are bitwise unchanged after applying RoPE.
- [x] The rotation helpers preserve the dtype of the input queries/keys (bf16/fp16/f32) while using f32 sin/cos tables.

## Performance
- [x] After cache warm-up, repeated `get_sin_cos` + `apply_rope_to_qk` invocations with `seq_len=4096` perform zero additional heap allocations in the
  steady state (validated via allocator hooks or instrumentation).
- [x] The sin/cos cache is bounded via LRU or similar eviction and does not grow unbounded.

## Long-Context Behaviour
- [x] With the `long-context` feature enabled, `PositionInterpolation { scale >= 8 }` yields smoke tests without NaNs/Infs.
- [ ] With the `long-context` feature enabled, `NTKAware { alpha >= 4 }` yields smoke tests at extended sequence lengths without NaNs/Infs.

## Documentation & Demo
- `docs/rope.md` describes RoPE usage, scaling, caching, and environment overrides.
- Environment variable overrides (`ROPE_MODE`, `ROPE_THETA`, `ROPE_ROTATE_DIM`, `ROPE_ALPHA`, `ROPE_PI_SCALE`) are honoured and logged via
  `scaling_fingerprint`.
- The `run_rope_demo` path executes end-to-end, prints norm checks, and verifies cache reuse.

## Future Work
- YaRN-style scaling and other advanced schemes remain optional follow-ups once the core implementation ships.
