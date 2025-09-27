# Rotary Positional Embeddings (RoPE)

Rotary positional embeddings inject relative position information by rotating the query (`Q`) and key (`K`) vectors in attention. While the full
attention stack is not yet implemented in this workspace, the embedding crate already exposes the plumbing needed to prepare RoPE inputs and caches.

## Where RoPE Applies

RoPE operates on `Q` and `K` tensors shaped `[batch, n_heads, seq_len, head_dim]`. The helper API `apply_rope_to_qk` expects callers to supply
synthetic or real tensors alongside precomputed sine/cosine tables. Only the first `rotate_dim` channels of each head are rotated; the remaining
features pass through unchanged.

## Configuration Surface

`RopeConfig` describes how rotations should be computed:

- `head_dim`: size of the per-head embedding (e.g. 64 or 128).
- `rope_theta`: base frequency (`θ`) controlling the spacing between rotary angles.
- `rotate_dim`: optional override for how many dimensions to rotate. `None` means rotate the entire `head_dim`.
- `scaling`: optional scaling mode to support longer contexts.

### Scaling Modes

Scaling helps reuse models at inference lengths that exceed the training context (`train_ctx_len`).

- `None`: no scaling; positions are used exactly as provided.
- `PositionInterpolation { scale }`: effective positions are computed via `floor(position / scale)` and clamped into `[0, train_ctx_len)`.
- `NTKAware { alpha }`: leaves positions untouched but adjusts the frequency base derived from `rope_theta`. The chosen base must remain constant for the entire decode stream.

### Environment Overrides

Callers can customise RoPE without changing code by setting environment variables before constructing a `RopeConfig`:

| Variable          | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `ROPE_MODE`       | `none`, `ntk`, or `pi` to select the scaling variant.       |
| `ROPE_THETA`      | Overrides `rope_theta` (floating point).                    |
| `ROPE_ROTATE_DIM` | Overrides `rotate_dim` (usize; `0` maps to "rotate all").  |
| `ROPE_ALPHA`      | NTK-aware `alpha` when `ROPE_MODE=ntk`.                      |
| `ROPE_PI_SCALE`   | Position interpolation scale when `ROPE_MODE=pi`.           |

Implementations should resolve these values once, construct the `RopeConfig`, and log the resulting `scaling_fingerprint` (see below) so runs are
reproducible.

## Sin/Cos Cache

`get_sin_cos` returns f32 sine/cosine tables shaped `[max_seq_len, rotate_dim/2]`. The cache is keyed by the sequence geometry, scaling fingerprint,
`rope_theta`, and device identifier. Working in f32 keeps precomputation numerically stable; callers may downcast to bf16/fp16 at the point of use to
match the query/key tensors. The cache must be thread-safe, bounded (e.g. LRU), and should log whether requests were cache hits or misses.

## Effective Positions & Fingerprints

`effective_positions` computes scaled positions using the selected mode and the training context length (`train_ctx_len`).
`scaling_fingerprint` encodes the scaling variant, its parameters, and the training context. Logging the fingerprint once per run aids
post-mortem analysis without leaking implementation-specific details.

### Choosing Between PI and NTK Scaling

The two long-context strategies serve different purposes:

- **PositionInterpolation (PI)** compresses positional indices via `floor(position / scale)` while clamping to the training window. Use PI when
  you want to reuse a model verbatim at longer sequence lengths and are comfortable with the mild loss of resolution that comes from downscaling.
  PI is available behind the `long-context` feature flag and is enabled by selecting `ROPE_MODE=pi` (and an optional `ROPE_PI_SCALE`).
- **NTKAware (NTK)** keeps positions unchanged but adjusts the rotary base frequency derived from `rope_theta`. Reach for NTK when you would like to
  preserve per-token spacing and are ready to retune the effective theta/alpha pair for the target context.

## Demo Path

`main.rs` ships a minimal demo that exercises RoPE end-to-end.

### How to Run the Demo

```bash
cargo run
```

The program prints the usual start-up banner followed by a summary similar to:

```
rotate_dim=8 l2_delta_first_slice=52.708580 tail_preserved=true cache_hits=1 cache_misses=1 second_call_hit=true
```

This confirms that the leading `rotate_dim` slice changed (non-zero L2 delta), the tail remained bitwise identical, and the second lookup reused the
cached sin/cos tables (`second_call_hit=true`). Add `--features long-context` if you want to experiment with PI/NTK scaling in the demo or tests.

Once attention integration lands, these helpers will slot into the standard transformer attention mechanism without further changes.
