# Transformer (Rust)

A generative pre-trained transformer implemented with [Candle](https://github.com/huggingface/candle).

## Usage

1. Place your training corpus at `pt-data/input.txt` (Shakespeare is bundled).
2. Run training with a preset that fits your hardware:

```
cargo run --release --features metal -- --preset=light
```

Available presets:

- `light` *(default)* – tuned for Apple M-series laptops (smaller context, batch 32-48)
- `balanced` – larger width/context while still feasible on consumer GPUs
- `max` – pushes width/context and batch size; expect higher VRAM and longer runs

During training you’ll see losses, perplexity, throughput, and live text previews.
Checkpoints are written to `checkpoints/iter_XXXXX.safetensors` so you can resume
or analyse later.

After training an interactive REPL launches. Sampling uses `max_tokens=200`,
`temperature=0.2`, `top-k=10`, and `top-p=0.9` by default; edit `src/main.rs`
if you want different defaults.
