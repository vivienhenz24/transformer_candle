# transformer_candle (Rust)

I built a transformer using rust/[Candle](https://github.com/huggingface/candle)..  Because why not ;)

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
