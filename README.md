# Cascade Transformer

A transformer built on Rust/[Candle](https://github.com/huggingface/candle), because we do a little trolling ;)

(there is no rational reason for building a transformer using rust, except that it would be fun)

## Workspace Overview

```
├── core/          # `cascade-core`: model architecture, attention, generation, memory
├── tokenization/  # `transformer-tokenization`: adaptive BPE tokenizer pipeline
├── training/      # `cascade-training`: single-device trainer, optimisers, schedulers
├── utils/         # shared utilities (prompt templates)
└── src/           # binary entrypoint wiring everything together
```
## Usage

1. Place your training corpus at `pt-data/input.txt` (a Shakespeare sample is bundled).
2. Build and run with a preset tuned to your hardware:

```bash
cargo run --release --features metal -- --preset=light
```

Available presets:

- `light` *(default)* – trimmed width/context for CPUs or integrated GPUs.
- `balanced` – medium cascade stack, suited for modern consumer GPUs.
- `max` – wider model and longer context; expect higher VRAM demand.

After training finishes, an interactive REPL launches. Generation uses
progressive refinement with adaptive sampling—edit `src/main.rs` if you'd like
alternative creative modes or sampling defaults. The system prompt template
remains in `utils/src/prompts.rs`.

## Tests

Run the full suite (workspace-wide):

```bash
cargo test
```

This exercises the attention modules, tokenizer pipeline, training helpers, and
entry-point smoke tests.
