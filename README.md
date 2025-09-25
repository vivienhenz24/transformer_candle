# Cascade Transformer

A transformer built on Rust/[Candle](https://github.com/huggingface/candle), because we do a little trolling ;)

(there is no rational reason for building a transformer using rust, except that it would be fun)

## Workspace Overview

```
├── cli/           # `cascade-cli`: binary entrypoint wiring everything together
├── core/          # `cascade-core`: model architecture, attention, generation, memory
├── tokenization/  # `transformer-tokenization`: adaptive BPE tokenizer pipeline
├── training/      # `cascade-training`: single-device trainer, optimisers, schedulers
└── utils/         # shared utilities (prompt templates)
```
## Usage

1. Download the compressed dump `enwiki-latest-pages-articles-multistream.xml.bz2` into `pt-data/`.
   The binary will stream the archive, clean the markup, and cache a
   `*-clean.txt` file automatically (override with `WIKI_MAX_ARTICLES` to limit
   preprocessing).
2. Build and run with a preset tuned to your hardware:

```bash
cargo run -p cascade-cli --release --features metal -- --preset=light
```

Available presets:

- `light` *(default)* – trimmed width/context for CPUs or integrated GPUs.
- `balanced` – medium cascade stack, suited for modern consumer GPUs.
- `max` – wider model and longer context; expect higher VRAM demand.

After training finishes, an interactive REPL launches. Generation uses
progressive refinement with adaptive sampling—edit `cli/src/main.rs` if you'd like
alternative creative modes or sampling defaults. The system prompt template
remains in `utils/src/prompts.rs`.

> **Metal note:** Candle’s Metal backend is still evolving. On start-up the
> binary runs a small preflight check; if the GPU kernels misbehave the program
> automatically falls back to CPU to keep training reliable. Set
> `CANDLE_FORCE_CPU=1` to skip Metal detection entirely.

## Tests

Run the full suite (workspace-wide):

```bash
cargo test
```

This exercises the attention modules, tokenizer pipeline, training helpers, and
entry-point smoke tests.
