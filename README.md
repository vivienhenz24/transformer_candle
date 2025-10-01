## transformer_candle

Hi! We built a transformer using Huggingface's candle. And no, there is absolutely no rational reason for building this in rust. Except that we thought it would be more fun (it wasn't).

The workspace is split into focused crates so you can train, evaluate, and repurpose components without touching unrelated code.

```text
                                ┌────────────────────────┐
                                │ configs/* & guides     │
                                │  (experiment recipes)  │
                                └────────────┬───────────┘
                                             │
                 ┌───────────────────────────┴──────────────────────────┐
                 │                   orchestrate CLI                   │
                 │ tokenization ▸ sharding ▸ config generation ▸ launch │
                 └───────────────┬──────────────────────────┬───────────┘
                                 │                          │
                      tokenizer crate                 pretraining-data crate
                    (byte-level BPE load/train)     (newline shards & HF streaming)
                                 │                          │
                                 └──────────┬───────────────┘
                                            │
                                 training crate (`train`)
                      config parsing ▸ data loader ▸ optimizer ▸ logs
                                            │
                                            v
                    model crate (decoder-only transformer with RoPE + caches)
                   attention ▸ embeddings ▸ layers ▸ gradient checkpoint toggle
                                            │
                                            v
                         checkpoints ▸ metrics ▸ downstream consumers
```

### Why You Might Like It
- Modular Candle crates: fused/reference attention, rotary-aware decoder blocks, reusable embedding layers, and shared linear/norm utilities.
- Training CLI with automatic CUDA/Metal/CPU choice, mixed-precision policy, gradient scaling, checkpoint rotation, and a gradient-checkpoint flag (currently a passthrough loop). 
- Data ingest from newline shards or on-demand Hugging Face streaming with optional local sharding to keep corpora manageable.
- Orchestration binary that can tokenize, shard, emit configs, and launch training for laptops or RunPod-backed cloud experiments.
- Byte-level BPE tokenizer support, including deterministic training when the `train` feature is enabled (used by the training crate).

### Workspace Tour
- `crates/attention` – fused/reference kernels, KV cache helpers, and RoPE-aware configs; tweak behaviour with `ATTN_*` environment variables.
- `crates/embedding` – token/positional embeddings with tied output heads.
- `crates/layers` – linear, activation, norm, residual, and dtype helpers shared across blocks.
- `crates/model` – decoder-only transformer that stitches embeddings, attention, and feed-forward layers with rotary support and mask/position caches.
- `crates/tokenizer` – config-driven byte-level BPE loader/trainer.
- `crates/pretraining-data` – newline shard streaming plus a Hugging Face dataset subprocess bridge.
- `crates/training` – `orchestrate` + `train` binaries, training loop, optimizer/scheduler, logging, metrics, and checkpoint management.
- `configs/` – experiment presets from local smoke tests to multi-billion parameter RunPod runs.
- `infra/` – RunPod manifests, full setup guide, and supporting docs.

### Launch a Local Run
1. Install Rust (`rustup default stable`) and ensure `cargo` is on your path.
2. Fetch dependencies: `cargo build --workspace`.
3. Generate artifacts via orchestrate:
   ```
   cargo run -p training --bin orchestrate -- \
     --mode local \
     --input crates/pretraining-data/input.txt \
     --run-dir runs/local-smoke
   ```
   You will get tokenizer assets, optional shards, `training.toml`, and `streaming_config.json` when streaming is toggled.
4. Train with the emitted config:
   ```
   cargo run -p training --bin train -- \
     --config runs/local-smoke/training.toml
   ```
   Add `--resume` to pick up the latest checkpoint or `--override key=value` to patch settings on the fly.

### Streaming & Cloud Notes
- Drop a `streaming_config.json` next to your training config (or let orchestrate create it) and `train` will fetch Hugging Face data into temporary shards before launching.
- Cloud mode (`--mode cloud`) swaps directory layouts to `/workspace/...`, shards corpora automatically, and wires RunPod-friendly checkpoint/log paths.
- Public checkpoints live at `https://huggingface.co/vivienhenz/sconce`; mirror them into `runs/pretrained` (or your chosen path) rather than committing weights.

### Configuration & Precision
- Training configs accept TOML/YAML/JSON. `runtime.precision` maps to Candle dtypes (BF16 by default) and enables master weights plus gradient scaling.
- Model hyperparameters resolve through `ModelOverrides`, with validation ensuring vocab/heads/ff-ratio sanity before the trainer touches the device.
- Gradient checkpointing flags and the attention mask/position caches help keep memory manageable for longer sequences.

### Develop Confidently
- Lint/format: `cargo fmt` and `cargo clippy --all-targets --workspace`.
- Test: `cargo test --workspace`; add `--features train` to exercise tokenizer training.
- Instrumentation: `RUST_LOG=info` for richer logs, `ATTN_BACKEND=fused|reference` to force attention kernels, and `cuda_is_available`/`metal_is_available` checks print automatically at startup.
- Clean up temporary streaming shards in `/workspace/tmp_streaming_shards` (created when streaming on cloud backends).

### Dive Deeper
- `COMPLETE_STREAMING_GUIDE.md` – full walkthrough of streaming workflows.
- `QUICK_START_STREAMING.md` – minimal checklist for streaming experiments.
- `RUNPOD_COMPLETE_SETUP.md` & `infra/runpod.md` – provisioning notes for RunPod.
- `RTX5090_TRAINING_GUIDE.md` – RTX 5090 specific tuning advice.
