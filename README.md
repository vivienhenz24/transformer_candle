## transformer_candle

Hi! This is a generative pre-trained transformer built using rust and Huggingface's candle. And no, there is no rational reason for building a transformer in rust. We just thought it would be more fun (spoiler alert: it wasn't).

Currently training the ~1.2B param version on a RTX5090 via runpod, should take about 6-7 days.

### Crate Interaction Map
```text
configs/*  ───────► training crate (orchestrate.rs, train.rs)
                       │  parses configs, drives Trainer, handles checkpoints
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
 tokenizer crate            pretraining-data crate
  builds/trains BBPE          streams newline shards or HF datasets
         │                           │
         └──────────────┬────────────┘
                        ▼
         StreamingTextDataLoader (training crate)
                        │ tokenized batches
                        ▼
                   model crate
                        │ forward pass
       ┌────────────┬────────────┬────────────┐
       ▼            ▼            ▼            ▼
 embedding     attention      layers      cache utils
 crate         crate          crate       (inside model)
 (token +      (kernels,      (linear,    (masks,
 positional)   RoPE, KV)      norm, MLP)  positions)
                        │
                        ▼
              logits ▸ trainer metrics / checkpoints
```

### Transformer Stack Layout
```text
input token ids
      │
      ▼
embedding::token (tied input/output weights)
      │ hidden states
      ▼
[repeat n_layers times]
    ┌─────────────────────────────────────────────────────────────┐
    │ layers::norm pre-norm                                        │
    │   ▼                                                          │
    │ attention::fused (QKV projection ▸ RoPE ▸ causal attention)   │
    │   ▼                                                          │
    │ residual connection (layers::residual)                       │
    │   ▼                                                          │
    │ layers::mlp (SiLU feed-forward, expansion ratio from config) │
    │   ▼                                                          │
    │ residual connection                                          │
    └─────────────────────────────────────────────────────────────┘
      │
      ▼
final norm (layers::norm) ──► embedding::token.linear_out ──► logits
```

### Directory Structure (trimmed)
```text
.
├── Cargo.toml
├── README.md
├── configs/                (experiment presets)
├── crates/
│   ├── attention/          (fused/reference kernels, RoPE + KV cache controls)
│   ├── embedding/          (token embeddings, positional/RoPE utilities)
│   ├── layers/             (linear, norm, residual, activations, dtype policies)
│   ├── model/              (decoder-only transformer wrapper & caches)
│   ├── pretraining-data/   (newline shard streaming + HF dataset bridge)
│   ├── tokenizer/          (config-driven byte-level BPE loader/trainer)
│   └── training/           (Trainer core, orchestrate/train binaries)
├── infra/                  (RunPod manifests and setup notes)
├── runs/                   (generated artifacts: tokenizer, checkpoints, logs)
└── guides/*.md             (streaming and hardware walkthroughs)
```

### Key Crates
- `crates/training` – Trainer, optimizer, scheduler, logging, checkpointing, and the `orchestrate` / `train` binaries.
- `crates/tokenizer` – Byte-level BPE loader with optional deterministic training when the `train` feature is on.
- `crates/pretraining-data` – Streaming corpus abstractions for newline shards and Hugging Face datasets.
- `crates/model` – Decoder-only transformer wrapper that stitches embeddings, attention, and MLP blocks with mask/position caches.
- `crates/attention` – Fused/reference kernels plus KV cache helpers and RoPE settings.
- `crates/embedding` – Token embeddings, positional/rotary helpers, and tied output projection.
- `crates/layers` – Linear layers, norms, residual/dropout wrappers, activation helpers, and dtype policies.

### Common Commands
- Build once to pull Candle/tokenizers: `cargo build --workspace`
- Run orchestrator locally:
  ```
  cargo run -p training --bin orchestrate -- \
    --mode local \
    --input crates/pretraining-data/input.txt \
    --run-dir runs/local-smoke
  ```
- Launch training with the generated config:
  ```
  cargo run -p training --bin train -- \
    --config runs/local-smoke/training.toml
  ```
  Append `--resume` to pick up the latest checkpoint or `--override key=value` for on-the-fly tweaks.

### Notes
- Drop a `streaming_config.json` next to any training config and the `train` binary will stream from Hugging Face into temporary shards before starting.
- `runtime.precision` selects the Candle dtype (BF16 by default) and toggles gradient scaling/master weights.
- Attention backend, precision, and KV behaviour can be forced via `ATTN_*` environment variables.
- Lint/test helpers: `cargo fmt`, `cargo clippy --all-targets --workspace`, `cargo test --workspace` (`--features train` for tokenizer training coverage).
- Cloud manifests and longer walkthroughs live under `infra/` and the guide docs in the repository root.
