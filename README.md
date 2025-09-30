## transformer_candle

Hi!

We ([@SamC1249](https://github.com/SamC1249) & [@vivienhenz24](https://github.com/vivienhenz24)) are building a transformer from scratch...but in rust. And no, there is no rational reason for building a transformer in rust.

Main libs used are huggingface's candle and tokenizers.



### Getting Started
1. Install the Rust toolchain (`rustup install stable` is enough) and ensure `cargo` is on your path.
2. Clone the repository, then run `cargo build` at the root to download dependencies such as Candle and tokenizers.
3. Download checkpoints from the Hugging Face Hub (`huggingface-cli download vivienhenz/sconce --local-dir runs/pretrained`) or sync them to a directory referenced by your config.
4. Execute `cargo run --release` to launch the demo binary once you have artifacts in place.

### RunPod Cloud Training
Run large experiments in the cloud following a single command flow powered by the orchestrator.

1. **Provision** a pod with an A100 80 GB and mount your shared RunPod volume (`runpodctl create pod -f infra/runpod-pod.yaml`).
2. **(Optional) Shard** FineWeb by setting `RUN_SHARD=1` when launching the pod. The bundled sharder writes newline shards to `/workspace/datasets/{train,val}/<experiment>` and emits hashed manifests.
3. **Orchestrate** tokenizer + config generation: the pod manifest calls `cargo run -p training --bin orchestrate -- --mode cloud`, which validates shards, reuses or retrains the tokenizer, and saves `training.toml`/`training.yaml`.
4. **Train** with `cargo run -p training --bin train -- --config configs/runpod_2b.yaml` (default) or choose another profile via `RUN_CONFIG`.

| Profile | Size | Target environment | Notes |
| --- | --- | --- | --- |
| `configs/m3_medium.yaml` | ~22M params | Laptop / dev GPU / smoke tests | Use for local validation before burning cloud time. |
| `configs/runpod_1b.yaml` | ~1.0B params | RunPod A100 80 GB | Recommended starter profile; lighter memory footprint and faster iterations. |
| `configs/runpod_2b.yaml` | ~1.97B params | RunPod A100 80 GB | Higher-capacity model; requires more wall-clock time and I/O. |
| `configs/runpod_4b.yaml` | ~4B params | Multi-GPU (future) / single A100 with high GA | Very slow on a single GPU; provided for scale-out rehearsals. |

**Hugging Face datasets**: use `crates/pretraining-data/scripts/fineweb_sharder.py` inside the pod to stream `HuggingFaceFW/fineweb` directly into shard directories. The script respects `HF_HOME=/workspace/hf_cache` and records per-shard hashes for reproducibility.

**Tokenizer policy**: the orchestrator reuses any tokenizer bundle found under `/workspace/artifacts/tokenizer/<experiment>`. Pass `--force-retrain-tokenizer` to rebuild, or supply `--tokenizer-json` to load external artifacts.

Cost control tip: start with `RUN_SHARD=0` and the `m3_medium` profile locally. Once satisfied, enable cloud sharding and the 2B config; graduate to the 4B experiment only after you monitor memory/throughput.

### Layout Highlights
- `main.rs`: entry point that wires together model loading, tokenizer construction, and inference/demo logic.
- `crates/tokenizer`: thin wrapper around Hugging Face's tokenizer library, offering config-driven byte-level BPE loading and (optionally) training.
- `crates/pretraining-data`: helpers for sourcing and validating pretraining corpora.
- `tests/`: integration and regression coverage for tokenizer round-tripping, serialization, and end-to-end flows.
- `docs/rope.md`: overview of rotary positional embeddings, configuration knobs, and caching behaviour.

### Architecture Overview
```text
configs/, infra/, runs/*
        (experiment wiring + artifacts)
                       |
                       v
+-----------------------------------------------------------------------+
| crates/training (trainer CLI)                                         |
|  config parsing - dataloading - optimizer/scheduler - checkpoints     |
+-----------+-----------------------------------------------------------+
            | batches
            v
    +--------------------------+
    | crates/pretraining-data  |
    |  text shard streamer     |
    +-------------+------------+
                  | text
                  v
    +--------------------------+
    | crates/tokenizer         |
    |  HF tokenizer wrappers   |
    +-------------+------------+
                  | token ids
                  v
    +---------------------------------------------------------------+
    | crates/model (decoder-only transformer)                       |
    |  orchestrates crates/embedding, crates/attention, layers/*    |
    +-------------+-------------------------------------------------+
                  | logits
                  v
      downstream consumers (trainer loop, demos)
```

```text
Model forward pass

+------------------------------+          +------------------------------+
| embedding::token             |          | final norm (layers::norm)    |
|  token ids -> hidden states  |          |  prepares logits projection  |
+---------------+--------------+          +---------------+--------------+
                |                                         ^
                v                                         |
        repeat for n_layers                               |
+-----------------------------------------------------------------------+
| Decoder block (crates/model::block)                                  |
|                                                                       |
|  +------------------------------+   residual path via layers::residual|
|  | layers::norm (prenorm)       |------------------------------------+|
|  +---------------+--------------+                                     |
|                  |                                                    |
|                  v                                                    |
|  +------------------------------+    +------------------------------+  |
|  | attention::core/masks        |    | layers::linear (QKV/out proj)|  |
|  |  multi-head self-attn        |    |  parameter storage and matmuls|  |
|  |  rotary pos via embedding::  |    +------------------------------+  |
|  |  positional::rope            |                                   |
|  +---------------+--------------+                                   |
|                  |                                                  |
|                  v                                                  |
|        residual add (layers::residual::prenorm_step)                |
|                  |                                                  |
|                  v                                                  |
|  +------------------------------+                                   |
|  | layers::norm (prenorm)       |                                   |
|  +---------------+--------------+                                   |
|                  |                                                  |
|                  v                                                  |
|  +------------------------------+                                   |
|  | layers::mlp (FeedForward)    |  up/down projections and activation|
|  |  hidden -> ff_ratio*hidden   |  managed via layers::linear        |
|  +---------------+--------------+                                   |
|                  |                                                  |
|                  v                                                  |
|        residual add (layers::residual::prenorm_step)                |
+-----------------------------------------------------------------------+
```

`training` orchestrates the loop: it streams batches from `crates/pretraining-data`, tokenizes via `crates/tokenizer`, and drives `crates/model`. The model stitches together `crates/embedding`, `crates/attention`, and the `crates/layers` utilities (linear, norm, residual, mlp) to implement the transformer stack end to end.

### Tokenizer Artifacts
- Place pretrained artifacts under the directory configured in `Config.artifacts` (typically `crates/tokenizer/target/...`).
- To train new byte-level BPE artifacts, enable the training feature: `cargo test -p tokenizer --features train` or run your own driver that calls `train_bbpe`.
- The training pipeline accepts plain-text corpora, emits `tokenizer.json` or split vocab/merge files, and records a manifest with hash and timestamp metadata.

### Pretrained Weights
- Public checkpoints are hosted at `https://huggingface.co/vivienhenz/sconce`. Use `huggingface-cli download`, `git lfs clone`, or the Hub Python API to mirror them locally.
- Point configs such as `configs/runpod_1b.yaml.training.checkpoint_dir` or CLI flags at the directory where the downloaded `safetensors` weights live (for example `runs/pretrained/checkpoints`).
- Do not commit large binaries to this repository; reference the Hub location instead so updates stay lightweight.

### Running the M3 Medium Training Loop
- **Prepare tokenizer artifacts**: make sure `runs/m3-medium/tokenizer/tokenizer.json` (plus optional `vocab.json`/`merges.txt`) and `runs/m3-medium/special_tokens.txt` exist. If you trained your own tokenizer, copy the artifacts into that directory or update `configs/m3_medium.yaml.tokenizer` paths to match your layout.
- **Point the corpus at your data**: by default the config reads `crates/pretraining-data/input.txt`. Replace or symlink that file to your training corpus before launching.
- **Kick off training**: run `cargo run -p training --bin train -- --config configs/m3_medium.yaml`. The binary loads the tokenizer via `build_from_artifacts`, constructs the 22M parameter model, and begins streaming data through the optimizer.
- **Resuming or tweaking**: append `--resume` to continue from the latest checkpoint under `runs/m3-medium/checkpoints`, or pass overrides such as `--override optimizer.learning_rate=4e-5` to experiment without editing the YAML.
- **Monitoring**: logs land on stdout and TensorBoard summaries are written to `runs/m3-medium/tensorboard`. Point TensorBoard at that directory (`tensorboard --logdir runs/m3-medium/tensorboard`) to visualize loss and grad norms.

### Suggested Commands
- `cargo fmt && cargo clippy` keeps formatting and linting tidy.
- `cargo test` runs unit/integration suites; add `--features train` when exercising tokenizer training scenarios.
- `RUST_LOG=info cargo run --release` enables logging hooks when Candle executes inference steps.
- cargo run -p training --bin orchestrate -- \        
  --profile local \
  --input crates/pretraining-data/input.txt \
  --run-dir runs/local-smoke

for local smoke test.


 cargo run -p training --bin train -- --config configs/m3_medium.yaml

 for full m3 run

  cargo run -p training --bin train -- --config configs/m3_medium.yaml --resume

  for resume a run from last checkpoint
