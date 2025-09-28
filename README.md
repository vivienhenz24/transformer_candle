## transformer_candle

hi!

We ([@SamC1249](https://github.com/SamC1249) & [@vivienhenz24](https://github.com/vivienhenz24)) are building a transformer from scratch...but in rust. And no, there is no rational reason for building a transformer in rust. We do a little trolling ;)

main libs used are huggingface's candle and tokenizers.



### Getting Started
1. Install the Rust toolchain (`rustup install stable` is enough) and ensure `cargo` is on your path.
2. Clone the repository, then run `cargo build` at the root to download dependencies such as Candle and tokenizers.
3. Execute `cargo run --release` to launch the demo binary once you have artifacts in place.

### Layout Highlights
- `main.rs`: entry point that wires together model loading, tokenizer construction, and inference/demo logic.
- `crates/tokenizer`: thin wrapper around Hugging Face's tokenizer library, offering config-driven byte-level BPE loading and (optionally) training.
- `crates/pretraining-data`: helpers for sourcing and validating pretraining corpora.
- `tests/`: integration and regression coverage for tokenizer round-tripping, serialization, and end-to-end flows.
- `docs/rope.md`: overview of rotary positional embeddings, configuration knobs, and caching behaviour.

### Architecture Overview
```
                              +---------------------+
                              |  pretraining-data   |
                              |  (corpus streamer)  |
                              +----------+----------+
                                         |
                                         v
                            +------------+-------------+
                            |         training         |
                            | (CLI, trainer, runtime)  |
                            +----+------+------+------+
                                 |      |      |
                                 |      |      +-------------------------------+
                                 |      |                                      |
                                 |      |                           +----------v-----------+
                                 |      |                           |      tokenizer       |
                                 |      |                           | (load/train BBPE)    |
                                 |      |                           +----------+-----------+
                                 |      |                                      |
                                 |      |                            token IDs |
                                 |      |                                      v
                                 |      |                           +----------+-----------+
                                 |      +--------------------------->      model          |
                                 |                                  | (forward pass)      |
                                 |                                  +----+-----------+----+
                                 |                                       |           |
                                 v                                       |           |
                        +--------+--------+                              |           |
                        |   optimizer     |                              |           |
                        | schedulers, etc |
                        +--------+--------+                              |           |
                                 |                                       |           |
                                 | gradients                             |           |
                                 v                                       |           |
                        +--------+--------+                              |           |
                        | gradient scaler |                              |           |
                        +--------+--------+                              |           |
                                 |                                       |           |
                                 | parameters                            |           |
                                 v                                       |           |
                      +----------+----------+                            |           |
                      |    layers           |<---------------------------+           |
                      | (MLP/blocks utils)  |                                        |
                      +----------+----------+                                        |
                                 |                                                   |
                                 v                                                   |
                      +----------+----------+                                        |
                      |   attention        |<----------------------------------------+
                      +----------+----------+
                                 |
                                 v
                      +----------+----------+
                      |  embedding        |
                      +-------------------+
```

`training` orchestrates the full loop: it streams batches from `pretraining-data`, pulls tokens via `tokenizer`, and invokes `model` for forward passes. The `model` crate composes building blocks from `embedding`, `attention`, and `layers`, while `training` also owns optimization, gradient scaling, checkpointing, and scheduling logic.

### Tokenizer Artifacts
- Place pretrained artifacts under the directory configured in `Config.artifacts` (typically `crates/tokenizer/target/...`).
- To train new byte-level BPE artifacts, enable the training feature: `cargo test -p tokenizer --features train` or run your own driver that calls `train_bbpe`.
- The training pipeline accepts plain-text corpora, emits `tokenizer.json` or split vocab/merge files, and records a manifest with hash and timestamp metadata.

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
