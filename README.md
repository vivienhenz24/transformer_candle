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

### Tokenizer Artifacts
- Place pretrained artifacts under the directory configured in `Config.artifacts` (typically `crates/tokenizer/target/...`).
- To train new byte-level BPE artifacts, enable the training feature: `cargo test -p tokenizer --features train` or run your own driver that calls `train_bbpe`.
- The training pipeline accepts plain-text corpora, emits `tokenizer.json` or split vocab/merge files, and records a manifest with hash and timestamp metadata.

### Suggested Commands
- `cargo fmt && cargo clippy` keeps formatting and linting tidy.
- `cargo test` runs unit/integration suites; add `--features train` when exercising tokenizer training scenarios.
- `RUST_LOG=info cargo run --release` enables logging hooks when Candle executes inference steps.

