//! Byte-level BPE utilities for pretraining and inference.
//!
//! This crate exposes a stable, minimal surface for loading or assembling
//! tokenizers configured via [`Config`]. The configuration describes a
//! byte-level BPE pipeline: raw bytes are pre-tokenized with a configurable
//! byte-level splitter, merged by a BPE model, and optionally processed with
//! template-based BOS/EOS handling. Artifacts can come from a bundled
//! `tokenizer.json` or from split `vocab.json` + `merges.txt` pairs.
//!
//! # Configuration
//!
//! `Config` must specify the model hyperparameters (vocab size, minimum
//! frequency, byte fallback behaviour, special tokens), byte-level
//! pre-tokenizer settings, optional post-processing, and artifact locations.
//! The loader prefers `tokenizer.json` when present; otherwise both
//! `vocab.json` and `merges.txt` must be provided. Validation ensures all
//! required pieces exist before touching the filesystem.
//!
//! # Training Feature
//!
//! Enabling the `train` feature unlocks `train_bbpe`, which performs
//! deterministic, memory-bounded training from one or more text corpora.
//! Otherwise only artifact loading is available.
//!
//! # Thread Safety
//!
//! Built tokenizers are validated to be `Send + Sync`, so they can be shared
//! across threads without additional synchronization.

pub mod config;
pub mod errors;

mod artifacts;
mod bbpe;
mod postprocessor;
mod pretokenizer;
mod types;
mod validate;
#[cfg(feature = "train")]
mod trainer;

pub use config::{ArtifactsCfg, ByteLevelCfg, Config, ModelCfg, PostCfg};
pub use errors::{Error, Result};

pub fn build_from_artifacts(cfg: &Config) -> Result<tokenizers::Tokenizer> {
    validate::validate_config(cfg)?;
    let tokenizer = bbpe::build_from_artifacts(cfg)?;
    validate::validate_tokenizer(&tokenizer, cfg)?;
    Ok(tokenizer)
}

#[cfg(feature = "train")]
pub fn train_bbpe(cfg: &Config) -> Result<tokenizers::Tokenizer> {
    validate::validate_config(cfg)?;
    let tokenizer = bbpe::train_and_build(cfg)?;
    validate::validate_tokenizer(&tokenizer, cfg)?;
    Ok(tokenizer)
}
