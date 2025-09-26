use crate::config::Config;
use crate::errors::{Error, Result};
use std::collections::HashSet;
use std::path::Path;
use tokenizers::models::ModelWrapper;
use tokenizers::Tokenizer;

const BOS_TOKEN: &str = "<bos>";
const EOS_TOKEN: &str = "<eos>";

pub fn validate_config(cfg: &Config) -> Result<()> {
    if cfg.model.vocab_size == 0 {
        return Err(Error::Validation(
            "model.vocab_size must be greater than zero".into(),
        ));
    }

    if cfg.model.min_frequency < 1 {
        return Err(Error::Validation(
            "model.min_frequency must be at least 1".into(),
        ));
    }

    let mut seen = HashSet::new();
    for token in &cfg.model.special_tokens {
        if !seen.insert(token) {
            return Err(Error::Validation(format!(
                "special token '{token}' appears multiple times"
            )));
        }
    }

    if let Some(post_cfg) = &cfg.postprocessor {
        if post_cfg.add_bos && !cfg.model.special_tokens.iter().any(|t| t == BOS_TOKEN) {
            return Err(Error::Validation(
                "postprocessor.add_bos requires '<bos>' in model.special_tokens".into(),
            ));
        }
        if post_cfg.add_eos && !cfg.model.special_tokens.iter().any(|t| t == EOS_TOKEN) {
            return Err(Error::Validation(
                "postprocessor.add_eos requires '<eos>' in model.special_tokens".into(),
            ));
        }
    }

    ensure_directory_creatable(cfg.artifacts.dir.as_path())
}

pub fn validate_tokenizer(tok: &Tokenizer, cfg: &Config) -> Result<()> {
    for token in &cfg.model.special_tokens {
        if tok.token_to_id(token).is_none() {
            return Err(Error::Validation(format!(
                "expected special token '{token}' to be present in tokenizer vocab"
            )));
        }
    }

    if cfg.model.byte_fallback_on_decode {
        match tok.get_model() {
            ModelWrapper::BPE(bpe) => {
                if !bpe.byte_fallback {
                    return Err(Error::Validation(
                        "config requires byte fallback on decode, but tokenizer model has it disabled"
                            .into(),
                    ));
                }
            }
            _ => {
                return Err(Error::Validation(
                    "byte_fallback_on_decode only supported for BPE models".into(),
                ));
            }
        }
    }

    let vocab_limit = cfg.model.vocab_size + cfg.model.special_tokens.len();
    let actual_size = tok.get_vocab_size(true);
    if actual_size > vocab_limit {
        return Err(Error::Validation(format!(
            "tokenizer vocab size {actual_size} exceeds configured limit {vocab_limit}"
        )));
    }

    Ok(())
}

fn ensure_directory_creatable(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        return Ok(());
    }

    if dir.exists() {
        return Err(Error::Validation(format!(
            "artifact directory path '{}' exists but is not a directory",
            dir.display()
        )));
    }

    if let Some(parent) = dir.parent() {
        if parent.as_os_str().is_empty() {
            return Ok(());
        }
        if parent.is_dir() {
            return Ok(());
        }
        if parent.exists() {
            return Err(Error::Validation(format!(
                "artifact directory parent '{}' is not a directory",
                parent.display()
            )));
        }
        return Err(Error::Validation(format!(
            "artifact directory parent '{}' does not exist",
            parent.display()
        )));
    }

    Ok(())
}
