use crate::artifacts::{
    load_bpe_from_vocab_merges, load_tokenizer_from_json, resolve_paths,
};
use crate::config::Config;
use crate::errors::{Error, Result};
use crate::postprocessor::maybe_build_template;
use crate::pretokenizer::build_byte_level;
use crate::types::ArtifactPaths;
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub fn build_from_artifacts(cfg: &Config) -> Result<Tokenizer> {
    let ArtifactPaths {
        json,
        vocab,
        merges,
        ..
    } = resolve_paths(&cfg.artifacts)?;

    let mut tokenizer = if let Some(json_path) = json {
        load_tokenizer_from_json(&json_path)?
    } else {
        let vocab_path = vocab.ok_or_else(|| {
            Error::Artifact("vocab_json path is required when tokenizer_json is absent".into())
        })?;
        let merges_path = merges.ok_or_else(|| {
            Error::Artifact("merges_txt path is required when tokenizer_json is absent".into())
        })?;

        let mut bpe = load_bpe_from_vocab_merges(&vocab_path, &merges_path)?;
        bpe.dropout = cfg.model.dropout;
        bpe.byte_fallback = cfg.model.byte_fallback_on_decode;
        Tokenizer::new(bpe)
    };

    let byte_level = build_byte_level(&cfg.pretokenizer)?;
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(crate::pretokenizer::build_byte_level_decoder()));

    let mut special_ids: HashMap<String, u32> = HashMap::new();
    for token in &cfg.model.special_tokens {
        if let Some(id) = tokenizer.token_to_id(token) {
            special_ids.insert(token.clone(), id);
        }
    }

    if let Some(post_cfg) = cfg.postprocessor.as_ref() {
        if let Some(post) = maybe_build_template(post_cfg, &special_ids)? {
            tokenizer.with_post_processor(Some(post));
        }
    }

    ensure_send_sync(&tokenizer);

    Ok(tokenizer)
}

#[cfg(feature = "train")]
pub fn train_and_build(cfg: &Config) -> Result<Tokenizer> {
    crate::trainer::train_from_corpus(cfg)
}

fn ensure_send_sync<T: Send + Sync>(_: &T) {}
