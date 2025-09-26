use crate::config::ArtifactsCfg;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub cfg_hash: String,
    pub created_at: String,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct ArtifactPaths {
    pub json: Option<PathBuf>,
    pub vocab: Option<PathBuf>,
    pub merges: Option<PathBuf>,
    pub manifest: Option<PathBuf>,
}

impl From<&ArtifactsCfg> for ArtifactPaths {
    fn from(cfg: &ArtifactsCfg) -> Self {
        Self {
            json: cfg.tokenizer_json.clone(),
            vocab: cfg.vocab_json.clone(),
            merges: cfg.merges_txt.clone(),
            manifest: cfg.manifest.clone(),
        }
    }
}
