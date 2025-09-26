use crate::config::ArtifactsCfg;
use crate::errors::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

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

pub fn sha256_of_files(paths: &[&Path]) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8 * 1024];

    for path in paths {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }

    Ok(format!("{:x}", hasher.finalize()))
}
