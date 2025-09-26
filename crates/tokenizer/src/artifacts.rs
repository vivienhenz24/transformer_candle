use crate::config::{ArtifactsCfg, Config};
use crate::errors::{Error, Result};
use crate::types::{ArtifactManifest, ArtifactPaths};
use serde_json;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

const TOKENIZER_JSON_ERR: &str = "tokenizer json not found at";
const VOCAB_JSON_ERR: &str = "vocab json not found at";
const MERGES_TXT_ERR: &str = "merges txt not found at";
const MANIFEST_ERR: &str = "manifest not found at";

pub fn load_tokenizer_from_json(path: &Path) -> Result<Tokenizer> {
    ensure_file(path, TOKENIZER_JSON_ERR)?;
    Tokenizer::from_file(path).map_err(Error::from)
}

pub fn load_bpe_from_vocab_merges(vocab: &Path, merges: &Path) -> Result<BPE> {
    ensure_file(vocab, VOCAB_JSON_ERR)?;
    ensure_file(merges, MERGES_TXT_ERR)?;

    let vocab_str = path_to_string(vocab)?;
    let merges_str = path_to_string(merges)?;

    BPE::from_file(&vocab_str, &merges_str)
        .build()
        .map_err(Error::from)
}

pub fn save_tokenizer_json(tok: &Tokenizer, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(Error::from)?;
        }
    }

    tok.save(path, None).map_err(Error::from)
}

pub fn write_manifest(manifest_path: &Path, manifest: &ArtifactManifest) -> Result<()> {
    if let Some(parent) = manifest_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(Error::from)?;
        }
    }

    let file = File::create(manifest_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, manifest)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub fn read_manifest(manifest_path: &Path) -> Result<ArtifactManifest> {
    ensure_file(manifest_path, MANIFEST_ERR)?;
    let file = File::open(manifest_path)?;
    let reader = BufReader::new(file);
    let manifest = serde_json::from_reader(reader)?;
    Ok(manifest)
}

pub fn compute_config_hash(cfg: &Config, extra_paths: &[&Path]) -> Result<String> {
    let mut hasher = Sha256::new();
    let cfg_bytes = serde_json::to_vec(cfg)?;
    hasher.update(&cfg_bytes);

    let mut sorted = extra_paths.to_vec();
    sorted.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));

    let mut buffer = [0u8; 8 * 1024];
    for path in sorted {
        ensure_file(path, "cannot hash missing file at")?;
        hasher.update(path.to_string_lossy().as_bytes());

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

pub fn resolve_paths(cfg: &ArtifactsCfg) -> Result<ArtifactPaths> {
    let dir = cfg.dir.as_path();
    if !dir.is_dir() {
        return Err(Error::Artifact(format!(
            "artifact directory not found at {}",
            dir.display()
        )));
    }

    let resolve = |value: &Option<PathBuf>| -> Option<PathBuf> {
        value.as_ref().map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                dir.join(path)
            }
        })
    };

    let tokenizer_json = resolve(&cfg.tokenizer_json);
    let vocab_json = resolve(&cfg.vocab_json);
    let merges_txt = resolve(&cfg.merges_txt);
    let manifest = resolve(&cfg.manifest);

    if let Some(ref path) = tokenizer_json {
        ensure_file(path, TOKENIZER_JSON_ERR)?;
    }

    if let Some(ref path) = vocab_json {
        ensure_file(path, VOCAB_JSON_ERR)?;
    }

    if let Some(ref path) = merges_txt {
        ensure_file(path, MERGES_TXT_ERR)?;
    }

    if tokenizer_json.is_none() {
        match (&vocab_json, &merges_txt) {
            (Some(_), Some(_)) => {}
            (None, Some(path)) => {
                return Err(Error::Artifact(format!(
                    "vocab json path is required when merges txt is set (missing for {})",
                    path.display()
                )));
            }
            (Some(path), None) => {
                return Err(Error::Artifact(format!(
                    "merges txt path is required when vocab json is set (missing for {})",
                    path.display()
                )));
            }
            (None, None) => {
                return Err(Error::Artifact(
                    "artifacts must specify either tokenizer_json or both vocab_json and merges_txt".into(),
                ));
            }
        }
    }

    Ok(ArtifactPaths {
        json: tokenizer_json,
        vocab: vocab_json,
        merges: merges_txt,
        manifest,
    })
}

fn ensure_file(path: &Path, context: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(Error::Artifact(format!("{context} {}", path.display())))
    }
}

fn path_to_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(|s| s.to_owned())
        .ok_or_else(|| Error::Artifact(format!("path is not valid UTF-8: {}", path.display())))
}
