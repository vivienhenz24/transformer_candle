#![cfg(feature = "train")]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokenizer::config::TrainingCfg;
use tokenizer::errors::Result;
use tokenizer::{
    build_from_artifacts, train_bbpe, ArtifactsCfg, ByteLevelCfg, Config, ModelCfg, PostCfg,
};
use tokenizer::Error;
use tokenizers::Tokenizer;

const SAMPLE_INPUTS: [&str; 4] = [
    "byte level test",
    "Unicode ☂ sample",
    " spaced input ",
    "emoji 😊 alignment",
];

#[test]
fn save_and_reload_tokenizer_json() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = config_for_serialization(&tmp)?;
    let tok1 = train_bbpe(&cfg)?;

    let tokenizer_path = cfg
        .artifacts
        .tokenizer_json
        .as_ref()
        .expect("tokenizer.json path configured");
    assert!(tokenizer_path.exists(), "tokenizer.json should exist after training");

    let mut cfg_no_json = cfg.clone();
    cfg_no_json.artifacts.tokenizer_json = None;
    let tok2 = build_from_artifacts(&cfg_no_json)?;
    assert_tokenizers_equivalent(&tok1, &tok2, &SAMPLE_INPUTS)?;

    Ok(())
}

#[test]
fn reload_from_vocab_and_merges() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = config_for_serialization(&tmp)?;
    let tok1 = train_bbpe(&cfg)?;

    let tokenizer_path = cfg
        .artifacts
        .tokenizer_json
        .as_ref()
        .expect("tokenizer.json path configured");
    let backup_path = tokenizer_path.with_extension("bak");
    fs::copy(tokenizer_path, &backup_path)?;
    fs::remove_file(tokenizer_path)?;

    let mut cfg_no_json = cfg.clone();
    cfg_no_json.artifacts.tokenizer_json = None;

    let tok2 = build_from_artifacts(&cfg_no_json)?;
    assert_tokenizers_equivalent(&tok1, &tok2, &SAMPLE_INPUTS)?;

    fs::rename(&backup_path, tokenizer_path)?;
    Ok(())
}

#[test]
fn manifest_integrity_and_vocab_shape() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = config_for_serialization(&tmp)?;
    train_bbpe(&cfg)?;

    let manifest_path = cfg
        .artifacts
        .manifest
        .as_ref()
        .expect("manifest path configured");
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;

    let cfg_hash = manifest
        .get("cfg_hash")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .expect("manifest cfg_hash present");
    assert!(!cfg_hash.trim().is_empty());

    let created_at = manifest
        .get("created_at")
        .and_then(Value::as_str)
        .expect("manifest created_at present");
    assert!(created_at.starts_with("unix:"));

    let token_count = manifest
        .get("token_count")
        .and_then(Value::as_u64)
        .expect("manifest token_count present");
    assert!(token_count >= cfg.model.special_tokens.len() as u64);

    let vocab_path = cfg
        .artifacts
        .vocab_json
        .as_ref()
        .expect("vocab path configured");
    let vocab_bytes = fs::read(vocab_path)?;
    let vocab_value: Value = serde_json::from_slice(&vocab_bytes)?;
    let vocab_len = vocab_value
        .as_object()
        .map(|map| map.len())
        .expect("vocab json should be an object");

    let min_vocab = cfg.model.special_tokens.len();
    let max_vocab = cfg.model.vocab_size + cfg.model.special_tokens.len() + 64;
    assert!(vocab_len >= min_vocab, "vocab should include special tokens");
    assert!(vocab_len <= max_vocab, "vocab larger than expected upper bound");

    Ok(())
}

#[test]
fn stable_ids_for_specials() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = config_for_serialization(&tmp)?;
    let tok1 = train_bbpe(&cfg)?;
    let tok2 = build_from_artifacts(&cfg)?;

    for special in cfg.model.special_tokens.iter() {
        let id1 = tok1
            .token_to_id(special)
            .unwrap_or_else(|| panic!("missing special token {} in trained tokenizer", special));
        let id2 = tok2
            .token_to_id(special)
            .unwrap_or_else(|| panic!("missing special token {} in reloaded tokenizer", special));
        assert_eq!(id1, id2, "special token {} ID mismatch", special);
    }

    let bos_id = tok1.token_to_id("<bos>").expect("<bos> id exists");
    let eos_id = tok1.token_to_id("<eos>").expect("<eos> id exists");
    let encoding = tok2.encode("Consistent specials", true)?;
    let ids = encoding.get_ids();
    assert_eq!(ids.first().copied(), Some(bos_id));
    assert_eq!(ids.last().copied(), Some(eos_id));

    Ok(())
}

#[test]
fn json_roundtrip_does_not_change_behavior() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = config_for_serialization(&tmp)?;
    let tok1 = train_bbpe(&cfg)?;

    let tokenizer_path = cfg
        .artifacts
        .tokenizer_json
        .as_ref()
        .expect("tokenizer.json path configured");
    let raw = fs::read(tokenizer_path)?;
    let value: Value = serde_json::from_slice(&raw)?;

    let copy_path = tokenizer_path
        .with_file_name("tokenizer_copy.json");
    let mut file = fs::File::create(&copy_path)?;
    serde_json::to_writer_pretty(&mut file, &value)?;
    file.write_all(b"\n")?;

    let mut cfg_copy = cfg.clone();
    cfg_copy.artifacts.tokenizer_json = Some(copy_path.clone());

    let tok2 = build_from_artifacts(&cfg_copy)?;
    assert_tokenizers_equivalent(&tok1, &tok2, &SAMPLE_INPUTS)?;

    fs::remove_file(copy_path)?;
    Ok(())
}

fn assert_tokenizers_equivalent(
    tok1: &Tokenizer,
    tok2: &Tokenizer,
    samples: &[&str],
) -> Result<()> {
    for &sample in samples.iter() {
        compare_encoding(tok1, tok2, sample, false)?;
        compare_encoding(tok1, tok2, sample, true)?;
    }
    Ok(())
}

fn compare_encoding(
    tok1: &Tokenizer,
    tok2: &Tokenizer,
    sample: &str,
    add_special_tokens: bool,
) -> Result<()> {
    let enc1 = tok1.encode(sample, add_special_tokens)?;
    let enc2 = tok2.encode(sample, add_special_tokens)?;
    assert_eq!(enc1.get_ids(), enc2.get_ids());

    let dec1 = strip_special_tokens(&tok1.decode(enc1.get_ids(), false)?);
    let dec2 = strip_special_tokens(&tok2.decode(enc2.get_ids(), false)?);
    assert_eq!(dec1, dec2);

    Ok(())
}

fn tmp_dir() -> Result<PathBuf> {
    let pid = process::id();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::Validation(format!("time went backwards: {e}")))?
        .as_nanos();
    let path = PathBuf::from("target")
        .join("bbpe_tests")
        .join(format!("serialization_{}_{}", pid, timestamp));
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn config_for_serialization(tmp: &Path) -> Result<Config> {
    write_corpus(tmp)?;

    Ok(Config {
        model: ModelCfg {
            vocab_size: 256,
            min_frequency: 1,
            dropout: None,
            special_tokens: vec![
                "<pad>".to_string(),
                "<unk>".to_string(),
                "<bos>".to_string(),
                "<eos>".to_string(),
            ],
            byte_fallback_on_decode: true,
        },
        pretokenizer: ByteLevelCfg {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true,
        },
        postprocessor: Some(PostCfg {
            add_bos: true,
            add_eos: true,
            pair_template: false,
        }),
        #[cfg(feature = "train")]
        training: Some(TrainingCfg {
            inputs: vec![tmp.join("corpus.txt")],
            seed: 123,
            shuffle: false,
            max_lines: None,
            num_threads: Some(1),
        }),
        artifacts: ArtifactsCfg {
            dir: tmp.to_path_buf(),
            tokenizer_json: Some(tmp.join("tokenizer.json")),
            vocab_json: Some(tmp.join("vocab.json")),
            merges_txt: Some(tmp.join("merges.txt")),
            manifest: Some(tmp.join("manifest.json")),
        },
    })
}

fn write_corpus(dir: &Path) -> Result<()> {
    let path = dir.join("corpus.txt");
    let contents = [
        "byte level byte level",
        "tokenize this text",
        "unknown ☂ bytes are fine",
    ]
    .join("\n");
    fs::write(path, contents + "\n")?;
    Ok(())
}

fn strip_special_tokens(input: &str) -> String {
    let mut result = input.to_owned();
    for token in ["<bos>", "<eos>", "<pad>", "<unk>"] {
        result = result.replace(token, "");
    }
    result.trim().to_string()
}
