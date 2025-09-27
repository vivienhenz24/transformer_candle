#![cfg(feature = "train")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use tokenizer::config::TrainingCfg;
use tokenizer::errors::Result;
use tokenizer::Error;
use tokenizer::{
    build_from_artifacts, train_bbpe, ArtifactsCfg, ByteLevelCfg, Config, ModelCfg, PostCfg,
};

const CORPUS_LINES: [&str; 18] = [
    "Hello, world!",
    "hello hello helper help",
    " leading space",
    "trailing space ",
    "multiple spaces",
    "naïve café",
    "emoji 😊 works",
    "日本語テキスト mixed",
    "über Straße",
    "tabs\tand spaces",
    "Question? Answer!",
    "Don’t make the bucket public unless you truly intend it.",
    "Make sure your corpus license allows cloud storage and training use. Some corpora have redistribution or commercial-use limits.",
    "Don't forget to include the training configuration in your dataset.",
    "Keep your dataset size manageable, as larger datasets require more compute resources.",
    "Consider using a dataset split to ensure your model generalizes well.",
    "Monitor the training process closely to detect any issues early.",
    "Regularly save checkpoints to avoid losing progress."
];

#[test]
fn train_and_basic_roundtrip() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = default_config(&tmp)?;
    train_bbpe(&cfg)?;
    let tok = build_from_artifacts(&cfg)?;

    assert!(
        tok.get_vocab_size(true) >= cfg.model.special_tokens.len(),
        "vocab missing expected special tokens"
    );

    let text = "Hello world";
    let encoding = tok.encode(text, false)?;
    let decoded = tok.decode(encoding.get_ids(), true)?;
    assert_eq!(decoded, text);

    let encoding = tok.encode(text, true)?;
    let decoded = tok.decode(encoding.get_ids(), true)?;
    assert_eq!(decoded, text);

    let start = text.find("world").expect("substring present");
    let end = start + "world".len();
    let mut reconstructed = String::new();
    for (offset_start, offset_end) in encoding.get_offsets() {
        let overlap_start = (*offset_start).max(start);
        let overlap_end = (*offset_end).min(end);
        if overlap_start < overlap_end {
            reconstructed.push_str(&text[overlap_start..overlap_end]);
        }
    }
    assert_eq!(
        reconstructed, "world",
        "expected offsets to recover substring 'world'"
    );

    Ok(())
}

#[test]
fn unicode_roundtrip() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = default_config(&tmp)?;
    train_bbpe(&cfg)?;
    let tok = build_from_artifacts(&cfg)?;

    let samples = [
        "naïve café",
        "emoji 😊 works",
        "über Straße",
        "日本語テキスト",
    ];

    for &sample in samples.iter() {
        let encoding = tok.encode(sample, false)?;
        let decoded = tok.decode(encoding.get_ids(), true)?;
        assert_eq!(decoded, sample);

        let encoding = tok.encode(sample, true)?;
        let decoded = tok.decode(encoding.get_ids(), true)?;
        assert_eq!(decoded, sample);
    }

    Ok(())
}

#[test]
fn pair_inputs_roundtrip() -> Result<()> {
    let tmp = tmp_dir()?;
    let mut cfg = default_config(&tmp)?;
    if let Some(post) = cfg.postprocessor.as_mut() {
        post.pair_template = true;
    }
    train_bbpe(&cfg)?;
    let tok = build_from_artifacts(&cfg)?;

    let encoding = tok.encode(("Question?", "Answer!"), true)?;
    let decoded = tok.decode(encoding.get_ids(), true)?;
    let first_pos = decoded.find("Question?").unwrap();
    let second_pos = decoded.find("Answer!").unwrap();
    assert!(first_pos < second_pos);

    Ok(())
}

#[test]
fn whitespace_stability() -> Result<()> {
    let tmp = tmp_dir()?;
    let cfg = default_config(&tmp)?;
    train_bbpe(&cfg)?;
    let tok = build_from_artifacts(&cfg)?;

    let samples = [" leading", "trailing ", "a b c", "tabs\tand spaces"];

    for &sample in samples.iter() {
        let encoding = tok.encode(sample, false)?;
        let decoded = tok.decode(encoding.get_ids(), true)?;
        assert_eq!(decoded.as_bytes(), sample.as_bytes());
    }

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
        .join(format!("roundtrip_{}_{}", pid, timestamp));
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn default_config(tmp: &Path) -> Result<Config> {
    write_corpus(tmp)?;

    Ok(Config {
        model: ModelCfg {
            vocab_size: 320,
            min_frequency: 2,
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
            seed: 42,
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
    println!("roundtrip test corpus lines: {:?}", CORPUS_LINES);
    let contents = CORPUS_LINES.join("\n");
    fs::write(path, contents + "\n")?;
    Ok(())
}
