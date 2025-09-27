#![cfg(feature = "train")]

use crate::artifacts::{compute_config_hash, save_tokenizer_json, write_manifest};
use crate::config::Config;
use crate::errors::{Error, Result};
use crate::postprocessor::maybe_build_template;
use crate::pretokenizer::build_byte_level;
use crate::types::ArtifactManifest;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::models::{ModelWrapper, TrainerWrapper};
use tokenizers::tokenizer::AddedToken;
use tokenizers::{Model, Tokenizer};

const SHUFFLE_CHUNK_SIZE: usize = 2048;

pub fn train_from_corpus(cfg: &Config) -> Result<Tokenizer> {
    let training_cfg = cfg.training.as_ref().ok_or(Error::InvalidConfig(
        "training section required when feature=\"train\"",
    ))?;

    if training_cfg.inputs.is_empty() {
        return Err(Error::InvalidConfig(
            "training inputs must contain at least one path",
        ));
    }

    for path in &training_cfg.inputs {
        if !path.is_file() {
            return Err(Error::Artifact(format!(
                "training input not found at {}",
                path.display()
            )));
        }
    }

    if let Some(num_threads) = training_cfg.num_threads {
        std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());
        tokenizers::utils::parallelism::set_parallelism(num_threads > 1);
    }

    let mut tokenizer = Tokenizer::new(BPE::default());
    let byte_level = build_byte_level(&cfg.pretokenizer)?;
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(crate::pretokenizer::build_byte_level_decoder()));

    let special_tokens: Vec<AddedToken> = cfg
        .model
        .special_tokens
        .iter()
        .cloned()
        .map(|token| AddedToken::from(token, true))
        .collect();

    let mut trainer: TrainerWrapper = BpeTrainer::builder()
        .vocab_size(cfg.model.vocab_size)
        .min_frequency(cfg.model.min_frequency.into())
        .show_progress(false)
        .special_tokens(special_tokens)
        .build()
        .into();

    let mut corpus = CorpusIterator::new(
        training_cfg.inputs.clone(),
        training_cfg.shuffle,
        training_cfg.seed,
        training_cfg.max_lines,
    );

    tokenizer.train(&mut trainer, corpus.by_ref())?;

    if let Some(err) = corpus.take_error() {
        return Err(err);
    }

    let mut model = match tokenizer.get_model() {
        ModelWrapper::BPE(bpe) => bpe.clone(),
        _ => unreachable!("trainer operates on BPE model"),
    };
    model.dropout = cfg.model.dropout;
    model.byte_fallback = cfg.model.byte_fallback_on_decode;
    tokenizer.with_model(model);

    let mut special_ids = HashMap::new();
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

    let mut extra_paths: Vec<PathBuf> = Vec::new();
    let base_dir = &cfg.artifacts.dir;
    if !base_dir.exists() {
        std::fs::create_dir_all(base_dir)?;
    }

    if let Some(tokenizer_path) = cfg
        .artifacts
        .tokenizer_json
        .as_ref()
        .map(|p| absolute_in_dir(base_dir, p.as_path()))
    {
        save_tokenizer_json(&tokenizer, &tokenizer_path)?;
        extra_paths.push(tokenizer_path);
    }

    let vocab_path = cfg
        .artifacts
        .vocab_json
        .as_ref()
        .map(|p| absolute_in_dir(base_dir, p.as_path()));
    let merges_path = cfg
        .artifacts
        .merges_txt
        .as_ref()
        .map(|p| absolute_in_dir(base_dir, p.as_path()));

    match (vocab_path, merges_path) {
        (Some(vocab), Some(merges)) => {
            save_bpe_files(tokenizer.get_model(), &vocab, &merges)?;
            extra_paths.push(vocab);
            extra_paths.push(merges);
        }
        (None, None) => {}
        _ => {
            return Err(Error::Artifact(
                "must provide both vocab_json and merges_txt when saving split artifacts".into(),
            ));
        }
    }

    let extra_refs: Vec<&Path> = extra_paths.iter().map(|p| p.as_path()).collect();
    let cfg_hash = compute_config_hash(cfg, &extra_refs)?;
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::Artifact(format!("failed to compute timestamp: {e}")))?
        .as_secs();
    let created_at = format!("unix:{created_at}");
    let token_count = tokenizer.get_vocab_size(true);

    if let Some(manifest_rel) = cfg.artifacts.manifest.as_ref() {
        let manifest_path = absolute_in_dir(base_dir, manifest_rel.as_path());
        let manifest = ArtifactManifest {
            cfg_hash,
            created_at,
            token_count,
        };
        write_manifest(&manifest_path, &manifest)?;
    }

    Ok(tokenizer)
}

fn absolute_in_dir(dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with(dir) {
        path.to_path_buf()
    } else {
        dir.join(path)
    }
}

fn save_bpe_files(model: &ModelWrapper, vocab_path: &Path, merges_path: &Path) -> Result<()> {
    let dir = vocab_path
        .parent()
        .ok_or_else(|| Error::Artifact("vocab path missing parent".into()))?;

    if dir
        != merges_path
            .parent()
            .ok_or_else(|| Error::Artifact("merges path missing parent".into()))?
    {
        return Err(Error::Artifact(
            "vocab and merges must share the same parent directory".into(),
        ));
    }

    std::fs::create_dir_all(dir)?;

    let temp_files = match model {
        ModelWrapper::BPE(bpe) => bpe.save(dir, None).map_err(Error::from)?,
        _ => return Err(Error::Artifact("only BPE models supported".into())),
    };

    let mut saved_vocab = None;
    let mut saved_merges = None;
    for path in temp_files {
        if path.file_name().map_or(false, |n| n == "vocab.json") {
            saved_vocab = Some(path);
        } else if path.file_name().map_or(false, |n| n == "merges.txt") {
            saved_merges = Some(path);
        }
    }

    let saved_vocab =
        saved_vocab.ok_or_else(|| Error::Artifact("BPE save did not produce vocab.json".into()))?;
    let saved_merges = saved_merges
        .ok_or_else(|| Error::Artifact("BPE save did not produce merges.txt".into()))?;

    if saved_vocab != *vocab_path {
        if vocab_path.exists() {
            std::fs::remove_file(vocab_path)?;
        }
        std::fs::rename(&saved_vocab, vocab_path)?;
    }
    if saved_merges != *merges_path {
        if merges_path.exists() {
            std::fs::remove_file(merges_path)?;
        }
        std::fs::rename(&saved_merges, merges_path)?;
    }

    Ok(())
}

struct CorpusIterator {
    files: VecDeque<PathBuf>,
    current: Option<FileState>,
    buffer: VecDeque<String>,
    rng: Option<StdRng>,
    max_lines: Option<usize>,
    produced: usize,
    error: Option<Error>,
}

struct FileState {
    reader: BufReader<File>,
}

impl CorpusIterator {
    fn new(mut inputs: Vec<PathBuf>, shuffle: bool, seed: u64, max_lines: Option<usize>) -> Self {
        let rng = if shuffle {
            let mut rng = StdRng::seed_from_u64(seed);
            inputs.shuffle(&mut rng);
            Some(rng)
        } else {
            None
        };

        Self {
            files: VecDeque::from(inputs),
            current: None,
            buffer: VecDeque::new(),
            rng,
            max_lines,
            produced: 0,
            error: None,
        }
    }

    fn take_error(&mut self) -> Option<Error> {
        self.error.take()
    }

    fn refill(&mut self) -> Result<bool> {
        while self.buffer.is_empty() {
            if self.should_finish() {
                return Ok(false);
            }

            if self.current.is_none() {
                let next = match self.files.pop_front() {
                    Some(path) => path,
                    None => return Ok(false),
                };
                let file = File::open(&next)?;
                self.current = Some(FileState {
                    reader: BufReader::new(file),
                });
            }

            let mut batch = Vec::with_capacity(SHUFFLE_CHUNK_SIZE);
            while batch.len() < SHUFFLE_CHUNK_SIZE {
                let mut line = String::new();
                let read = {
                    let state = self
                        .current
                        .as_mut()
                        .expect("reader state should exist while filling batch");
                    state.reader.read_line(&mut line)?
                };
                if read == 0 {
                    self.current = None;
                    break;
                }
                trim_line(&mut line);
                if line.is_empty() {
                    continue;
                }

                batch.push(line);

                if self.max_lines_reached_with(batch.len()) {
                    break;
                }
            }

            if batch.is_empty() {
                continue;
            }

            if let Some(rng) = self.rng.as_mut() {
                batch.shuffle(rng);
            }

            for line in batch {
                self.buffer.push_back(line);
            }
        }

        Ok(!self.buffer.is_empty())
    }

    fn should_finish(&self) -> bool {
        match self.max_lines {
            Some(limit) => self.produced >= limit,
            None => false,
        }
    }

    fn max_lines_reached_with(&self, batched: usize) -> bool {
        match self.max_lines {
            Some(limit) => self.produced + self.buffer.len() + batched >= limit,
            None => false,
        }
    }
}

impl Iterator for CorpusIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.error.is_some() || self.should_finish() {
            return None;
        }

        match self.refill() {
            Ok(true) => {
                if let Some(line) = self.buffer.pop_front() {
                    self.produced += 1;
                    Some(line)
                } else {
                    None
                }
            }
            Ok(false) => None,
            Err(err) => {
                self.error = Some(err);
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .max_lines
            .map(|limit| limit.saturating_sub(self.produced));
        (0, remaining)
    }
}

fn trim_line(line: &mut String) {
    while line.ends_with(['\r', '\n']) {
        line.pop();
    }
}
