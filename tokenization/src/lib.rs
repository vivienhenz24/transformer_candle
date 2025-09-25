use anyhow::{Context, Result};
use candle_core::{Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use std::{env, io::Write, path::Path};

pub mod processors;
pub mod subword;
pub mod vocabularies;

pub use processors::preprocessing::PreprocessorChain;
pub use processors::streaming::StreamingTokenizer;
pub use subword::adaptive_bpe::{AdaptiveBpeConfig, AdaptiveBpeTokenizer};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataSplit {
    Train,
    Val,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub validation_fraction: f32,
    pub unk_token: String,
    pub pad_token: Option<String>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 4096,
            validation_fraction: 0.1,
            unk_token: "<unk>".to_string(),
            pad_token: Some("<pad>".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdvancedTokenizer {
    tokenizer: AdaptiveBpeTokenizer,
    train_data: Vec<u32>,
    val_data: Vec<u32>,
    device: Device,
}

impl AdvancedTokenizer {
    pub fn from_text(text: &str, device: Device, config: TokenizerConfig) -> Result<Self> {
        let preprocessor = processors::preprocessing::PreprocessorChain::default();
        let cleaned = preprocessor.run(text);
        let slice = select_training_slice(&cleaned);
        if let Some(source) = slice.reason {
            if slice.truncated {
                println!(
                    "Tokenizer: using {} chars ({} truncated) for merge training [{}]",
                    slice.effective_len,
                    cleaned.len().saturating_sub(slice.effective_len),
                    source
                );
            } else {
                println!(
                    "Tokenizer: corpus chars available for merge training {} [{}]",
                    slice.effective_len, source
                );
            }
        } else {
            println!(
                "Tokenizer: corpus chars available for merge training {}",
                slice.effective_len
            );
        }
        let tokenizer_impl = AdaptiveBpeTokenizer::train_from_text(slice.text, config.clone())?;
        println!(
            "Tokenizer: encoding corpus to token ids ({} chars)...",
            cleaned.len()
        );
        let _ = std::io::stdout().flush();
        let encode_started = std::time::Instant::now();
        let mut encoding = tokenizer_impl.encode_to_ids_with_progress(&cleaned, 200_000);
        println!(
            "Tokenizer: produced {} tokens in {:.2?}",
            encoding.len(),
            encode_started.elapsed()
        );
        let _ = std::io::stdout().flush();
        if let Some((limit, reason)) = tokenizer_token_limit() {
            if encoding.len() > limit {
                println!(
                    "Tokenizer: truncating tokens from {} to {} ({}).",
                    encoding.len(),
                    limit,
                    reason
                );
                encoding.truncate(limit);
            }
        }
        let split_idx =
            ((encoding.len() as f32) * (1.0 - config.validation_fraction)).max(1.0) as usize;
        let split_idx = split_idx.min(encoding.len().saturating_sub(1));
        let train_data = encoding[..split_idx].to_vec();
        let val_data = encoding[split_idx..].to_vec();

        Ok(Self {
            tokenizer: tokenizer_impl,
            train_data,
            val_data,
            device,
        })
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        device: Device,
        config: TokenizerConfig,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let text = std::fs::read_to_string(path_ref)
            .with_context(|| format!("failed to read corpus at {}", path_ref.display()))?;
        Self::from_text(&text, device, config)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode_to_ids(text))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode_ids(ids))
    }

    fn split_data(&self, split: DataSplit) -> &[u32] {
        match split {
            DataSplit::Train => &self.train_data,
            DataSplit::Val => &self.val_data,
        }
    }

    pub fn train_len(&self) -> usize {
        self.train_data.len()
    }

    pub fn val_len(&self) -> usize {
        self.val_data.len()
    }

    pub fn get_batch(
        &self,
        split: DataSplit,
        batch_size: usize,
        block_size: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let data = self.split_data(split);
        if data.len() <= block_size {
            return Err(candle_core::Error::Msg(
                "data length must exceed block size".into(),
            ));
        }
        let max_start = data.len() - block_size - 1;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for _ in 0..batch_size {
            let start = fastrand::usize(0..max_start);
            inputs.extend_from_slice(&data[start..start + block_size]);
            targets.extend_from_slice(&data[start + 1..start + block_size + 1]);
        }
        let inputs = Tensor::from_vec(inputs, (batch_size, block_size), &Device::Cpu)?
            .to_device(&self.device)?;
        let targets = Tensor::from_vec(targets, (batch_size, block_size), &Device::Cpu)?
            .to_device(&self.device)?;
        Ok((inputs, targets))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.id_to_token.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

struct TrainingSlice<'a> {
    text: &'a str,
    truncated: bool,
    effective_len: usize,
    reason: Option<&'static str>,
}

fn select_training_slice(text: &str) -> TrainingSlice {
    if let Some(limit) = parse_env_limit("TOKENIZER_CHAR_LIMIT") {
        if limit == 0 {
            return TrainingSlice {
                text,
                truncated: false,
                effective_len: text.len(),
                reason: Some("TOKENIZER_CHAR_LIMIT (no cap)"),
            };
        }
        let capped = limit.min(text.len());
        return TrainingSlice {
            text: &text[..capped],
            truncated: capped < text.len(),
            effective_len: capped,
            reason: Some("TOKENIZER_CHAR_LIMIT"),
        };
    }

    let default_limit: usize = 2_000_000; // ~2 MB of text
    let capped = default_limit.min(text.len());
    TrainingSlice {
        text: &text[..capped],
        truncated: capped < text.len(),
        effective_len: capped,
        reason: if capped < text.len() {
            Some("default 2M char cap")
        } else {
            None
        },
    }
}

fn parse_env_limit(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn tokenizer_token_limit() -> Option<(usize, &'static str)> {
    if let Some(limit) = parse_env_limit("TOKENIZER_MAX_TOKENS") {
        if limit == 0 {
            return None;
        }
        return Some((limit, "TOKENIZER_MAX_TOKENS"));
    }
    Some((1_000_000, "default 1M token cap"))
}
