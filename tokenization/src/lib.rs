use anyhow::{Context, Result};
use candle_core::{Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};

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
        let tokenizer_impl = AdaptiveBpeTokenizer::train_from_text(&cleaned, config.clone())?;
        let encoding = tokenizer_impl.encode_to_ids(&cleaned);
        let split_idx = ((encoding.len() as f32) * (1.0 - config.validation_fraction)).max(1.0) as usize;
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

    pub fn from_file(path: &str, device: Device, config: TokenizerConfig) -> Result<Self> {
        let text = std::fs::read_to_string(path).context("failed to read corpus")?;
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

    pub fn get_batch(&self, split: DataSplit, batch_size: usize, block_size: usize) -> CandleResult<(Tensor, Tensor)> {
        let data = self.split_data(split);
        if data.len() <= block_size {
            return Err(candle_core::Error::Msg("data length must exceed block size".into()));
        }
        let max_start = data.len() - block_size - 1;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for _ in 0..batch_size {
            let start = fastrand::usize(0..max_start);
            inputs.extend_from_slice(&data[start..start + block_size]);
            targets.extend_from_slice(&data[start + 1..start + block_size + 1]);
        }
        let inputs = Tensor::from_vec(inputs, (batch_size, block_size), &self.device)?;
        let targets = Tensor::from_vec(targets, (batch_size, block_size), &self.device)?;
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
