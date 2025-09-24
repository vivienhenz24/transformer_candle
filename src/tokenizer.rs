use anyhow::{Context, Result};
use candle_core::{Device, Result as CandleResult, Tensor};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Character-level tokenizer for transformer training
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    /// Vocabulary size (number of unique characters)
    pub vocab_size: usize,
    /// Character to index mapping
    pub char_to_idx: HashMap<char, usize>,
    /// Index to character mapping
    pub idx_to_char: Vec<char>,
    /// Training data as indices
    pub train_data: Vec<usize>,
    /// Validation data as indices
    pub val_data: Vec<usize>,
    /// Device for tensor operations
    pub device: Device,
}

impl CharTokenizer {
    /// Create a new tokenizer from a text file
    pub fn from_file<P: AsRef<Path>>(file_path: P, device: Device) -> Result<Self> {
        let text = fs::read_to_string(file_path).context("Failed to read input file")?;

        Self::from_text(&text, device)
    }

    /// Create a new tokenizer from text string
    pub fn from_text(text: &str, device: Device) -> Result<Self> {
        // Get unique characters and sort them for consistent ordering
        let mut chars: Vec<char> = text
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort();

        let vocab_size = chars.len();

        // Build character to index mapping
        let char_to_idx: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        // Index to character mapping
        let idx_to_char = chars;

        // Encode the entire text
        let encoded_data: Vec<usize> = text.chars().map(|c| char_to_idx[&c]).collect();

        // Split into 90% training and 10% validation
        let split_idx = (encoded_data.len() as f64 * 0.9) as usize;
        let train_data = encoded_data[..split_idx].to_vec();
        let val_data = encoded_data[split_idx..].to_vec();

        Ok(CharTokenizer {
            vocab_size,
            char_to_idx,
            idx_to_char,
            train_data,
            val_data,
            device,
        })
    }

    /// Encode a string to a list of indices
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c))
            .cloned()
            .collect()
    }

    /// Decode a list of indices back to a string
    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter_map(|&idx| self.idx_to_char.get(idx))
            .collect()
    }

    /// Convert indices to a Candle tensor
    pub fn indices_to_tensor(&self, indices: &[usize]) -> CandleResult<Tensor> {
        let data: Vec<u32> = indices.iter().map(|&x| x as u32).collect();
        Tensor::new(data, &self.device)
    }

    /// Convert text directly to a Candle tensor
    pub fn text_to_tensor(&self, text: &str) -> CandleResult<Tensor> {
        let indices = self.encode(text);
        self.indices_to_tensor(&indices)
    }

    /// Get a random batch of training sequences
    /// Returns (inputs, targets) where targets are inputs shifted by 1
    pub fn get_batch(
        &self,
        split: DataSplit,
        batch_size: usize,
        block_size: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let data = match split {
            DataSplit::Train => &self.train_data,
            DataSplit::Val => &self.val_data,
        };

        if data.len() <= block_size {
            return Err(candle_core::Error::Msg(
                "Data length must be greater than block_size".to_string(),
            ));
        }

        let max_start_idx = data.len() - block_size - 1;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        // Generate random starting indices for each sequence in the batch
        for _ in 0..batch_size {
            let start_idx = fastrand::usize(0..max_start_idx);

            // Extract input sequence (block_size tokens)
            let input_seq: Vec<u32> = data[start_idx..start_idx + block_size]
                .iter()
                .map(|&x| x as u32)
                .collect();

            // Extract target sequence (shifted by 1)
            let target_seq: Vec<u32> = data[start_idx + 1..start_idx + block_size + 1]
                .iter()
                .map(|&x| x as u32)
                .collect();

            inputs.extend(input_seq);
            targets.extend(target_seq);
        }

        // Reshape to (batch_size, block_size)
        let input_tensor = Tensor::new(inputs, &self.device)?.reshape(&[batch_size, block_size])?;
        let target_tensor =
            Tensor::new(targets, &self.device)?.reshape(&[batch_size, block_size])?;

        Ok((input_tensor, target_tensor))
    }

    /// Get the training data size
    pub fn train_size(&self) -> usize {
        self.train_data.len()
    }

    /// Get the validation data size
    pub fn val_size(&self) -> usize {
        self.val_data.len()
    }

    /// Access the full training data sequence
    pub fn train_tokens(&self) -> &[usize] {
        &self.train_data
    }

    /// Access the full validation data sequence
    pub fn val_tokens(&self) -> &[usize] {
        &self.val_data
    }

    /// Print tokenizer statistics
    pub fn print_stats(&self) {
        println!("Tokenizer Statistics:");
        println!("  Vocabulary size: {}", self.vocab_size);
        println!("  Training data size: {} tokens", self.train_size());
        println!("  Validation data size: {} tokens", self.val_size());
        println!(
            "  Characters in vocabulary: {:?}",
            self.idx_to_char.iter().collect::<String>()
        );
    }

    /// Change the device for tensor operations
    pub fn to_device(&mut self, device: Device) {
        self.device = device;
    }
}

/// Enum for specifying which data split to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSplit {
    Train,
    Val,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let text = "hello world";
        let device = Device::Cpu;
        let tokenizer = CharTokenizer::from_text(text, device).unwrap();

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_vocab_creation() {
        let text = "abcabc";
        let device = Device::Cpu;
        let tokenizer = CharTokenizer::from_text(text, device).unwrap();

        assert_eq!(tokenizer.vocab_size, 3); // a, b, c
        assert!(tokenizer.char_to_idx.contains_key(&'a'));
        assert!(tokenizer.char_to_idx.contains_key(&'b'));
        assert!(tokenizer.char_to_idx.contains_key(&'c'));
    }

    #[test]
    fn test_data_split() {
        let text = "0123456789".repeat(10); // 100 characters
        let device = Device::Cpu;
        let tokenizer = CharTokenizer::from_text(&text, device).unwrap();

        // Should be 90/10 split
        assert_eq!(tokenizer.train_size(), 90);
        assert_eq!(tokenizer.val_size(), 10);
    }
}
