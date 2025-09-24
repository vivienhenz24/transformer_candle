use candle_core::{Result as CandleResult, Tensor};

#[derive(Debug, Clone)]
pub struct MemoryCompressionConfig {
    pub compress_ratio: f32,
    pub min_tokens: usize,
}

impl Default for MemoryCompressionConfig {
    fn default() -> Self {
        Self {
            compress_ratio: 0.5,
            min_tokens: 64,
        }
    }
}

#[derive(Debug)]
pub struct MemoryCompressor {
    config: MemoryCompressionConfig,
}

impl MemoryCompressor {
    pub fn new(config: MemoryCompressionConfig) -> Self {
        Self { config }
    }

    pub fn compress(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        let (_batch, seq_len, _hidden) = tensor.dims3()?;
        if seq_len <= self.config.min_tokens {
            return Ok(tensor.clone());
        }
        let keep = ((seq_len as f32) * self.config.compress_ratio).round() as usize;
        let keep = keep.max(self.config.min_tokens).min(seq_len);
        let stride = seq_len.max(keep) / keep.max(1);
        let mut slices = Vec::new();
        let mut index = 0usize;
        while index < seq_len {
            let end = (index + 1).min(seq_len);
            slices.push(tensor.narrow(1, index, end - index)?);
            index += stride;
        }
        Tensor::cat(&slices, 1)
    }
}
