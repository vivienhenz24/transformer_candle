use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct CascadeEmbeddingConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub rotary_scaling: Option<f32>,
    pub learned_frequency: bool,
}

impl Default for CascadeEmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65,
            block_size: 256,
            n_embd: 384,
            rotary_scaling: None,
            learned_frequency: true,
        }
    }
}

#[derive(Debug)]
pub struct CascadeEmbeddings {
    token: Embedding,
    position: Embedding,
    freq: Option<Embedding>,
    config: CascadeEmbeddingConfig,
}

impl CascadeEmbeddings {
    pub fn new(config: CascadeEmbeddingConfig, vb: VarBuilder) -> CandleResult<Self> {
        let token = candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let position = candle_nn::embedding(config.block_size, config.n_embd, vb.pp("wpe"))?;
        let freq = if config.learned_frequency {
            Some(candle_nn::embedding(config.block_size, config.n_embd, vb.pp("wfe"))?)
        } else {
            None
        };

        Ok(Self {
            token,
            position,
            freq,
            config,
        })
    }

    fn apply_rotary(&self, embeddings: &Tensor) -> CandleResult<Tensor> {
        if let Some(scale) = self.config.rotary_scaling {
            let (_batch, seq_len, n_embd) = embeddings.dims3()?;
            if n_embd < 2 {
                return Ok(embeddings.clone());
            }
            let half = n_embd / 2;
            let left = embeddings.narrow(2, 0, half)?;
            let right = embeddings.narrow(2, half, n_embd - half)?;
            let rot = Tensor::arange(0u32, seq_len as u32, embeddings.device())?
                .to_dtype(DType::F32)?
                .reshape(&[1, seq_len, 1])?
                .affine(scale as f64, 0.0)?;
            let sin = rot.sin()?.broadcast_as(left.shape())?;
            let cos = rot.cos()?.broadcast_as(left.shape())?;
            let left_rot = left.mul(&cos)?.add(&right.mul(&sin)?)?;
            let right_rot = right.mul(&cos)?.sub(&left.mul(&sin)?)?;
            Tensor::cat(&[left_rot, right_rot], 2)
        } else {
            Ok(embeddings.clone())
        }
    }

    pub fn forward(&self, idx: &Tensor) -> CandleResult<Tensor> {
        let tokens = self.token.forward(idx)?;
        let (_, seq_len) = idx.dims2()?;
        let pos_idx = Tensor::arange(0u32, seq_len as u32, idx.device())?;
        let pos_emb = self.position.forward(&pos_idx)?;
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(tokens.shape())?;

        let mut embeddings = tokens.add(&pos_emb)?;
        if let Some(freq) = &self.freq {
            let freq_emb = freq.forward(&pos_idx)?;
            let freq_emb = freq_emb.unsqueeze(0)?.broadcast_as(tokens.shape())?;
            embeddings = embeddings.add(&freq_emb)?;
        }

        self.apply_rotary(&embeddings)
    }

    pub fn config(&self) -> &CascadeEmbeddingConfig {
        &self.config
    }
}
