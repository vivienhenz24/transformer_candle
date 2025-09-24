use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, Module, VarBuilder};

pub mod adaptive_window;
pub mod cascade_attention;
pub mod hierarchical;
pub mod sparse_patterns;

pub use adaptive_window::AdaptiveWindowAttention;
pub use cascade_attention::CascadeAttentionComposer;
pub use hierarchical::HierarchicalAttention;
pub use sparse_patterns::{PatternSpec, SparseAttentionMask};

#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub dropout: f32,
}

impl AttentionConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head.max(1)
    }
}

#[derive(Debug)]
pub struct AttentionHead {
    key: Linear,
    query: Linear,
    value: Linear,
    dropout: Dropout,
    _head_size: usize,
    _block_size: usize,
    scale: f64,
}

impl AttentionHead {
    pub fn new(config: AttentionConfig, vb: VarBuilder) -> CandleResult<Self> {
        let head_size = config.head_dim();
        let key = candle_nn::linear_no_bias(config.n_embd, head_size, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(config.n_embd, head_size, vb.pp("query"))?;
        let value = candle_nn::linear_no_bias(config.n_embd, head_size, vb.pp("value"))?;
        let dropout = Dropout::new(config.dropout);
        let scale = 1.0 / (head_size as f64).sqrt();

        Ok(Self {
            key,
            query,
            value,
            dropout,
            _head_size: head_size,
            _block_size: config.block_size,
            scale,
        })
    }

    pub fn create_causal_mask(&self, seq_len: usize, device: &Device) -> CandleResult<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;
        let ones = Tensor::ones((seq_len, seq_len), DType::F32, device)?;
        let mask = ones.sub(&mask)?;
        mask.affine(-1e9, 0.0)
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> CandleResult<Tensor> {
        let (_batch, seq_len, _n_embd) = x.dims3()?;

        let queries = self.query.forward(x)?;
        let keys = self.key.forward(x)?;
        let values = self.value.forward(x)?;

        let keys_t = keys.transpose(1, 2)?;
        let mut scores = queries.matmul(&keys_t)?;
        scores = scores.affine(self.scale as f64, 0.0)?;

        let base_mask = self.create_causal_mask(seq_len, x.device())?;
        let base_mask = base_mask.unsqueeze(0)?.broadcast_as(scores.shape())?;
        scores = scores.add(&base_mask)?;

        if let Some(extra_mask) = mask {
            let broadcast = extra_mask.broadcast_as(scores.shape())?;
            scores = scores.add(&broadcast)?;
        }

        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let weights = if train {
            self.dropout.forward(&weights, train)?
        } else {
            weights
        };

        weights.matmul(&values)
    }
}

#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    heads: Vec<AttentionHead>,
    proj: Linear,
    dropout: Dropout,
}

impl MultiHeadSelfAttention {
    pub fn new(config: AttentionConfig, vb: VarBuilder) -> CandleResult<Self> {
        assert_eq!(config.n_embd % config.n_head, 0, "n_embd must be divisible by n_head");

        let mut heads = Vec::with_capacity(config.n_head);
        for idx in 0..config.n_head {
            let head = AttentionHead::new(config, vb.pp(&format!("head_{idx}")))?;
            heads.push(head);
        }

        let proj = candle_nn::linear(config.n_embd, config.n_embd, vb.pp("proj"))?;
        let dropout = Dropout::new(config.dropout);

        Ok(Self { heads, proj, dropout })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> CandleResult<Tensor> {
        let mut outputs = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            outputs.push(head.forward(x, mask, train)?);
        }
        let cat = Tensor::cat(&outputs, 2)?;
        let projected = self.proj.forward(&cat)?;
        if train {
            self.dropout.forward(&projected, train)
        } else {
            Ok(projected)
        }
    }

    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }
}

#[derive(Debug)]
pub struct ResidualAttention {
    pub attention: MultiHeadSelfAttention,
    pub norm: LayerNorm,
}

impl ResidualAttention {
    pub fn new(config: AttentionConfig, vb: VarBuilder) -> CandleResult<Self> {
        let attention = MultiHeadSelfAttention::new(config, vb.pp("attn"))?;
        let norm = candle_nn::layer_norm(config.n_embd, 1e-5, vb.pp("ln"))?;
        Ok(Self { attention, norm })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> CandleResult<Tensor> {
        let normed = self.norm.forward(x)?;
        let attn = self.attention.forward(&normed, mask, train)?;
        x.add(&attn)
    }
}

pub fn build_attention_mask(pattern: Option<&SparseAttentionMask>, device: &Device, seq_len: usize) -> CandleResult<Option<Tensor>> {
    if let Some(pattern) = pattern {
        pattern.as_tensor(device, seq_len).map(Some)
    } else {
        Ok(None)
    }
}

pub fn blend_attention_outputs(primary: &Tensor, secondary: &Tensor, alpha: f32) -> CandleResult<Tensor> {
    let alpha = alpha.clamp(0.0, 1.0) as f64;
    let primary_scaled = primary.affine(alpha, 0.0)?;
    let secondary_scaled = secondary.affine(1.0 - alpha, 0.0)?;
    primary_scaled.add(&secondary_scaled)
}
