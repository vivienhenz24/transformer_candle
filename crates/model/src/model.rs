use std::sync::Arc;

use attention::core::{Config as AttentionConfig, RopeMode};
use attention::masks::build_causal_mask;
use candle_core::{Error, Result, Tensor};
use embedding::token::{TokenEmbedding, TokenEmbeddingConfig};
use layers::{dtypes::PrecisionPolicy, norm::NormalizationLayer};

use crate::{
    block::{build_norm, DecoderBlock},
    config::ModelConfig,
};

/// Decoder-only transformer assembled from the shared crates.
pub struct Model {
    config: ModelConfig,
    embedding: TokenEmbedding,
    blocks: Vec<DecoderBlock>,
    final_norm: Arc<dyn NormalizationLayer>,
    policy: PrecisionPolicy,
    uses_preapply_rope: bool,
}

impl Model {
    /// Builds the model and its component blocks according to `config`.
    pub fn new(config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let policy = PrecisionPolicy::from_parameter_dtype(config.dtype);

        let embed_cfg = TokenEmbeddingConfig {
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_dim,
            dtype: config.dtype,
            device: config.device.clone(),
        };
        let embedding = TokenEmbedding::new(embed_cfg)?;

        let final_norm = build_norm(
            config.norm_kind,
            config.hidden_dim,
            config.dtype,
            &config.device,
        )?;

        let mut attn_cfg = AttentionConfig::from_env();
        attn_cfg.dropout_p = config.attn_dropout_p;
        attn_cfg.rope_mode = config.rope_mode;
        attn_cfg.use_padding_mask = false;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            blocks.push(DecoderBlock::new(layer, &config, attn_cfg.clone())?);
        }

        let uses_preapply_rope = matches!(config.rope_mode, RopeMode::Preapply);

        Ok(Self {
            config,
            embedding,
            blocks,
            final_norm,
            policy,
            uses_preapply_rope,
        })
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Produces logits shaped `(batch, seq, vocab_size)` for the provided token ids.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embedding.forward(token_ids)?;
        let dims = hidden.dims();
        if dims.len() != 3 {
            return Err(Error::Msg(format!(
                "token embeddings expected [batch, seq, hidden] got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let seq = dims[1];

        let mask = build_causal_mask(&self.config.device, batch, self.config.n_heads, seq, seq)?;
        let positions = if self.uses_preapply_rope {
            Some(build_positions(seq)?)
        } else {
            None
        };

        for block in &self.blocks {
            hidden = block.forward(&hidden, Some(&mask), positions.as_ref())?;
        }

        let normalized = self.final_norm.forward(&hidden, &self.policy)?;
        self.embedding.linear_out(&normalized)
    }
}

fn build_positions(seq_len: usize) -> Result<Tensor> {
    if seq_len == 0 {
        return Err(Error::Msg("sequence length must be non-zero".into()));
    }
    let data: Vec<u32> = (0..seq_len).map(|idx| idx as u32).collect();
    Tensor::from_vec(data, seq_len, &candle_core::Device::Cpu)
}
