use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use attention::core::{Config as AttentionConfig, RopeMode};
use attention::masks::build_causal_mask;
use candle_core::{DType, Error, Result, Tensor, Var};
use embedding::token::{TokenEmbedding, TokenEmbeddingConfig};
use layers::{dtypes::PrecisionPolicy, norm::NormalizationLayer};

use crate::{
    block::{build_norm, DecoderBlock},
    config::ModelConfig,
};

/// Overrides applied on top of a base [`ModelConfig`].
#[derive(Debug, Clone, Default)]
pub struct ModelConfigOverrides {
    pub attn_dropout_p: Option<Option<f32>>,
    pub residual_dropout_p: Option<Option<f32>>,
    pub dtype: Option<DType>,
    pub rope_mode: Option<RopeMode>,
}

struct MaskCacheEntry {
    batch: usize,
    seq: usize,
    tensor: Tensor,
}

struct PositionCacheEntry {
    seq: usize,
    tensor: Tensor,
}

/// Decoder-only transformer assembled from the shared crates.
pub struct Model {
    config: ModelConfig,
    embedding: TokenEmbedding,
    blocks: Vec<DecoderBlock>,
    final_norm: Arc<dyn NormalizationLayer>,
    policy: PrecisionPolicy,
    uses_preapply_rope: bool,
    training: AtomicBool,
    mask_cache: RefCell<Option<MaskCacheEntry>>,
    position_cache: RefCell<Option<PositionCacheEntry>>,
}

impl Model {
    /// Builds the model and its component blocks according to `config`.
    pub fn new(config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let policy = PrecisionPolicy::from_parameter_dtype(config.dtype);

        println!(
            "[model crate] constructing decoder-only model: layers={} hidden={} heads={} rope_mode={:?}",
            config.n_layers, config.hidden_dim, config.n_heads, config.rope_mode
        );

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
        let rope_mode = config.rope_mode.clone();
        attn_cfg.rope_mode = rope_mode.clone();
        attn_cfg.use_padding_mask = true;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            blocks.push(DecoderBlock::new(layer, &config, attn_cfg.clone())?);
        }

        let uses_preapply_rope = matches!(rope_mode, RopeMode::Preapply);

        let model = Self {
            config,
            embedding,
            blocks,
            final_norm,
            policy,
            uses_preapply_rope,
            training: AtomicBool::new(true),
            mask_cache: RefCell::new(None),
            position_cache: RefCell::new(None),
        };

        model.set_training(true);
        Ok(model)
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

        let mask = self.cached_causal_mask(batch, seq)?;
        let positions = if self.uses_preapply_rope {
            Some(self.cached_positions(seq)?)
        } else {
            None
        };

        for block in &self.blocks {
            hidden = block.forward(&hidden, Some(&mask), positions.as_ref())?;
        }

        let normalized = self.final_norm.forward(&hidden, &self.policy)?;
        self.embedding.linear_out(&normalized)
    }

    fn cached_causal_mask(&self, batch: usize, seq: usize) -> Result<Tensor> {
        {
            let cache = self.mask_cache.borrow();
            if let Some(entry) = cache.as_ref() {
                if entry.batch == batch && entry.seq == seq {
                    return Ok(entry.tensor.clone());
                }
            }
        }

        let tensor = build_causal_mask(&self.config.device, batch, self.config.n_heads, seq, seq)?;
        let mut cache = self.mask_cache.borrow_mut();
        *cache = Some(MaskCacheEntry {
            batch,
            seq,
            tensor: tensor.clone(),
        });
        Ok(tensor)
    }

    fn cached_positions(&self, seq: usize) -> Result<Tensor> {
        {
            let cache = self.position_cache.borrow();
            if let Some(entry) = cache.as_ref() {
                if entry.seq == seq {
                    return Ok(entry.tensor.clone());
                }
            }
        }

        let tensor = build_positions(seq)?;
        let mut cache = self.position_cache.borrow_mut();
        *cache = Some(PositionCacheEntry {
            seq,
            tensor: tensor.clone(),
        });
        Ok(tensor)
    }

    /// Toggle training mode for the model and its blocks.
    pub fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
        for block in &self.blocks {
            block.set_training(training);
        }
    }

    /// Returns the current training flag.
    pub fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    /// Returns the trainable parameters for the model.
    pub fn parameters(&self) -> Vec<(String, Var)> {
        let mut params = Vec::new();
        params.extend(self.embedding.named_parameters("embedding"));
        for (idx, block) in self.blocks.iter().enumerate() {
            let scope = format!("blocks.{}", idx);
            params.extend(block.parameters(&scope));
        }
        params.extend(self.final_norm.named_parameters("final_norm"));
        params
    }

    /// Legacy accessor preserved for downstream crates.
    pub fn named_parameters(&self) -> Vec<(String, Var)> {
        self.parameters()
    }
}

fn build_positions(seq_len: usize) -> Result<Tensor> {
    if seq_len == 0 {
        return Err(Error::Msg("sequence length must be non-zero".into()));
    }
    let data: Vec<u32> = (0..seq_len).map(|idx| idx as u32).collect();
    Tensor::from_vec(data, seq_len, &candle_core::Device::Cpu)
}

/// Compose a [`ModelConfig`] from a base configuration and override hints.
pub fn build_model_config_with_overrides(
    mut base: ModelConfig,
    overrides: ModelConfigOverrides,
) -> Result<ModelConfig> {
    if let Some(attn_dropout) = overrides.attn_dropout_p {
        base.attn_dropout_p = attn_dropout;
    }
    if let Some(residual_dropout) = overrides.residual_dropout_p {
        base.residual_dropout_p = residual_dropout;
    }
    if let Some(dtype) = overrides.dtype {
        base.dtype = dtype;
    }
    if let Some(mode) = overrides.rope_mode {
        base.rope_mode = mode;
    }
    base.validate()?;
    Ok(base)
}
