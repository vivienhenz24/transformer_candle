use attention::core::RopeMode;
use candle_core::{DType, Device, Error, Result};
use layers::norm::NormKind;

/// High-level configuration for assembling the decoder-only transformer.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ff_ratio: f32,
    pub norm_kind: NormKind,
    pub rope_mode: RopeMode,
    pub dtype: DType,
    pub device: Device,
    pub attn_dropout_p: Option<f32>,
    pub residual_dropout_p: Option<f32>,
}

impl ModelConfig {
    /// Validate structural invariants derived from the project ground rules.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(Error::Msg("vocab_size must be greater than zero".into()));
        }
        if self.hidden_dim == 0 {
            return Err(Error::Msg("hidden_dim must be greater than zero".into()));
        }
        if self.n_layers == 0 {
            return Err(Error::Msg("n_layers must be greater than zero".into()));
        }
        if self.n_heads == 0 {
            return Err(Error::Msg("n_heads must be greater than zero".into()));
        }
        if self.head_dim == 0 {
            return Err(Error::Msg("head_dim must be greater than zero".into()));
        }
        if self.n_heads * self.head_dim != self.hidden_dim {
            return Err(Error::Msg(format!(
                "hidden_dim ({}) must equal n_heads ({}) * head_dim ({})",
                self.hidden_dim, self.n_heads, self.head_dim
            )));
        }
        if self.ff_ratio <= 0.0 {
            return Err(Error::Msg("ff_ratio must be positive".into()));
        }
        if let Some(p) = self.attn_dropout_p {
            if !(0.0..1.0).contains(&p) {
                return Err(Error::Msg("attn_dropout_p must be in [0, 1)".into()));
            }
        }
        if let Some(p) = self.residual_dropout_p {
            if !(0.0..1.0).contains(&p) {
                return Err(Error::Msg(
                    "residual_dropout_p must be in [0, 1)".into(),
                ));
            }
        }
        Ok(())
    }
}
