use anyhow::Result;
use attention::core::RopeMode;
use candle_core::{DType, Device, Tensor};
use layers::norm::NormKind;
use model::{Model, ModelConfig};

fn build_config(rope_mode: RopeMode) -> ModelConfig {
    ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        n_layers: 2,
        n_heads: 2,
        head_dim: 4,
        ff_ratio: 4.0,
        norm_kind: NormKind::LayerNorm,
        rope_mode,
        dtype: DType::F32,
        device: Device::Cpu,
        attn_dropout_p: None,
        residual_dropout_p: None,
    }
}

#[test]
fn forward_produces_logits() -> Result<()> {
    let model = Model::new(build_config(RopeMode::Off))?;
    let token_ids = Tensor::from_slice(&[0i64, 1, 2, 3, 4, 5], (2, 3), &Device::Cpu)?;

    let logits = model.forward(&token_ids)?;

    assert_eq!(logits.dims(), &[2, 3, 16]);
    assert_eq!(logits.dtype(), DType::F32);
    Ok(())
}

#[test]
fn forward_with_preapplied_rope() -> Result<()> {
    let model = Model::new(build_config(RopeMode::Preapply))?;
    let token_ids = Tensor::from_slice(&[0i64, 1, 2, 3], (1, 4), &Device::Cpu)?;

    let logits = model.forward(&token_ids)?;

    assert_eq!(logits.dims(), &[1, 4, 16]);
    assert_eq!(logits.dtype(), DType::F32);
    Ok(())
}
