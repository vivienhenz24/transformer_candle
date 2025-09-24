use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use cascade_core::{
    AdaptiveSamplingConfig,
    AdaptiveWindowAttention,
    AdaptiveWindowConfig,
    CascadeTransformerConfig,
    CascadeTransformer,
};

fn var_builder(device: &Device) -> VarBuilder {
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, device)
}

fn token_tensor(batch: usize, seq: usize, vocab: usize, device: &Device) -> Tensor {
    let data: Vec<u32> = (0..batch * seq).map(|i| (i % vocab) as u32).collect();
    Tensor::from_vec(data, (batch, seq), device).unwrap()
}

#[test]
fn adaptive_window_masks_sequence() {
    let device = Device::Cpu;
    let vb = var_builder(&device);
    let config = cascade_core::attention::AttentionConfig {
        n_embd: 64,
        n_head: 4,
        block_size: 32,
        dropout: 0.0,
    };
    let attention = AdaptiveWindowAttention::new(config, AdaptiveWindowConfig::default(), vb).unwrap();
    let input = Tensor::randn(0.0f32, 1.0f32, (1, 16, 64), &device).unwrap();
    let output = attention.forward(&input, false).unwrap();
    assert_eq!(output.dims3().unwrap(), (1, 16, 64));
}

#[test]
fn cascade_transformer_forward() {
    let device = Device::Cpu;
    let mut config = CascadeTransformerConfig::default();
    config.vocab_size = 32;
    config.block_size = 16;
    config.n_layer = 2;
    config.n_head = 4;
    config.n_embd = 48;
    config.embedding.vocab_size = 32;
    config.embedding.block_size = 16;
    config.embedding.n_embd = 48;
    let vb = var_builder(&device);
    let model = CascadeTransformer::new(config, vb).unwrap();
    let input = token_tensor(2, 16, 32, &device);
    let (logits, loss) = model.forward(&input, Some(&input), false).unwrap();
    assert_eq!(logits.dims3().unwrap(), (2, 16, 32));
    assert!(loss.is_some());
}

#[test]
fn cascade_generation_produces_tokens() {
    let device = Device::Cpu;
    let mut config = CascadeTransformerConfig::default();
    config.vocab_size = 16;
    config.block_size = 8;
    config.n_layer = 1;
    config.n_head = 4;
    config.n_embd = 32;
    config.embedding.vocab_size = 16;
    config.embedding.block_size = 8;
    config.embedding.n_embd = 32;
    let vb = var_builder(&device);
    let model = CascadeTransformer::new(config, vb).unwrap();
    let context = token_tensor(1, 4, 16, &device);
    let config = AdaptiveSamplingConfig::default();
    let generated = model.generate(&context, config).unwrap();
    assert!(generated.dims2().unwrap().1 > 4);
}
