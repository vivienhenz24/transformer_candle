use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use cascade_core::{AdaptiveSamplingConfig, CascadeTransformerBuilder};

fn var_builder(device: &Device) -> VarBuilder {
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, device)
}

fn token_tensor(batch_size: usize, seq_len: usize, vocab_size: usize, device: &Device) -> Tensor {
    let data: Vec<u32> = (0..batch_size * seq_len)
        .map(|i| (i % vocab_size) as u32)
        .collect();
    Tensor::from_vec(data, (batch_size, seq_len), device).unwrap()
}

#[test]
fn cascade_transformer_forward_shapes() {
    let device = Device::Cpu;
    let vb = var_builder(&device);
    let builder = CascadeTransformerBuilder::new(128)
        .block_size(32)
        .model_width(64)
        .layers(2, 4)
        .dropout(0.05);
    let model = builder.build(vb).unwrap();

    let inputs = token_tensor(2, 16, 128, &device);
    let (logits, loss) = model.forward(&inputs, None, false).unwrap();

    assert_eq!(logits.dims3().unwrap(), (2, 16, 128));
    assert!(loss.is_none());
}

#[test]
fn cascade_transformer_generate_extends_sequence() {
    let device = Device::Cpu;
    let vb = var_builder(&device);
    let builder = CascadeTransformerBuilder::new(96)
        .block_size(48)
        .model_width(72)
        .layers(3, 6);
    let model = builder.build(vb).unwrap();

    let context = token_tensor(1, 6, 96, &device);
    let mut config = AdaptiveSamplingConfig::default();
    config.max_tokens = 4;
    config.min_tokens = 2;

    let generated = model.generate(&context, config).unwrap();
    let (batch, seq_len) = generated.dims2().unwrap();

    assert_eq!(batch, 1);
    assert!(seq_len >= 8, "generation should extend the sequence");
}

#[test]
fn cascade_transformer_parameter_count_positive() {
    let device = Device::Cpu;
    let vb = var_builder(&device);
    let builder = CascadeTransformerBuilder::new(64)
        .block_size(32)
        .model_width(48)
        .layers(2, 4);
    let model = builder.build(vb).unwrap();

    assert!(model.count_parameters() > 0);
}
