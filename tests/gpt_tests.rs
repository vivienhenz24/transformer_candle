use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use transformer::{
    Block,
    FeedForward,
    GPTConfig,
    GPTLanguageModel,
    Head,
    MultiHeadAttention,
};

fn var_builder(device: &Device) -> VarBuilder {
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, device)
}

fn token_tensor(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &Device,
) -> Tensor {
    let data: Vec<u32> = (0..batch_size * seq_len)
        .map(|i| (i % vocab_size) as u32)
        .collect();
    Tensor::from_vec(data, (batch_size, seq_len), device).unwrap()
}

#[test]
fn test_head_creation() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let head = Head::new(512, 64, 128, 0.1, vb).unwrap();
    assert_eq!(head.head_size(), 64);
    assert_eq!(head.block_size(), 128);
}

#[test]
fn test_causal_mask() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let head = Head::new(512, 64, 128, 0.1, vb).unwrap();
    let mask = head.create_causal_mask(4, &device).unwrap();

    assert_eq!(mask.dims(), &[4, 4]);
}

#[test]
fn test_multi_head_attention() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let mha = MultiHeadAttention::new(512, 8, 128, 0.1, vb).unwrap();
    assert_eq!(mha.num_heads(), 8);
}

#[test]
fn test_six_head_attention() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 72;
    let n_head = 6;
    let mha = MultiHeadAttention::new(n_embd, n_head, 128, 0.1, vb).unwrap();

    assert_eq!(mha.num_heads(), 6);

    let batch_size = 2;
    let seq_len = 10;
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = mha.forward(&input, false).unwrap();

    assert_eq!(input.shape(), output.shape());
    assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, n_embd));
}

#[test]
fn test_feedforward_creation() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    FeedForward::new(512, 0.1, vb).unwrap();
}

#[test]
fn test_feedforward_shape_preservation() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 64;
    let ff = FeedForward::new(n_embd, 0.1, vb).unwrap();

    let batch_size = 3;
    let seq_len = 12;
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = ff.forward(&input, false).unwrap();

    assert_eq!(input.shape(), output.shape());
    assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, n_embd));
}

#[test]
fn test_feedforward_training_mode() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 48;
    let ff = FeedForward::new(n_embd, 0.5, vb).unwrap();

    let batch_size = 2;
    let seq_len = 8;
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let train_output = ff.forward(&input, true).unwrap();
    let eval_output = ff.forward(&input, false).unwrap();

    assert_eq!(input.shape(), train_output.shape());
    assert_eq!(input.shape(), eval_output.shape());
}

#[test]
fn test_feedforward_expansion_contraction() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 32;
    let ff = FeedForward::new(n_embd, 0.0, vb).unwrap();

    let batch_size = 1;
    let seq_len = 1;
    let input = Tensor::ones((batch_size, seq_len, n_embd), DType::F32, &device).unwrap();

    let output = ff.forward(&input, false).unwrap();

    assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, n_embd));
}

#[test]
fn test_block_creation() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    Block::new(512, 8, 128, 0.1, vb).unwrap();
}

#[test]
fn test_block_shape_preservation() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 72;
    let n_head = 6;
    let block_size = 32;
    let dropout_rate = 0.1;

    let block = Block::new(n_embd, n_head, block_size, dropout_rate, vb).unwrap();

    let batch_size = 2;
    let seq_len = 16;
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = block.forward(&input, false).unwrap();

    assert_eq!(input.shape(), output.shape());
    assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, n_embd));
}

#[test]
fn test_block_residual_connections() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 48;
    let n_head = 4;
    let block_size = 16;
    let dropout_rate = 0.0;

    let block = Block::new(n_embd, n_head, block_size, dropout_rate, vb).unwrap();

    let batch_size = 1;
    let seq_len = 8;
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = block.forward(&input, false).unwrap();

    assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, n_embd));
}

#[test]
fn test_block_training_mode() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let n_embd = 60;
    let n_head = 5;
    let block_size = 20;
    let dropout_rate = 0.3;

    let block = Block::new(n_embd, n_head, block_size, dropout_rate, vb).unwrap();

    let batch_size = 3;
    let seq_len = 12;
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let train_output = block.forward(&input, true).unwrap();
    let eval_output = block.forward(&input, false).unwrap();

    assert_eq!(train_output.dims3().unwrap(), (batch_size, seq_len, n_embd));
    assert_eq!(eval_output.dims3().unwrap(), (batch_size, seq_len, n_embd));
}

#[test]
fn test_language_model_creation() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let config = GPTConfig {
        vocab_size: 10,
        block_size: 12,
        n_embd: 12,
        n_layer: 1,
        n_head: 3,
        dropout_rate: 0.1,
    };

    let model = GPTLanguageModel::new(config, vb).unwrap();
    assert!(model.count_parameters() > 0);
}

#[test]
fn test_language_model_forward_shapes() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let config = GPTConfig {
        vocab_size: 65,
        block_size: 32,
        n_embd: 48,
        n_layer: 2,
        n_head: 4,
        dropout_rate: 0.1,
    };

    let model = GPTLanguageModel::new(config.clone(), vb).unwrap();
    let input = token_tensor(4, config.block_size, config.vocab_size, &device);

    let (logits, loss) = model.forward(&input, None, false).unwrap();

    assert_eq!(logits.dims3().unwrap(), (4, config.block_size, config.vocab_size));
    assert!(loss.is_none());
}

#[test]
fn test_language_model_forward_with_targets() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let config = GPTConfig {
        vocab_size: 65,
        block_size: 32,
        n_embd: 48,
        n_layer: 2,
        n_head: 4,
        dropout_rate: 0.1,
    };

    let model = GPTLanguageModel::new(config.clone(), vb).unwrap();
    let input = token_tensor(4, config.block_size, config.vocab_size, &device);

    let (_logits, loss) = model.forward(&input, Some(&input), false).unwrap();

    assert!(loss.is_some());
}

#[test]
fn test_sample_next_token() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let config = GPTConfig {
        vocab_size: 65,
        block_size: 32,
        n_embd: 48,
        n_layer: 2,
        n_head: 4,
        dropout_rate: 0.1,
    };

    let model = GPTLanguageModel::new(config.clone(), vb).unwrap();
    let input = token_tensor(1, 5, config.vocab_size, &device);

    let (logits, _) = model.forward(&input, None, false).unwrap();
    let (_batch, seq_len, _vocab) = logits.dims3().unwrap();
    let last_logits = logits.narrow(1, seq_len - 1, 1).unwrap().squeeze(1).unwrap();

    let next_token = model
        .sample_next_token(&last_logits, 0.8, Some(40), Some(0.95))
        .unwrap();

    assert_eq!(next_token.dims2().unwrap(), (1, 1));
    let value = next_token.to_vec2::<u32>().unwrap()[0][0] as usize;
    assert!(value < config.vocab_size);
}

#[test]
fn test_count_parameters() {
    let device = Device::Cpu;
    let vb = var_builder(&device);

    let config = GPTConfig {
        vocab_size: 10,
        block_size: 12,
        n_embd: 12,
        n_layer: 1,
        n_head: 3,
        dropout_rate: 0.1,
    };

    let model = GPTLanguageModel::new(config, vb).unwrap();
    assert!(model.count_parameters() > 0);
}
