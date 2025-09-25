use candle_core::Device;
use transformer_tokenization::{AdvancedTokenizer, DataSplit, TokenizerConfig};

#[test]
fn test_encode_decode_roundtrip() {
    let text = "To be, or not to be";
    let tokenizer =
        AdvancedTokenizer::from_text(text, Device::Cpu, TokenizerConfig::default()).unwrap();
    let encoded = tokenizer.encode("To be").unwrap();
    let decoded = tokenizer.decode(&encoded).unwrap();
    assert!(decoded.contains("To"));
}

#[test]
fn test_batch_shapes() {
    let text = (0..1024)
        .map(|i| format!("token{}_", i))
        .collect::<String>();
    let mut cfg = TokenizerConfig::default();
    cfg.vocab_size = 128;
    cfg.validation_fraction = 0.1;
    let tokenizer = AdvancedTokenizer::from_text(&text, Device::Cpu, cfg).unwrap();
    let max_block = tokenizer.train_len().saturating_sub(2);
    assert!(
        max_block > 0,
        "tokenizer should yield enough tokens for batching"
    );
    let block_size = max_block.min(32).max(2);
    let (inputs, targets) = tokenizer
        .get_batch(DataSplit::Train, 4, block_size)
        .expect("batch should be available");
    assert_eq!(inputs.dims2().unwrap(), (4, block_size));
    assert_eq!(targets.dims2().unwrap(), (4, block_size));
}
