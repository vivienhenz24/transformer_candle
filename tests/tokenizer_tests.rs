use candle_core::Device;
use transformer_tokenization::{AdvancedTokenizer, DataSplit, TokenizerConfig};

#[test]
fn test_encode_decode_roundtrip() {
    let text = "To be, or not to be";
    let tokenizer = AdvancedTokenizer::from_text(text, Device::Cpu, TokenizerConfig::default()).unwrap();
    let encoded = tokenizer.encode("To be").unwrap();
    let decoded = tokenizer.decode(&encoded).unwrap();
    assert!(decoded.contains("To"));
}

#[test]
fn test_batch_shapes() {
    let text: String = (0..2000)
        .map(|i| char::from(b'a' + (i % 16) as u8))
        .collect();
    let mut cfg = TokenizerConfig::default();
    cfg.vocab_size = 128;
    cfg.validation_fraction = 0.05;
    let tokenizer = AdvancedTokenizer::from_text(&text, Device::Cpu, cfg).unwrap();
    let block_size = 4.min(tokenizer.train_len().saturating_sub(2));
    assert!(block_size > 0);
    let (inputs, targets) = tokenizer.get_batch(DataSplit::Train, 4, block_size).unwrap();
    assert_eq!(inputs.dims2().unwrap(), (4, block_size));
    assert_eq!(targets.dims2().unwrap(), (4, block_size));
}
