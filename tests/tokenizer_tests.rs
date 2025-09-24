use candle_core::Device;
use transformer::CharTokenizer;

#[test]
fn test_encode_decode() {
    let text = "hello world";
    let tokenizer = CharTokenizer::from_text(text, Device::Cpu).unwrap();

    let encoded = tokenizer.encode("hello");
    let decoded = tokenizer.decode(&encoded);
    assert_eq!(decoded, "hello");
}

#[test]
fn test_vocab_creation() {
    let text = "abcabc";
    let tokenizer = CharTokenizer::from_text(text, Device::Cpu).unwrap();

    assert_eq!(tokenizer.vocab_size, 3);
    assert!(tokenizer.char_to_idx.contains_key(&'a'));
    assert!(tokenizer.char_to_idx.contains_key(&'b'));
    assert!(tokenizer.char_to_idx.contains_key(&'c'));
}

#[test]
fn test_data_split() {
    let text = "0123456789".repeat(10); // 100 characters
    let tokenizer = CharTokenizer::from_text(&text, Device::Cpu).unwrap();

    assert_eq!(tokenizer.train_size(), 90);
    assert_eq!(tokenizer.val_size(), 10);
}
