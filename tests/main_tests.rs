use std::path::Path;

use candle_core::Device;
use transformer::{create_medium_gpt_config, setup_device, CharTokenizer};

#[test]
fn test_device_setup() {
    let device = setup_device();
    assert!(device.is_ok());
}

#[test]
fn test_data_path_exists() {
    let data_path = "pt-data/input.txt";
    assert!(
        Path::new(data_path).exists(),
        "Shakespeare data file should exist"
    );
}

#[test]
fn test_model_config_creation() {
    let config = create_medium_gpt_config(65);
    assert_eq!(config.vocab_size, 65);
    assert!(config.n_embd > 0);
    assert!(config.n_layer > 0);
    assert!(config.n_head > 0);
    assert!(config.block_size > 0);
}

#[test]
fn test_tokenizer_integration() {
    if Path::new("pt-data/input.txt").exists() {
        let tokenizer = CharTokenizer::from_file("pt-data/input.txt", Device::Cpu);
        assert!(tokenizer.is_ok());

        if let Ok(tokenizer) = tokenizer {
            assert!(tokenizer.vocab_size > 0);

            let test_text = "Hello";
            let encoded = tokenizer.encode(test_text);
            let decoded = tokenizer.decode(&encoded);
            assert_eq!(test_text, decoded);
        }
    }
}
