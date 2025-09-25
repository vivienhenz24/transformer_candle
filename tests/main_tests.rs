use std::path::Path;

use cascade_core::CascadeTransformerBuilder;
use cascade_training::setup_device;
use transformer_tokenization::TokenizerConfig;

#[test]
fn test_device_setup() {
    let device = setup_device();
    assert!(device.is_ok());
}

#[test]
fn test_workspace_has_dataset() {
    assert!(Path::new("pt-data/input.txt").exists());
}

#[test]
fn test_builder_defaults() {
    let builder = CascadeTransformerBuilder::new(128);
    let config = builder.config();
    assert_eq!(config.vocab_size, 128);
    assert!(config.n_embd > 0);
}

#[test]
fn tokenizer_config_defaults() {
    let cfg = TokenizerConfig::default();
    assert!(cfg.vocab_size > 0);
    assert!(cfg.validation_fraction > 0.0);
}
