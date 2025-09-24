use candle_core::Device;
use cascade_core::CascadeTransformerBuilder;
use cascade_training::{TrainingConfig, TrainingStats};

#[test]
fn training_config_defaults() {
    let config = TrainingConfig::default();
    assert_eq!(config.learning_rate, 5e-4);
    assert_eq!(config.batch_size, 48);
    assert_eq!(config.block_size, 256);
    assert!(matches!(config.device, Device::Cpu));
}

#[test]
fn builder_applies_dimensions() {
    let builder = CascadeTransformerBuilder::new(100)
        .block_size(192)
        .model_width(256)
        .layers(3, 4);
    let config = builder.config().clone();
    assert_eq!(config.vocab_size, 100);
    assert_eq!(config.block_size, 192);
    assert_eq!(config.n_embd, 256);
    assert_eq!(config.n_layer, 3);
    assert_eq!(config.n_head, 4);
}

#[test]
fn training_stats_records() {
    let stats = TrainingStats {
        iteration: 10,
        train_loss: 1.2,
        val_loss: 1.5,
        tokens_per_sec: 900.0,
        elapsed_time: 12.0,
    };
    assert_eq!(stats.iteration, 10);
    assert_eq!(stats.train_loss, 1.2);
    assert_eq!(stats.val_loss, 1.5);
}
