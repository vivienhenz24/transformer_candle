use candle_core::Device;
use transformer::{
    create_medium_gpt_config,
    create_small_gpt_config,
    TrainingConfig,
    TrainingStats,
};

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();
    assert_eq!(config.learning_rate, 6e-4);
    assert_eq!(config.batch_size, 64);
    assert_eq!(config.block_size, 256);
    assert!(config.max_iters > 0);
    assert!(config.eval_interval > 0);
    assert_eq!(config.beta1, 0.9);
    assert_eq!(config.beta2, 0.95);
    assert!(matches!(config.device, Device::Cpu));
}

#[test]
fn test_small_gpt_config() {
    let config = create_small_gpt_config(50);
    assert_eq!(config.vocab_size, 50);
    assert_eq!(config.block_size, 64);
    assert_eq!(config.n_embd, 64);
    assert_eq!(config.n_head, 4);
    assert_eq!(config.n_layer, 2);
    assert_eq!(config.dropout_rate, 0.0);
}

#[test]
fn test_medium_gpt_config() {
    let config = create_medium_gpt_config(100);
    assert_eq!(config.vocab_size, 100);
    assert_eq!(config.block_size, 256);
    assert_eq!(config.n_embd, 384);
    assert_eq!(config.n_head, 6);
    assert_eq!(config.n_layer, 6);
    assert_eq!(config.dropout_rate, 0.1);
}

#[test]
fn test_training_stats_creation() {
    let stats = TrainingStats {
        iteration: 100,
        train_loss: 2.5,
        val_loss: 2.7,
        learning_rate: 6e-4,
        tokens_per_sec: 1000.0,
        elapsed_time: 120.0,
    };

    assert_eq!(stats.iteration, 100);
    assert_eq!(stats.train_loss, 2.5);
    assert_eq!(stats.val_loss, 2.7);
    assert_eq!(stats.learning_rate, 6e-4);
    assert_eq!(stats.tokens_per_sec, 1000.0);
    assert_eq!(stats.elapsed_time, 120.0);
}
