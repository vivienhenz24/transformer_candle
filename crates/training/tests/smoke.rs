use std::{fs, path::PathBuf};

use tempfile::tempdir;
use tokenizer::{
    config::{
        ArtifactsCfg as TokenizerArtifactsCfg, ByteLevelCfg as TokenizerByteLevelCfg,
        Config as TokenizerTrainConfig, ModelCfg as TokenizerModelCfg, PostCfg as TokenizerPostCfg,
        TrainingCfg as TokenizerTrainingCfg,
    },
    train_bbpe,
};
use training::{
    config::{
        BestCheckpointConfig, CheckpointConfig, DataConfig, EvaluationConfig, LearningRateSchedule,
        LoggingConfig, ModelOverrides, OptimizerConfig, RuntimeConfig, SchedulerConfig,
        TokenizerConfig,
    },
    Trainer, TrainingConfig,
};

fn build_tokenizer(artifacts_dir: &PathBuf, corpus_path: &PathBuf) {
    let tokenizer_cfg = TokenizerTrainConfig {
        model: TokenizerModelCfg {
            vocab_size: 128,
            min_frequency: 1,
            dropout: None,
            special_tokens: vec!["<pad>".into(), "<bos>".into(), "<eos>".into()],
            byte_fallback_on_decode: true,
        },
        pretokenizer: TokenizerByteLevelCfg {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
        postprocessor: Some(TokenizerPostCfg {
            add_bos: true,
            add_eos: true,
            pair_template: false,
        }),
        training: Some(TokenizerTrainingCfg {
            inputs: vec![corpus_path.clone()],
            seed: 17,
            shuffle: false,
            max_lines: Some(128),
            num_threads: Some(1),
        }),
        artifacts: TokenizerArtifactsCfg {
            dir: artifacts_dir.clone(),
            tokenizer_json: Some(artifacts_dir.join("tokenizer.json")),
            vocab_json: Some(artifacts_dir.join("vocab.json")),
            merges_txt: Some(artifacts_dir.join("merges.txt")),
            manifest: Some(artifacts_dir.join("manifest.json")),
        },
    };

    train_bbpe(&tokenizer_cfg).expect("tokenizer training");
}

#[test]
fn smoke_training_checkpoint_resume() {
    let tmp = tempdir().expect("tempdir");
    let base = tmp.path();

    let train_path = base.join("train.txt");
    let valid_path = base.join("valid.txt");
    let corpus_text = (0..64)
        .map(|i| {
            format!(
                "sequence number {} quick brown fox jumps over the lazy dog",
                i
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&train_path, &corpus_text).unwrap();
    fs::write(&valid_path, &corpus_text).unwrap();

    let tokenizer_dir = base.join("tokenizer");
    fs::create_dir_all(&tokenizer_dir).unwrap();
    build_tokenizer(&tokenizer_dir, &train_path);
    let tokenizer_json = tokenizer_dir.join("tokenizer.json");

    let special_tokens_path = base.join("special_tokens.txt");
    fs::write(&special_tokens_path, "<pad>\n<bos>\n<eos>\n").unwrap();

    let checkpoint_dir = base.join("checkpoints");
    let best_dir = base.join("best");

    let training_config = TrainingConfig {
        model: ModelOverrides {
            hidden_size: Some(32),
            intermediate_size: Some(64),
            num_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(4),
            max_position_embeddings: Some(32),
            vocab_size: None,
            attn_dropout: None,
            residual_dropout: None,
            rope_mode: None,
            rope_theta: None,
        },
        tokenizer: TokenizerConfig {
            tokenizer_json: Some(tokenizer_json.clone()),
            vocab: None,
            merges: None,
            special_tokens: Some(special_tokens_path.clone()),
        },
        data: DataConfig {
            train_shards: vec![train_path.clone()],
            validation_shards: vec![valid_path.clone()],
            batch_size: 2,
            gradient_accumulation_steps: 1,
            num_workers: None,
            shuffle_buffer_size: Some(8),
            cache_dir: None,
        },
        optimizer: OptimizerConfig {
            learning_rate: 5e-2,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            algorithm: training::config::OptimizerType::AdamW,
        },
        scheduler: SchedulerConfig {
            strategy: LearningRateSchedule::Constant,
            warmup_steps: None,
            total_steps: Some(6),
            total_epochs: None,
            min_lr: None,
            max_lr: None,
        },
        runtime: RuntimeConfig {
            seed: 42,
            precision: training::config::Precision::Fp32,
            log_every_n_steps: 1,
            checkpoint: Some(CheckpointConfig {
                directory: checkpoint_dir.clone(),
                every_n_steps: Some(3),
                every_n_epochs: None,
                max_keep: Some(3),
            }),
            evaluation: EvaluationConfig {
                every_n_steps: Some(3),
                every_n_epochs: None,
                max_batches: Some(1),
                best: Some(BestCheckpointConfig {
                    directory: best_dir.clone(),
                    max_keep: Some(1),
                }),
            },
            logging: LoggingConfig {
                enable_stdout: false,
                tensorboard: None,
                tensorboard_flush_every_n: 1,
            },
        },
    };

    let config_path = base.join("config.toml");
    fs::write(&config_path, toml::to_string(&training_config).unwrap()).unwrap();

    let mut trainer = Trainer::new(TrainingConfig::load(&config_path).unwrap()).unwrap();
    let initial = trainer.evaluate(Some(1)).unwrap();
    trainer.train().unwrap();
    let after = trainer.evaluate(Some(1)).unwrap();

    assert!(after.perplexity <= initial.perplexity * 1.20);

    let checkpoint_entries = fs::read_dir(&checkpoint_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .count();
    assert!(checkpoint_entries > 0, "no checkpoints produced");

    let best_entries = fs::read_dir(&best_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .count();
    assert!(best_entries > 0, "no best checkpoint produced");

    let mut resumed = Trainer::new(TrainingConfig::load(&config_path).unwrap()).unwrap();
    let descriptor = resumed
        .resume_from_latest()
        .unwrap()
        .expect("resume checkpoint");
    assert!(descriptor.manifest.progress.optimizer_step > 0);
}
