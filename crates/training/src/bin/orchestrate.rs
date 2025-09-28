use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use tokenizer::config::{
    ArtifactsCfg as TokenizerArtifactsCfg, ByteLevelCfg as TokenizerByteLevelCfg,
    Config as TokenizerTrainConfig, ModelCfg as TokenizerModelCfg, PostCfg as TokenizerPostCfg,
    TrainingCfg as TokenizerTrainingCfg,
};
use tokenizer::train_bbpe;
use training::config::{
    BestCheckpointConfig, CheckpointConfig, DataConfig, EvaluationConfig, LearningRateSchedule,
    LoggingConfig, ModelOverrides, OptimizerConfig, Precision, RuntimeConfig, SchedulerConfig,
    TokenizerConfig, TrainingConfig,
};
use training::{Trainer, TrainingError};

#[derive(Debug, Parser)]
#[command(
    name = "orchestrate",
    about = "End-to-end tokenizer + transformer training pipeline"
)]
struct Args {
    #[arg(long, help = "Path to the text corpus used for training")]
    input: PathBuf,

    #[arg(
        long,
        help = "Optional validation corpus; defaults to the training corpus"
    )]
    validation: Option<PathBuf>,

    #[arg(
        long,
        help = "Directory where run artifacts (tokenizer, checkpoints, logs) are stored"
    )]
    run_dir: Option<PathBuf>,

    #[arg(long, help = "Existing tokenizer.json to reuse instead of retraining")]
    tokenizer_json: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = 32_000,
        help = "Tokenizer vocabulary size when training a new tokenizer"
    )]
    vocab_size: usize,

    #[arg(long, help = "Limit tokenizer training to the first N lines")]
    tokenizer_max_lines: Option<usize>,

    #[arg(long, default_value_t = 512, help = "Model hidden size")]
    hidden_size: usize,

    #[arg(
        long,
        default_value_t = 2_048,
        help = "Model feed-forward intermediate size"
    )]
    intermediate_size: usize,

    #[arg(long, default_value_t = 8, help = "Number of transformer layers")]
    layers: usize,

    #[arg(long, default_value_t = 8, help = "Number of attention heads")]
    heads: usize,

    #[arg(long, default_value_t = 1_024, help = "Maximum sequence length")]
    seq_len: usize,

    #[arg(
        long,
        default_value_t = 16,
        help = "Global batch size (after gradient accumulation)"
    )]
    batch_size: usize,

    #[arg(long, default_value_t = 1, help = "Gradient accumulation steps")]
    grad_accum: usize,

    #[arg(
        long,
        help = "Maximum optimizer steps; training stops after this many steps"
    )]
    steps: Option<usize>,

    #[arg(long, default_value_t = 3e-4_f32, help = "Optimizer learning rate")]
    learning_rate: f32,

    #[arg(long, default_value_t = 0.1_f32, help = "Weight decay coefficient")]
    weight_decay: f32,

    #[arg(long, help = "Number of warmup steps for schedulers that support it")]
    warmup_steps: Option<usize>,

    #[arg(long, value_enum, default_value_t = ScheduleFlag::CosineWithWarmup,
        help = "Learning rate schedule")]
    schedule: ScheduleFlag,

    #[arg(
        long,
        default_value_t = 200,
        help = "Checkpoint every N optimizer steps"
    )]
    checkpoint_every: usize,

    #[arg(long, default_value_t = 5, help = "Keep at most N checkpoints")]
    max_checkpoints: usize,

    #[arg(
        long,
        default_value_t = 200,
        help = "Run evaluation every N optimizer steps"
    )]
    evaluate_every: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Number of evaluation batches per run"
    )]
    eval_batches: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Log training metrics every N optimizer steps"
    )]
    log_every: usize,

    #[arg(
        long,
        default_value_t = 16_384,
        help = "Shuffle buffer size for streaming loader"
    )]
    shuffle_buffer: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Resume from the latest checkpoint if present"
    )]
    resume: bool,

    #[arg(long, value_enum, default_value_t = PrecisionFlag::Bf16, help = "Computation precision")]
    precision: PrecisionFlag,

    #[arg(long, default_value_t = 17, help = "Seed for tokenizer training")]
    tokenizer_seed: u64,

    #[arg(
        long,
        default_value_t = 42,
        help = "Seed for model initialization and dataloader"
    )]
    seed: u64,

    #[arg(
        long,
        help = "Optional TensorBoard directory (relative to run dir when omitted)"
    )]
    tensorboard: Option<PathBuf>,

    #[arg(long, help = "Skip tokenizer training even if artifacts are missing")]
    skip_tokenizer: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = Profile::Full,
        help = "Preset that adjusts hyperparameters (use 'local' for quick CPU sanity runs)"
    )]
    profile: Profile,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum PrecisionFlag {
    Fp32,
    Fp16,
    Bf16,
    Mixed,
}

impl From<PrecisionFlag> for Precision {
    fn from(value: PrecisionFlag) -> Self {
        match value {
            PrecisionFlag::Fp32 => Precision::Fp32,
            PrecisionFlag::Fp16 => Precision::Fp16,
            PrecisionFlag::Bf16 => Precision::Bf16,
            PrecisionFlag::Mixed => Precision::Mixed,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ScheduleFlag {
    Constant,
    Cosine,
    LinearWarmup,
    CosineWithWarmup,
    Polynomial,
}

impl From<ScheduleFlag> for LearningRateSchedule {
    fn from(value: ScheduleFlag) -> Self {
        match value {
            ScheduleFlag::Constant => LearningRateSchedule::Constant,
            ScheduleFlag::Cosine => LearningRateSchedule::Cosine,
            ScheduleFlag::LinearWarmup => LearningRateSchedule::LinearWarmup,
            ScheduleFlag::CosineWithWarmup => LearningRateSchedule::CosineWithWarmup,
            ScheduleFlag::Polynomial => LearningRateSchedule::Polynomial,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Profile {
    Full,
    Local,
}

#[derive(Debug)]
enum PipelineError {
    Io(std::io::Error),
    Tokenizer(tokenizer::errors::Error),
    Training(TrainingError),
    Toml(toml::ser::Error),
    CtrlC(ctrlc::Error),
    Invalid(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Io(err) => write!(f, "io error: {err}"),
            PipelineError::Tokenizer(err) => write!(f, "tokenizer error: {err}"),
            PipelineError::Training(err) => write!(f, "training error: {err}"),
            PipelineError::Toml(err) => write!(f, "failed to serialize config: {err}"),
            PipelineError::CtrlC(err) => write!(f, "failed to install signal handler: {err}"),
            PipelineError::Invalid(msg) => write!(f, "invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(value: std::io::Error) -> Self {
        PipelineError::Io(value)
    }
}

impl From<tokenizer::errors::Error> for PipelineError {
    fn from(value: tokenizer::errors::Error) -> Self {
        PipelineError::Tokenizer(value)
    }
}

impl From<TrainingError> for PipelineError {
    fn from(value: TrainingError) -> Self {
        PipelineError::Training(value)
    }
}

impl From<toml::ser::Error> for PipelineError {
    fn from(value: toml::ser::Error) -> Self {
        PipelineError::Toml(value)
    }
}

impl From<ctrlc::Error> for PipelineError {
    fn from(value: ctrlc::Error) -> Self {
        PipelineError::CtrlC(value)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("pipeline failed: {err}");
        std::process::exit(1);
    }
}

type Result<T> = std::result::Result<T, PipelineError>;

fn run() -> Result<()> {
    let mut args = Args::parse();
    if args.profile == Profile::Local {
        println!("profile: applying 'local' preset for quick smoke testing");
        args.apply_local_preset();
    }
    let input = ensure_exists(&args.input)?;
    let validation = args
        .validation
        .as_ref()
        .map(|path| ensure_exists(path))
        .transpose()?;
    let validation = validation.unwrap_or_else(|| input.clone());

    if args.hidden_size % args.heads != 0 {
        return Err(PipelineError::Invalid(
            "hidden size must be divisible by number of heads".into(),
        ));
    }

    if args.grad_accum == 0 {
        return Err(PipelineError::Invalid(
            "gradient accumulation steps must be greater than zero".into(),
        ));
    }

    if args.batch_size == 0 {
        return Err(PipelineError::Invalid(
            "batch size must be greater than zero".into(),
        ));
    }

    let run_dir = args
        .run_dir
        .clone()
        .unwrap_or_else(|| default_run_dir("run"));
    fs::create_dir_all(&run_dir)?;

    let tokenizer_dir = run_dir.join("tokenizer");
    fs::create_dir_all(&tokenizer_dir)?;

    let checkpoints_dir = run_dir.join("checkpoints");
    fs::create_dir_all(&checkpoints_dir)?;

    let best_dir = run_dir.join("best");
    fs::create_dir_all(&best_dir)?;

    let tensorboard_dir = args
        .tensorboard
        .clone()
        .unwrap_or_else(|| run_dir.join("tensorboard"));
    fs::create_dir_all(&tensorboard_dir)?;

    let special_tokens_path = run_dir.join("special_tokens.txt");
    write_special_tokens(&special_tokens_path)?;

    let tokenizer_json_path = resolve_tokenizer(&args, &tokenizer_dir, &input)?;

    let training_config = build_training_config(
        &args,
        &tokenizer_json_path,
        &special_tokens_path,
        &input,
        &validation,
        &checkpoints_dir,
        &best_dir,
        &tensorboard_dir,
    )?;

    let config_path = run_dir.join("training.toml");
    let config_toml = toml::to_string_pretty(&training_config)?;
    fs::write(&config_path, config_toml)?;
    println!("saved training config to {}", config_path.display());

    let mut trainer = Trainer::new(training_config.clone())?;

    if args.resume {
        if let Some(descriptor) = trainer.resume_from_latest()? {
            println!(
                "resumed from checkpoint {} (step {})",
                descriptor.directory.display(),
                descriptor.manifest.progress.optimizer_step
            );
        }
    }

    if args.eval_batches > 0 {
        if let Ok(initial) = trainer.evaluate(Some(args.eval_batches)) {
            println!("initial evaluation perplexity={:.3}", initial.perplexity);
        }
    }

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let handler_flag = shutdown_flag.clone();
    ctrlc::set_handler(move || {
        handler_flag.store(true, Ordering::Relaxed);
    })?;

    trainer.train_with_shutdown(|| shutdown_flag.load(Ordering::Relaxed))?;

    if args.eval_batches > 0 {
        if let Ok(final_eval) = trainer.evaluate(Some(args.eval_batches)) {
            println!("final evaluation perplexity={:.3}", final_eval.perplexity);
        }
    }

    println!(
        "training complete; artifacts stored under {}",
        run_dir.display()
    );
    Ok(())
}

fn ensure_exists(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        Ok(path.to_path_buf())
    } else {
        Err(PipelineError::Invalid(format!(
            "path {} does not exist",
            path.display()
        )))
    }
}

fn resolve_tokenizer(args: &Args, tokenizer_dir: &Path, corpus: &Path) -> Result<PathBuf> {
    if let Some(existing) = args.tokenizer_json.as_ref() {
        println!("using existing tokenizer artifact {}", existing.display());
        return Ok(existing.clone());
    }

    let tokenizer_json = tokenizer_dir.join("tokenizer.json");
    if tokenizer_json.exists() {
        if args.skip_tokenizer {
            println!(
                "reusing tokenizer artifact {}; skipping training",
                tokenizer_json.display()
            );
            return Ok(tokenizer_json);
        }
    }

    if args.skip_tokenizer {
        return Err(PipelineError::Invalid(
            "skip-tokenizer was set but no tokenizer artifact was supplied".into(),
        ));
    }

    println!(
        "training tokenizer (vocab={} lines={:?})",
        args.vocab_size, args.tokenizer_max_lines
    );

    let tokenizer_cfg = TokenizerTrainConfig {
        model: TokenizerModelCfg {
            vocab_size: args.vocab_size,
            min_frequency: 2,
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
            inputs: vec![corpus.to_path_buf()],
            seed: args.tokenizer_seed,
            shuffle: true,
            max_lines: args.tokenizer_max_lines,
            num_threads: None,
        }),
        artifacts: TokenizerArtifactsCfg {
            dir: tokenizer_dir.to_path_buf(),
            tokenizer_json: Some(tokenizer_json.clone()),
            vocab_json: Some(tokenizer_dir.join("vocab.json")),
            merges_txt: Some(tokenizer_dir.join("merges.txt")),
            manifest: Some(tokenizer_dir.join("manifest.json")),
        },
    };

    let tokenizer = train_bbpe(&tokenizer_cfg)?;
    tokenizer
        .save(tokenizer_json.clone(), false)
        .map_err(tokenizer::errors::Error::from)?;
    println!(
        "tokenizer training complete; artifacts in {}",
        tokenizer_dir.display()
    );

    Ok(tokenizer_json)
}

fn build_training_config(
    args: &Args,
    tokenizer_json: &Path,
    special_tokens: &Path,
    train_corpus: &Path,
    validation_corpus: &Path,
    checkpoint_dir: &Path,
    best_dir: &Path,
    tensorboard_dir: &Path,
) -> Result<TrainingConfig> {
    let optimizer = OptimizerConfig {
        learning_rate: args.learning_rate,
        weight_decay: args.weight_decay,
        ..OptimizerConfig::default()
    };

    let scheduler = SchedulerConfig {
        strategy: args.schedule.into(),
        warmup_steps: args.warmup_steps,
        total_steps: args.steps,
        total_epochs: None,
        min_lr: None,
        max_lr: None,
    };

    let runtime = RuntimeConfig {
        seed: args.seed,
        precision: args.precision.into(),
        log_every_n_steps: args.log_every,
        checkpoint: Some(CheckpointConfig {
            directory: checkpoint_dir.to_path_buf(),
            every_n_steps: Some(args.checkpoint_every),
            every_n_epochs: None,
            max_keep: Some(args.max_checkpoints),
        }),
        evaluation: EvaluationConfig {
            every_n_steps: Some(args.evaluate_every),
            every_n_epochs: None,
            max_batches: Some(args.eval_batches.max(1)),
            best: Some(BestCheckpointConfig {
                directory: best_dir.to_path_buf(),
                max_keep: Some(1),
            }),
        },
        logging: LoggingConfig {
            enable_stdout: true,
            tensorboard: Some(tensorboard_dir.to_path_buf()),
            tensorboard_flush_every_n: 10,
        },
    };

    let model = ModelOverrides {
        hidden_size: Some(args.hidden_size),
        intermediate_size: Some(args.intermediate_size),
        num_attention_heads: Some(args.heads),
        num_key_value_heads: Some(args.heads),
        num_layers: Some(args.layers),
        max_position_embeddings: Some(args.seq_len),
        ..ModelOverrides::default()
    };

    let tokenizer_cfg = TokenizerConfig {
        tokenizer_json: Some(tokenizer_json.to_path_buf()),
        vocab: None,
        merges: None,
        special_tokens: Some(special_tokens.to_path_buf()),
    };

    let data = DataConfig {
        train_shards: vec![train_corpus.to_path_buf()],
        validation_shards: vec![validation_corpus.to_path_buf()],
        batch_size: args.batch_size,
        gradient_accumulation_steps: args.grad_accum.max(1),
        num_workers: None,
        shuffle_buffer_size: Some(args.shuffle_buffer.max(1)),
        cache_dir: None,
    };

    let config = TrainingConfig {
        model,
        tokenizer: tokenizer_cfg,
        data,
        optimizer,
        scheduler,
        runtime,
    };

    config.validate()?;

    Ok(config)
}

fn write_special_tokens(path: &Path) -> Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(b"<pad>\n<bos>\n<eos>\n")?;
    Ok(())
}

fn default_run_dir(prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from(format!("runs/{prefix}-{timestamp}"))
}
