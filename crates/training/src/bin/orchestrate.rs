use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use pretraining_data::sharding::shard_text_by_lines;
use serde::Serialize;
use sha2::{Digest, Sha256};
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

const CLOUD_TRAIN_ROOT: &str = "/workspace/datasets/train";
const CLOUD_VAL_ROOT: &str = "/workspace/datasets/val";
const CLOUD_TOKENIZER_ROOT: &str = "/workspace/artifacts/tokenizer";
const CLOUD_RUNS_ROOT: &str = "/workspace/runs";
const CLOUD_CACHE_ROOT: &str = "/workspace/cache";
const CLOUD_HF_CACHE_ROOT: &str = "/workspace/hf_cache";

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

    #[arg(
        long,
        value_enum,
        default_value_t = PipelineMode::Local,
        help = "Pipeline target; use 'cloud' to enable sharding and RunPod paths"
    )]
    mode: PipelineMode,

    #[arg(
        long,
        help = "Experiment slug for cloud layout (defaults to cloud-<timestamp>)"
    )]
    experiment: Option<String>,

    #[arg(
        long,
        default_value_t = 1_000_000,
        help = "Lines per training shard when sharding is enabled"
    )]
    train_shard_lines: usize,

    #[arg(long, help = "Lines per validation shard; defaults to train shard cap")]
    validation_shard_lines: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Reuse existing shards instead of regenerating them (cloud mode)"
    )]
    reuse_shards: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Regenerate shards even if dataset dirs already exist (cloud mode)"
    )]
    force_reshard: bool,

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

    #[arg(long, help = "Number of streaming dataloader worker threads")]
    loader_workers: Option<usize>,

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
        default_value_t = false,
        help = "Force retraining the tokenizer even if artifacts exist"
    )]
    force_retrain_tokenizer: bool,

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
enum PipelineMode {
    Local,
    Cloud,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Profile {
    Full,
    Local,
}

impl Args {
    fn apply_local_preset(&mut self) {
        self.vocab_size = 4_096;
        self.tokenizer_max_lines = Some(self.tokenizer_max_lines.unwrap_or(5_000).min(5_000));
        self.hidden_size = 128;
        self.intermediate_size = 512;
        self.layers = 2;
        self.heads = 2;
        self.seq_len = 256;
        self.batch_size = self.batch_size.max(1).min(4);
        self.grad_accum = 1;
        if self.steps.map_or(true, |steps| steps > 200) {
            self.steps = Some(200);
        }
        self.learning_rate = 5e-4;
        self.weight_decay = 0.0;
        self.warmup_steps = Some(self.warmup_steps.unwrap_or(20).min(20));
        self.schedule = ScheduleFlag::Constant;
        self.checkpoint_every = self.checkpoint_every.max(10).min(50);
        self.max_checkpoints = self.max_checkpoints.max(1).min(2);
        self.evaluate_every = self.evaluate_every.max(10).min(50);
        self.eval_batches = 1;
        self.log_every = 1;
        self.shuffle_buffer = self.shuffle_buffer.max(256).min(2_048);
        self.precision = PrecisionFlag::Fp32;
    }
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

impl From<serde_yaml::Error> for PipelineError {
    fn from(value: serde_yaml::Error) -> Self {
        PipelineError::Invalid(format!("failed to serialize config to yaml: {value}"))
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

    if args.reuse_shards && args.force_reshard {
        return Err(PipelineError::Invalid(
            "--reuse-shards and --force-reshard cannot be combined".into(),
        ));
    }

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

    if args.mode == PipelineMode::Cloud {
        if args.train_shard_lines == 0 {
            return Err(PipelineError::Invalid(
                "train-shard-lines must be greater than zero".into(),
            ));
        }
        if matches!(args.validation_shard_lines, Some(0)) {
            return Err(PipelineError::Invalid(
                "validation-shard-lines must be greater than zero when provided".into(),
            ));
        }
    }

    let input = ensure_exists(&args.input)?;
    let validation = args
        .validation
        .as_ref()
        .map(|path| ensure_exists(path))
        .transpose()?;
    let validation = validation.unwrap_or_else(|| input.clone());

    match args.mode {
        PipelineMode::Local => run_local(args, input, validation),
        PipelineMode::Cloud => run_cloud(args, input, validation),
    }
}

fn run_local(args: Args, train_source: PathBuf, validation_source: PathBuf) -> Result<()> {
    let run_dir = args
        .run_dir
        .clone()
        .unwrap_or_else(|| default_run_dir("run"));
    let artifacts = prepare_artifacts(&args, run_dir)?;

    let data = PipelineData {
        train_shards: vec![train_source.clone()],
        validation_shards: vec![validation_source],
        tokenizer_inputs: vec![train_source],
        cache_dir: None,
        shared_tokenizer_dir: None,
    };

    execute_pipeline(args, artifacts, data)
}

fn run_cloud(args: Args, train_source: PathBuf, validation_source: PathBuf) -> Result<()> {
    let experiment = args
        .experiment
        .clone()
        .unwrap_or_else(default_cloud_experiment);

    let run_dir = args
        .run_dir
        .clone()
        .unwrap_or_else(|| Path::new(CLOUD_RUNS_ROOT).join(&experiment));
    configure_hf_cache()?;
    let artifacts = prepare_artifacts(&args, run_dir)?;

    let datasets = prepare_cloud_datasets(&args, &experiment, &train_source, &validation_source)?;

    let cache_dir = Path::new(CLOUD_CACHE_ROOT).join(&experiment);
    fs::create_dir_all(&cache_dir)?;
    let shared_tokenizer_dir = Path::new(CLOUD_TOKENIZER_ROOT).join(&experiment);

    let data = PipelineData {
        train_shards: datasets.train_shards,
        validation_shards: datasets.validation_shards,
        tokenizer_inputs: datasets.tokenizer_inputs,
        cache_dir: Some(cache_dir),
        shared_tokenizer_dir: Some(shared_tokenizer_dir),
    };

    execute_pipeline(args, artifacts, data)
}

struct PipelineArtifacts {
    run_dir: PathBuf,
    tokenizer_dir: PathBuf,
    checkpoints_dir: PathBuf,
    best_dir: PathBuf,
    tensorboard_dir: PathBuf,
    special_tokens_path: PathBuf,
}

struct PipelineData {
    train_shards: Vec<PathBuf>,
    validation_shards: Vec<PathBuf>,
    tokenizer_inputs: Vec<PathBuf>,
    cache_dir: Option<PathBuf>,
    shared_tokenizer_dir: Option<PathBuf>,
}

struct TokenizerOutcome {
    path: PathBuf,
    trained: bool,
}

fn configure_hf_cache() -> Result<()> {
    let cache_root = Path::new(CLOUD_HF_CACHE_ROOT);
    fs::create_dir_all(cache_root)?;
    let datasets_cache = cache_root.join("datasets");
    let hub_cache = cache_root.join("hub");
    fs::create_dir_all(&datasets_cache)?;
    fs::create_dir_all(&hub_cache)?;

    env::set_var("HF_HOME", cache_root);
    env::set_var("HF_DATASETS_CACHE", &datasets_cache);
    env::set_var("HF_HUB_CACHE", &hub_cache);

    if env::var("HUGGINGFACEHUB_API_TOKEN").is_err() {
        if let Ok(token) = env::var("HF_TOKEN") {
            env::set_var("HUGGINGFACEHUB_API_TOKEN", token);
        }
    }

    Ok(())
}

fn prepare_artifacts(args: &Args, run_dir: PathBuf) -> Result<PipelineArtifacts> {
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

    Ok(PipelineArtifacts {
        run_dir,
        tokenizer_dir,
        checkpoints_dir,
        best_dir,
        tensorboard_dir,
        special_tokens_path,
    })
}

fn execute_pipeline(args: Args, artifacts: PipelineArtifacts, data: PipelineData) -> Result<()> {
    write_special_tokens(&artifacts.special_tokens_path)?;

    let PipelineData {
        train_shards,
        validation_shards,
        tokenizer_inputs,
        cache_dir,
        shared_tokenizer_dir,
    } = data;

    verify_shards("train", &train_shards)?;
    verify_shards("validation", &validation_shards)?;

    let tokenizer_outcome = resolve_tokenizer(
        &args,
        &artifacts.tokenizer_dir,
        &tokenizer_inputs,
        shared_tokenizer_dir.as_deref(),
    )?;
    let tokenizer_json_path = tokenizer_outcome.path.clone();
    println!(
        "[orchestrator] tokenizer artifacts ready at {}",
        tokenizer_json_path.display()
    );

    write_tokenizer_manifest(
        &args,
        &artifacts.tokenizer_dir,
        &tokenizer_inputs,
        tokenizer_outcome.trained,
    )?;

    let config_tokenizer_path = shared_tokenizer_dir
        .as_ref()
        .map(|dir| dir.join("tokenizer.json"))
        .unwrap_or_else(|| tokenizer_json_path.clone());

    let training_config = build_training_config(
        &args,
        &config_tokenizer_path,
        &artifacts.special_tokens_path,
        &train_shards,
        &validation_shards,
        &artifacts.checkpoints_dir,
        &artifacts.best_dir,
        &artifacts.tensorboard_dir,
        cache_dir.as_deref(),
    )?;

    let config_toml_path = artifacts.run_dir.join("training.toml");
    let config_toml = toml::to_string_pretty(&training_config)?;
    fs::write(&config_toml_path, config_toml)?;

    let config_yaml_path = artifacts.run_dir.join("training.yaml");
    let config_yaml = serde_yaml::to_string(&training_config)?;
    fs::write(&config_yaml_path, config_yaml)?;

    println!(
        "[orchestrator] saved training config to {} and {}",
        config_toml_path.display(),
        config_yaml_path.display()
    );
    println!(
        "[orchestrator] training config assembled (model.layers={} hidden={} heads={})",
        training_config.model.num_layers.unwrap_or_default(),
        training_config.model.hidden_size.unwrap_or_default(),
        training_config
            .model
            .num_attention_heads
            .unwrap_or_default()
    );
    println!(
        "[orchestrator] reproducibility -> tokenizer_vocab={} tokenizer_seed={} model_seed={} schedule={:?} precision={:?}",
        args.vocab_size,
        args.tokenizer_seed,
        args.seed,
        args.schedule,
        args.precision
    );
    println!(
        "[orchestrator] dataloader -> batch_size={} grad_accum={} shuffle_buffer={} num_workers={:?}",
        args.batch_size,
        args.grad_accum,
        args.shuffle_buffer,
        args.loader_workers
    );

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
        artifacts.run_dir.display()
    );
    Ok(())
}

struct CloudDatasets {
    train_shards: Vec<PathBuf>,
    validation_shards: Vec<PathBuf>,
    tokenizer_inputs: Vec<PathBuf>,
}

fn prepare_cloud_datasets(
    args: &Args,
    experiment: &str,
    train_source: &Path,
    validation_source: &Path,
) -> Result<CloudDatasets> {
    let train_dir = Path::new(CLOUD_TRAIN_ROOT).join(experiment);
    let train_shards = materialize_shards(
        train_source,
        &train_dir,
        "train",
        args.train_shard_lines,
        args.reuse_shards,
        args.force_reshard,
    )?;

    let val_dir = Path::new(CLOUD_VAL_ROOT).join(experiment);
    let validation_shards = if train_source == validation_source {
        mirror_shards(
            &train_shards,
            &val_dir,
            "val",
            args.reuse_shards,
            args.force_reshard,
        )?
    } else {
        materialize_shards(
            validation_source,
            &val_dir,
            "val",
            args.validation_shard_lines
                .unwrap_or(args.train_shard_lines),
            args.reuse_shards,
            args.force_reshard,
        )?
    };

    Ok(CloudDatasets {
        tokenizer_inputs: train_shards.clone(),
        train_shards,
        validation_shards,
    })
}

fn materialize_shards(
    source: &Path,
    destination_dir: &Path,
    prefix: &str,
    max_lines: usize,
    reuse_shards: bool,
    force_reshard: bool,
) -> Result<Vec<PathBuf>> {
    if reuse_shards {
        return collect_existing_shards(destination_dir);
    }

    if destination_dir.exists() {
        if force_reshard {
            fs::remove_dir_all(destination_dir)?;
        } else {
            return Err(PipelineError::Invalid(format!(
                "dataset directory {} already exists; pass --reuse-shards or --force-reshard",
                destination_dir.display()
            )));
        }
    }

    if source.is_dir() {
        replicate_directory_shards(source, destination_dir, prefix)
    } else if source.is_file() {
        shard_text_by_lines(source, destination_dir, prefix, max_lines).map_err(PipelineError::from)
    } else {
        Err(PipelineError::Invalid(format!(
            "path {} does not exist",
            source.display()
        )))
    }
}

fn replicate_directory_shards(
    source_dir: &Path,
    destination_dir: &Path,
    prefix: &str,
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(destination_dir)?;
    let mut files = Vec::new();
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(PipelineError::Invalid(format!(
            "no shards found under {}",
            source_dir.display()
        )));
    }

    let mut shards = Vec::new();
    for (idx, src) in files.iter().enumerate() {
        let dest = destination_dir.join(format!("{prefix}-{idx:05}.txt"));
        link_or_copy(src, &dest)?;
        shards.push(dest);
    }
    Ok(shards)
}

fn mirror_shards(
    source_shards: &[PathBuf],
    destination_dir: &Path,
    prefix: &str,
    reuse_shards: bool,
    force_reshard: bool,
) -> Result<Vec<PathBuf>> {
    if reuse_shards {
        return collect_existing_shards(destination_dir);
    }

    if destination_dir.exists() {
        if force_reshard {
            fs::remove_dir_all(destination_dir)?;
        } else {
            return Err(PipelineError::Invalid(format!(
                "dataset directory {} already exists; pass --reuse-shards or --force-reshard",
                destination_dir.display()
            )));
        }
    }

    fs::create_dir_all(destination_dir)?;
    let mut shards = Vec::new();
    for (idx, shard) in source_shards.iter().enumerate() {
        let dest = destination_dir.join(format!("{prefix}-{idx:05}.txt"));
        link_or_copy(shard, &dest)?;
        shards.push(dest);
    }
    Ok(shards)
}

fn collect_existing_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(PipelineError::Invalid(format!(
            "expected shard directory {} to exist",
            dir.display()
        )));
    }

    let mut shards = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            shards.push(entry.path());
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(PipelineError::Invalid(format!(
            "no shards found under {}",
            dir.display()
        )));
    }
    Ok(shards)
}

#[cfg(unix)]
fn link_or_copy(source: &Path, dest: &Path) -> io::Result<()> {
    use std::os::unix::fs::symlink;

    if dest.exists() {
        fs::remove_file(dest)?;
    }

    match symlink(source, dest) {
        Ok(_) => Ok(()),
        Err(_) => {
            let _ = fs::copy(source, dest)?;
            Ok(())
        }
    }
}

#[cfg(not(unix))]
fn link_or_copy(source: &Path, dest: &Path) -> io::Result<()> {
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    let _ = fs::copy(source, dest)?;
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

fn resolve_tokenizer(
    args: &Args,
    tokenizer_dir: &Path,
    corpora: &[PathBuf],
    shared_dir: Option<&Path>,
) -> Result<TokenizerOutcome> {
    if corpora.is_empty() {
        return Err(PipelineError::Invalid(
            "no tokenizer corpora supplied".into(),
        ));
    }

    let tokenizer_json = tokenizer_dir.join("tokenizer.json");

    if let Some(shared) = shared_dir {
        fs::create_dir_all(shared)?;
    }

    if let Some(existing) = args.tokenizer_json.as_ref() {
        let existing = ensure_exists(existing)?;
        copy_tokenizer_file(&existing, &tokenizer_json)?;
        if let Some(parent) = existing.parent() {
            copy_tokenizer_bundle(parent, tokenizer_dir)?;
        }
        sync_shared_tokenizer_dir(tokenizer_dir, shared_dir)?;
        println!(
            "[orchestrator] using provided tokenizer artifact {}; skipping training",
            tokenizer_json.display()
        );
        return Ok(TokenizerOutcome {
            path: tokenizer_json,
            trained: false,
        });
    }

    if !args.force_retrain_tokenizer {
        if tokenizer_json.exists() {
            println!(
                "[orchestrator] tokenizer artifacts already present at {}; pass --force-retrain-tokenizer to rebuild",
                tokenizer_json.display()
            );
            sync_shared_tokenizer_dir(tokenizer_dir, shared_dir)?;
            return Ok(TokenizerOutcome {
                path: tokenizer_json,
                trained: false,
            });
        }

        if let Some(shared) = shared_dir {
            let shared_json = shared.join("tokenizer.json");
            if shared_json.exists() {
                println!(
                    "[orchestrator] copying tokenizer bundle from shared dir {}; pass --force-retrain-tokenizer to rebuild",
                    shared.display()
                );
                copy_tokenizer_bundle(shared, tokenizer_dir)?;
                return Ok(TokenizerOutcome {
                    path: tokenizer_json,
                    trained: false,
                });
            }
        }
    }

    if args.skip_tokenizer {
        if tokenizer_json.exists() {
            println!(
                "[orchestrator] skip-tokenizer set; reusing artifact {}",
                tokenizer_json.display()
            );
            sync_shared_tokenizer_dir(tokenizer_dir, shared_dir)?;
            return Ok(TokenizerOutcome {
                path: tokenizer_json,
                trained: false,
            });
        }
        if let Some(shared) = shared_dir {
            let shared_json = shared.join("tokenizer.json");
            if shared_json.exists() {
                println!(
                    "[orchestrator] skip-tokenizer set; copying artifact from shared dir {}",
                    shared.display()
                );
                copy_tokenizer_bundle(shared, tokenizer_dir)?;
                return Ok(TokenizerOutcome {
                    path: tokenizer_json,
                    trained: false,
                });
            }
        }
        return Err(PipelineError::Invalid(
            "skip-tokenizer was set but no tokenizer artifact was supplied".into(),
        ));
    }

    if args.force_retrain_tokenizer && tokenizer_json.exists() {
        println!(
            "[orchestrator] force-retrain-tokenizer set; retraining tokenizer at {}",
            tokenizer_json.display()
        );
    }

    println!(
        "[orchestrator] training tokenizer (vocab={} lines={:?} shards={})",
        args.vocab_size,
        args.tokenizer_max_lines,
        corpora.len()
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
            inputs: corpora.to_vec(),
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
        "[orchestrator] tokenizer training complete; artifacts in {}",
        tokenizer_dir.display()
    );

    sync_shared_tokenizer_dir(tokenizer_dir, shared_dir)?;

    Ok(TokenizerOutcome {
        path: tokenizer_json,
        trained: true,
    })
}

const TOKENIZER_BUNDLE_FILES: &[&str] = &[
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "manifest.json",
];

fn copy_tokenizer_file(source: &Path, dest: &Path) -> Result<()> {
    if source == dest {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source, dest)?;
    Ok(())
}

fn copy_tokenizer_bundle(source_dir: &Path, destination_dir: &Path) -> Result<()> {
    fs::create_dir_all(destination_dir)?;
    for name in TOKENIZER_BUNDLE_FILES {
        let source_path = source_dir.join(name);
        if source_path.exists() {
            let destination_path = destination_dir.join(name);
            copy_tokenizer_file(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn sync_shared_tokenizer_dir(local_dir: &Path, shared_dir: Option<&Path>) -> Result<()> {
    if let Some(dir) = shared_dir {
        copy_tokenizer_bundle(local_dir, dir)?;
    }
    Ok(())
}

#[derive(Serialize)]
struct TokenizerManifestEntry {
    path: String,
    size_bytes: Option<u64>,
    sha256: Option<String>,
}

#[derive(Serialize)]
struct TokenizerManifest {
    version: u32,
    generated_at_unix: u64,
    trained: bool,
    force_retrain: bool,
    vocab_size: usize,
    tokenizer_seed: u64,
    tokenizer_max_lines: Option<usize>,
    corpora: Vec<TokenizerManifestEntry>,
}

fn write_tokenizer_manifest(
    args: &Args,
    tokenizer_dir: &Path,
    corpora: &[PathBuf],
    trained: bool,
) -> Result<()> {
    let generated_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut entries = Vec::new();
    for path in corpora {
        let (size_bytes, sha256) = if path.is_file() {
            let meta = fs::metadata(path)?;
            let hash = sha256_file(path)?;
            (Some(meta.len()), Some(hash))
        } else {
            (None, None)
        };

        entries.push(TokenizerManifestEntry {
            path: path.display().to_string(),
            size_bytes,
            sha256,
        });
    }

    let manifest = TokenizerManifest {
        version: 1,
        generated_at_unix,
        trained,
        force_retrain: args.force_retrain_tokenizer,
        vocab_size: args.vocab_size,
        tokenizer_seed: args.tokenizer_seed,
        tokenizer_max_lines: args.tokenizer_max_lines,
        corpora: entries,
    };

    let manifest_path = tokenizer_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(manifest_path, json)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_shards(label: &str, shards: &[PathBuf]) -> Result<()> {
    if shards.is_empty() {
        return Err(PipelineError::Invalid(format!(
            "no {label} shards provided; run the sharder before launching training"
        )));
    }

    let mut total_bytes = 0u64;
    for path in shards {
        if !path.is_file() {
            return Err(PipelineError::Invalid(format!(
                "{label} shard {} is missing or not a file",
                path.display()
            )));
        }
        let meta = fs::metadata(path)?;
        if meta.len() == 0 {
            return Err(PipelineError::Invalid(format!(
                "{label} shard {} is empty; regenerate shards",
                path.display()
            )));
        }
        total_bytes = total_bytes.saturating_add(meta.len());
    }

    let mb = total_bytes as f64 / 1_048_576_f64;
    println!(
        "[orchestrator] {label} shards verified ({} files, {:.2} MiB)",
        shards.len(),
        mb
    );
    Ok(())
}

fn build_training_config(
    args: &Args,
    tokenizer_json: &Path,
    special_tokens: &Path,
    train_shards: &[PathBuf],
    validation_shards: &[PathBuf],
    checkpoint_dir: &Path,
    best_dir: &Path,
    tensorboard_dir: &Path,
    cache_dir: Option<&Path>,
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
        train_shards: train_shards.to_vec(),
        validation_shards: validation_shards.to_vec(),
        batch_size: args.batch_size,
        gradient_accumulation_steps: args.grad_accum.max(1),
        num_workers: args.loader_workers,
        shuffle_buffer_size: Some(args.shuffle_buffer.max(1)),
        cache_dir: cache_dir.map(PathBuf::from),
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

fn default_cloud_experiment() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("cloud-{timestamp}")
}
