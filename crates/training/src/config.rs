use attention::core::RopeMode;
use candle_core::{DType, Device};
use layers::norm::NormKind;
use model::{build_model_config_with_overrides, ModelConfig, ModelConfigOverrides};
use serde::Deserialize;
use std::{
    fmt, fs,
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    #[serde(default)]
    pub model: ModelOverrides,
    pub tokenizer: TokenizerConfig,
    pub data: DataConfig,
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

#[derive(Debug)]
pub struct ResolvedModelHyperparameters {
    pub model: ModelConfig,
    pub max_position_embeddings: usize,
    pub precision: Precision,
}

impl TrainingConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, TrainingError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)?;
        let mut config: TrainingConfig = match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => serde_json::from_str(&contents)?,
            Some("toml") | Some("tml") | None => toml::from_str(&contents)?,
            Some(other) => {
                return Err(TrainingError::ConfigFormat(format!(
                    "unsupported configuration extension '{}'",
                    other
                )));
            }
        };

        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        config.apply_base_path(base_dir);
        config.validate()?;

        Ok(config)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, TrainingError> {
        Self::from_path(path)
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        let mut errors = Vec::new();

        if self.tokenizer.vocab.is_none() && self.tokenizer.tokenizer_json.is_none() {
            errors
                .push("tokenizer must provide either `vocab` or `tokenizer_json` path".to_string());
        }

        if self.data.train_shards.is_empty() {
            errors.push("data.train_shards must not be empty".to_string());
        }

        if self.data.batch_size == 0 {
            errors.push("data.batch_size must be greater than 0".to_string());
        }

        if self.data.gradient_accumulation_steps == 0 {
            errors.push("data.gradient_accumulation_steps must be greater than 0".to_string());
        }

        if self.optimizer.learning_rate <= 0.0 {
            errors.push("optimizer.learning_rate must be greater than 0".to_string());
        }

        if self.optimizer.weight_decay < 0.0 {
            errors.push("optimizer.weight_decay must be >= 0".to_string());
        }

        if !(0.0 < self.optimizer.beta1 && self.optimizer.beta1 < 1.0) {
            errors.push("optimizer.beta1 must be in (0, 1)".to_string());
        }

        if !(0.0 < self.optimizer.beta2 && self.optimizer.beta2 < 1.0) {
            errors.push("optimizer.beta2 must be in (0, 1)".to_string());
        }

        if self.runtime.log_every_n_steps == 0 {
            errors.push("runtime.log_every_n_steps must be greater than 0".to_string());
        }

        if let (Some(heads), Some(kv_heads)) = (
            self.model.num_attention_heads,
            self.model.num_key_value_heads,
        ) {
            if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
                errors.push(
                    "model.num_attention_heads must be divisible by model.num_key_value_heads"
                        .to_string(),
                );
            }
        }

        if let Some(steps) = self.scheduler.total_steps {
            if steps == 0 {
                errors.push("scheduler.total_steps must be greater than 0".to_string());
            }
        }

        if let Some(epochs) = self.scheduler.total_epochs {
            if epochs == 0 {
                errors.push("scheduler.total_epochs must be greater than 0".to_string());
            }
        }

        if let (Some(warmup), Some(total)) =
            (self.scheduler.warmup_steps, self.scheduler.total_steps)
        {
            if warmup > total {
                errors
                    .push("scheduler.warmup_steps cannot exceed scheduler.total_steps".to_string());
            }
        }

        if let Some(min_lr) = self.scheduler.min_lr {
            if min_lr < 0.0 {
                errors.push("scheduler.min_lr must be >= 0".to_string());
            }
            if min_lr > self.optimizer.learning_rate {
                errors.push("scheduler.min_lr cannot exceed optimizer.learning_rate".to_string());
            }
        }

        if let Some(max_lr) = self.scheduler.max_lr {
            if max_lr <= 0.0 {
                errors.push("scheduler.max_lr must be greater than 0".to_string());
            }
            if let Some(min_lr) = self.scheduler.min_lr {
                if max_lr < min_lr {
                    errors.push("scheduler.max_lr must be >= scheduler.min_lr".to_string());
                }
            }
        }

        if let Some(checkpoint) = &self.runtime.checkpoint {
            if checkpoint.directory.as_os_str().is_empty() {
                errors.push("runtime.checkpoint.directory must not be empty".to_string());
            }
            if checkpoint.every_n_steps.unwrap_or(0) == 0
                && checkpoint.every_n_epochs.unwrap_or(0) == 0
            {
                errors.push(
                    "runtime.checkpoint must specify `every_n_steps` and/or `every_n_epochs`"
                        .to_string(),
                );
            }
            if let Some(0) = checkpoint.every_n_steps {
                errors.push("runtime.checkpoint.every_n_steps must be greater than 0".to_string());
            }
            if let Some(0) = checkpoint.every_n_epochs {
                errors.push("runtime.checkpoint.every_n_epochs must be greater than 0".to_string());
            }
            if let Some(0) = checkpoint.max_keep {
                errors.push("runtime.checkpoint.max_keep must be greater than 0".to_string());
            }
        }

        if let Some(0) = self.runtime.evaluation.every_n_steps {
            errors.push("runtime.evaluation.every_n_steps must be greater than 0".to_string());
        }

        if let Some(0) = self.runtime.evaluation.every_n_epochs {
            errors.push("runtime.evaluation.every_n_epochs must be greater than 0".to_string());
        }

        if !errors.is_empty() {
            return Err(TrainingError::validation(errors));
        }

        Ok(())
    }

    fn apply_base_path(&mut self, base: &Path) {
        self.tokenizer.apply_base_path(base);
        self.data.apply_base_path(base);
        self.runtime.apply_base_path(base);
    }

    pub fn resolve_model_hyperparameters(
        &self,
    ) -> Result<ResolvedModelHyperparameters, TrainingError> {
        self.ensure_prerequisites()?;

        let precision = self.runtime.precision;
        let dtype = precision_to_dtype(precision);

        let vocab_size = self
            .model
            .vocab_size
            .or(self.detect_vocab_size()?)
            .unwrap_or(DEFAULT_MODEL_VOCAB_SIZE);

        let hidden_dim = self.model.hidden_size.unwrap_or(DEFAULT_MODEL_HIDDEN_SIZE);
        let n_layers = self.model.num_layers.unwrap_or(DEFAULT_MODEL_NUM_LAYERS);
        let n_heads = self
            .model
            .num_attention_heads
            .unwrap_or(DEFAULT_MODEL_NUM_HEADS);

        if n_heads == 0 {
            return Err(TrainingError::initialization(
                "model.num_attention_heads must be greater than zero",
            ));
        }
        if hidden_dim % n_heads != 0 {
            return Err(TrainingError::initialization(format!(
                "hidden size {} must be divisible by num_attention_heads {}",
                hidden_dim, n_heads
            )));
        }
        let head_dim = hidden_dim / n_heads;

        let ff_ratio = self
            .model
            .intermediate_size
            .map(|width| width as f32 / hidden_dim as f32)
            .unwrap_or(DEFAULT_MODEL_FF_RATIO);

        if ff_ratio <= 0.0 {
            return Err(TrainingError::initialization(format!(
                "feed-forward ratio must be positive (got {})",
                ff_ratio
            )));
        }

        let norm_kind = DEFAULT_MODEL_NORM_KIND;
        let rope_mode = match self.model.rope_mode.as_ref() {
            Some(value) => parse_rope_mode(value)?,
            None => DEFAULT_ROPE_MODE,
        };

        let attn_dropout = normalize_dropout("model.attn_dropout", self.model.attn_dropout)?;
        let residual_dropout =
            normalize_dropout("model.residual_dropout", self.model.residual_dropout)?;

        let max_positions = self
            .model
            .max_position_embeddings
            .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS);

        let base = ModelConfig {
            vocab_size,
            hidden_dim,
            n_layers,
            n_heads,
            head_dim,
            ff_ratio,
            norm_kind,
            rope_mode: RopeMode::Off,
            dtype: DType::F32,
            device: Device::Cpu,
            attn_dropout_p: None,
            residual_dropout_p: None,
        };

        let overrides = ModelConfigOverrides {
            attn_dropout_p: Some(attn_dropout),
            residual_dropout_p: Some(residual_dropout),
            dtype: Some(dtype),
            rope_mode: Some(rope_mode),
        };

        let model = build_model_config_with_overrides(base, overrides).map_err(|err| {
            TrainingError::initialization(format!("failed to build model config: {}", err))
        })?;

        Ok(ResolvedModelHyperparameters {
            model,
            max_position_embeddings: max_positions,
            precision,
        })
    }

    fn ensure_prerequisites(&self) -> Result<(), TrainingError> {
        let mut missing = Vec::new();

        if let Some(path) = &self.tokenizer.tokenizer_json {
            if !path.is_file() {
                missing.push(format!("tokenizer.tokenizer_json ({})", path.display()));
            }
        } else {
            if let Some(path) = &self.tokenizer.vocab {
                if !path.is_file() {
                    missing.push(format!("tokenizer.vocab ({})", path.display()));
                }
            }
            if let Some(path) = &self.tokenizer.merges {
                if !path.is_file() {
                    missing.push(format!("tokenizer.merges ({})", path.display()));
                }
            }
        }

        if let Some(path) = &self.tokenizer.special_tokens {
            if !path.exists() {
                missing.push(format!("tokenizer.special_tokens ({})", path.display()));
            }
        }

        for shard in &self.data.train_shards {
            if !shard.exists() {
                missing.push(format!("data.train_shards entry ({})", shard.display()));
            }
        }
        for shard in &self.data.validation_shards {
            if !shard.exists() {
                missing.push(format!(
                    "data.validation_shards entry ({})",
                    shard.display()
                ));
            }
        }

        if !missing.is_empty() {
            return Err(TrainingError::initialization(format!(
                "missing required artifacts: {}",
                missing.join(", ")
            )));
        }

        Ok(())
    }

    fn detect_vocab_size(&self) -> Result<Option<usize>, TrainingError> {
        if let Some(path) = &self.tokenizer.tokenizer_json {
            if path.is_file() {
                let tokenizer = Tokenizer::from_file(path).map_err(|err| {
                    TrainingError::initialization(format!(
                        "failed to load tokenizer json {}: {}",
                        path.display(),
                        err
                    ))
                })?;
                return Ok(Some(tokenizer.get_vocab_size(true)));
            }
        }

        if let Some(path) = &self.tokenizer.vocab {
            if path.is_file() {
                let contents = fs::read_to_string(path).map_err(|err| {
                    TrainingError::initialization(format!(
                        "failed to read vocab json {}: {}",
                        path.display(),
                        err
                    ))
                })?;
                let value: serde_json::Value = serde_json::from_str(&contents).map_err(|err| {
                    TrainingError::initialization(format!(
                        "failed to parse vocab json {}: {}",
                        path.display(),
                        err
                    ))
                })?;
                if let Some(map) = value.as_object() {
                    return Ok(Some(map.len()));
                }
            }
        }

        Ok(None)
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelOverrides {
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub num_layers: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub attn_dropout: Option<f32>,
    #[serde(default)]
    pub residual_dropout: Option<f32>,
    #[serde(default)]
    pub rope_mode: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub tokenizer_json: Option<PathBuf>,
    #[serde(default)]
    pub vocab: Option<PathBuf>,
    #[serde(default)]
    pub merges: Option<PathBuf>,
    #[serde(default)]
    pub special_tokens: Option<PathBuf>,
}

impl TokenizerConfig {
    fn apply_base_path(&mut self, base: &Path) {
        for path in [
            self.tokenizer_json.as_mut(),
            self.vocab.as_mut(),
            self.merges.as_mut(),
            self.special_tokens.as_mut(),
        ] {
            if let Some(path) = path {
                absolutize_in_place(path, base);
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct DataConfig {
    pub train_shards: Vec<PathBuf>,
    #[serde(default)]
    pub validation_shards: Vec<PathBuf>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: usize,
    #[serde(default)]
    pub num_workers: Option<usize>,
    #[serde(default)]
    pub shuffle_buffer_size: Option<usize>,
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
}

impl DataConfig {
    fn apply_base_path(&mut self, base: &Path) {
        for shard in &mut self.train_shards {
            absolutize_in_place(shard, base);
        }
        for shard in &mut self.validation_shards {
            absolutize_in_place(shard, base);
        }
        if let Some(cache_dir) = self.cache_dir.as_mut() {
            absolutize_in_place(cache_dir, base);
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct OptimizerConfig {
    #[serde(default)]
    pub algorithm: OptimizerType,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    #[serde(default)]
    pub weight_decay: f32,
    #[serde(default = "default_beta1")]
    pub beta1: f32,
    #[serde(default = "default_beta2")]
    pub beta2: f32,
    #[serde(default = "default_adam_eps")]
    pub epsilon: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            algorithm: OptimizerType::default(),
            learning_rate: default_learning_rate(),
            weight_decay: 0.0,
            beta1: default_beta1(),
            beta2: default_beta2(),
            epsilon: default_adam_eps(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerType {
    AdamW,
    Adam,
    Sgd,
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::AdamW
    }
}

#[derive(Debug, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default)]
    pub strategy: LearningRateSchedule,
    #[serde(default)]
    pub warmup_steps: Option<usize>,
    #[serde(default)]
    pub total_steps: Option<usize>,
    #[serde(default)]
    pub total_epochs: Option<usize>,
    #[serde(default)]
    pub min_lr: Option<f32>,
    #[serde(default)]
    pub max_lr: Option<f32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            strategy: LearningRateSchedule::default(),
            warmup_steps: None,
            total_steps: None,
            total_epochs: None,
            min_lr: None,
            max_lr: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningRateSchedule {
    Constant,
    Cosine,
    LinearWarmup,
    CosineWithWarmup,
    Polynomial,
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::Constant
    }
}

#[derive(Debug, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_precision")]
    pub precision: Precision,
    #[serde(default = "default_log_every_n_steps")]
    pub log_every_n_steps: usize,
    #[serde(default)]
    pub checkpoint: Option<CheckpointConfig>,
    #[serde(default)]
    pub evaluation: EvaluationConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            seed: default_seed(),
            precision: default_precision(),
            log_every_n_steps: default_log_every_n_steps(),
            checkpoint: None,
            evaluation: EvaluationConfig::default(),
        }
    }
}

impl RuntimeConfig {
    fn apply_base_path(&mut self, base: &Path) {
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint.apply_base_path(base);
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CheckpointConfig {
    pub directory: PathBuf,
    #[serde(default)]
    pub every_n_steps: Option<usize>,
    #[serde(default)]
    pub every_n_epochs: Option<usize>,
    #[serde(default)]
    pub max_keep: Option<usize>,
}

impl CheckpointConfig {
    fn apply_base_path(&mut self, base: &Path) {
        absolutize_in_place(&mut self.directory, base);
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct EvaluationConfig {
    #[serde(default)]
    pub every_n_steps: Option<usize>,
    #[serde(default)]
    pub every_n_epochs: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    Mixed,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::Bf16
    }
}

fn absolutize_in_place(path: &mut PathBuf, base: &Path) {
    if path.is_relative() {
        *path = base.join(&*path);
    }
}

const DEFAULT_MODEL_VOCAB_SIZE: usize = 32_000;
const DEFAULT_MODEL_HIDDEN_SIZE: usize = 4_096;
const DEFAULT_MODEL_NUM_LAYERS: usize = 32;
const DEFAULT_MODEL_NUM_HEADS: usize = 32;
const DEFAULT_MODEL_FF_RATIO: f32 = 4.0;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 2_048;
const DEFAULT_MODEL_NORM_KIND: NormKind = NormKind::RmsNorm;
const DEFAULT_ROPE_MODE: RopeMode = RopeMode::OnTheFly;

fn default_batch_size() -> usize {
    8
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

fn default_learning_rate() -> f32 {
    3e-4
}

fn default_beta1() -> f32 {
    0.9
}

fn default_beta2() -> f32 {
    0.95
}

fn default_adam_eps() -> f32 {
    1e-8
}

fn default_seed() -> u64 {
    42
}

fn default_precision() -> Precision {
    Precision::Bf16
}

fn default_log_every_n_steps() -> usize {
    100
}

fn normalize_dropout(label: &str, value: Option<f32>) -> Result<Option<f32>, TrainingError> {
    match value {
        Some(dropout) if dropout < 0.0 || dropout >= 1.0 => {
            Err(TrainingError::validation(vec![format!(
                "{} must be in [0, 1) (got {})",
                label, dropout
            )]))
        }
        Some(dropout) if dropout <= 0.0 => Ok(None),
        Some(dropout) => Ok(Some(dropout)),
        None => Ok(None),
    }
}

fn parse_rope_mode(value: &str) -> Result<RopeMode, TrainingError> {
    match value.to_ascii_lowercase().as_str() {
        "off" | "disabled" => Ok(RopeMode::Off),
        "preapply" | "pre-applied" | "pre" => Ok(RopeMode::Preapply),
        "on_the_fly" | "onthefly" | "online" | "auto" => Ok(RopeMode::OnTheFly),
        other => Err(TrainingError::validation(vec![format!(
            "unsupported rope_mode override '{}'",
            other
        )])),
    }
}

fn precision_to_dtype(precision: Precision) -> DType {
    match precision {
        Precision::Fp32 => DType::F32,
        Precision::Fp16 => DType::F16,
        Precision::Bf16 => DType::BF16,
        Precision::Mixed => DType::BF16,
    }
}

#[derive(Debug)]
pub enum TrainingError {
    Io(std::io::Error),
    ConfigFormat(String),
    Validation(Vec<String>),
    Initialization(String),
    Runtime(String),
}

impl TrainingError {
    pub fn initialization(message: impl Into<String>) -> Self {
        Self::Initialization(message.into())
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    pub fn validation(messages: Vec<String>) -> Self {
        Self::Validation(messages)
    }
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingError::Io(err) => write!(f, "failed to read config: {}", err),
            TrainingError::ConfigFormat(err) => write!(f, "failed to parse config: {}", err),
            TrainingError::Validation(messages) => {
                write!(f, "invalid configuration: {}", messages.join("; "))
            }
            TrainingError::Initialization(msg) => {
                write!(f, "trainer initialization failed: {}", msg)
            }
            TrainingError::Runtime(msg) => write!(f, "training failed: {}", msg),
        }
    }
}

impl std::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrainingError::Io(err) => Some(err),
            TrainingError::ConfigFormat(_) => None,
            TrainingError::Validation(_) => None,
            TrainingError::Initialization(_) | TrainingError::Runtime(_) => None,
        }
    }
}

impl From<std::io::Error> for TrainingError {
    fn from(value: std::io::Error) -> Self {
        TrainingError::Io(value)
    }
}

impl From<toml::de::Error> for TrainingError {
    fn from(value: toml::de::Error) -> Self {
        TrainingError::ConfigFormat(value.to_string())
    }
}

impl From<serde_json::Error> for TrainingError {
    fn from(value: serde_json::Error) -> Self {
        TrainingError::ConfigFormat(value.to_string())
    }
}
