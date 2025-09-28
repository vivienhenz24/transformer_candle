use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::safetensors::load as load_safetensors;
use hex::encode as hex_encode;
use model::Model;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    optimizer::{GradientScaler, GradientScalerState, OptimizerState, TrainerOptimizer},
    scheduler::{LRScheduler, SchedulerState},
    TrainingConfig, TrainingError,
};

pub const CHECKPOINT_VERSION: u32 = 1;
const MODEL_FILENAME: &str = "model.safetensors";
const OPTIMIZER_FILENAME: &str = "optimizer.json";
const SCHEDULER_FILENAME: &str = "scheduler.json";
const SCALER_FILENAME: &str = "scaler.json";
const MANIFEST_FILENAME: &str = "manifest.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub filename: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingProgressSnapshot {
    pub optimizer_step: usize,
    pub global_step: usize,
    pub epoch: usize,
    pub micro_batch_index: usize,
    pub micro_batches_per_step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RngSnapshot {
    pub master_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub version: u32,
    pub created_unix_timestamp: u64,
    pub config_sha256: String,
    pub model: FileRecord,
    pub optimizer: FileRecord,
    pub scheduler: Option<FileRecord>,
    pub scaler: FileRecord,
    pub progress: TrainingProgressSnapshot,
    pub rng: RngSnapshot,
}

pub struct SaveRequest<'a> {
    pub base_dir: &'a Path,
    pub config: &'a TrainingConfig,
    pub model: &'a Model,
    pub optimizer: &'a TrainerOptimizer,
    pub scheduler: Option<&'a (dyn LRScheduler)>,
    pub scaler: &'a GradientScaler,
    pub progress: TrainingProgressSnapshot,
    pub rng: RngSnapshot,
    pub max_keep: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CheckpointDescriptor {
    pub directory: PathBuf,
    pub manifest: CheckpointManifest,
}

pub struct LoadOutcome {
    pub manifest: CheckpointManifest,
    pub optimizer_state: OptimizerState,
    pub scheduler_state: Option<SchedulerState>,
    pub scaler_state: GradientScalerState,
    pub model_weights_path: PathBuf,
}

pub fn save_checkpoint(request: SaveRequest<'_>) -> Result<CheckpointDescriptor, TrainingError> {
    fs::create_dir_all(request.base_dir).map_err(|err| {
        TrainingError::runtime(format!(
            "failed to create checkpoint directory {}: {err}",
            request.base_dir.display()
        ))
    })?;

    let dir_name = format!(
        "step_{:012}_global_{:012}",
        request.progress.optimizer_step, request.progress.global_step
    );
    let checkpoint_dir = request.base_dir.join(dir_name);
    if checkpoint_dir.exists() {
        fs::remove_dir_all(&checkpoint_dir).map_err(|err| {
            TrainingError::runtime(format!(
                "failed to remove existing checkpoint directory {}: {err}",
                checkpoint_dir.display()
            ))
        })?;
    }
    fs::create_dir(&checkpoint_dir).map_err(|err| {
        TrainingError::runtime(format!(
            "failed to create checkpoint directory {}: {err}",
            checkpoint_dir.display()
        ))
    })?;

    let model_path = checkpoint_dir.join(MODEL_FILENAME);
    save_model_weights(request.model, &model_path)?;
    let model_record = file_record(&model_path)?;

    let optimizer_state = request.optimizer.state()?;
    let optimizer_path = checkpoint_dir.join(OPTIMIZER_FILENAME);
    write_json(&optimizer_path, &optimizer_state)?;
    let optimizer_record = file_record(&optimizer_path)?;

    let scheduler_record = if let Some(scheduler) = request.scheduler {
        let scheduler_state = scheduler.snapshot();
        let scheduler_path = checkpoint_dir.join(SCHEDULER_FILENAME);
        write_json(&scheduler_path, &scheduler_state)?;
        Some(file_record(&scheduler_path)?)
    } else {
        None
    };

    let scaler_state = request.scaler.state();
    let scaler_path = checkpoint_dir.join(SCALER_FILENAME);
    write_json(&scaler_path, &scaler_state)?;
    let scaler_record = file_record(&scaler_path)?;

    let manifest = CheckpointManifest {
        version: CHECKPOINT_VERSION,
        created_unix_timestamp: unix_timestamp(),
        config_sha256: fingerprint_config(request.config)?,
        model: model_record,
        optimizer: optimizer_record,
        scheduler: scheduler_record,
        scaler: scaler_record,
        progress: request.progress,
        rng: request.rng,
    };

    let manifest_path = checkpoint_dir.join(MANIFEST_FILENAME);
    write_json(&manifest_path, &manifest)?;

    prune_checkpoints(request.base_dir, request.max_keep)?;

    Ok(CheckpointDescriptor {
        directory: checkpoint_dir,
        manifest,
    })
}

pub fn latest_checkpoint(base_dir: &Path) -> Result<Option<CheckpointDescriptor>, TrainingError> {
    let entries = checkpoint_directories(base_dir)?;
    let Some(path) = entries.into_iter().max() else {
        return Ok(None);
    };
    let manifest = load_manifest(&path)?;
    Ok(Some(CheckpointDescriptor {
        directory: path,
        manifest,
    }))
}

pub fn load_checkpoint(directory: &Path) -> Result<LoadOutcome, TrainingError> {
    let manifest = load_manifest(directory)?;
    ensure_version_supported(manifest.version)?;

    let model_path = directory.join(&manifest.model.filename);
    validate_file(&model_path, &manifest.model.sha256)?;

    let optimizer_path = directory.join(&manifest.optimizer.filename);
    validate_file(&optimizer_path, &manifest.optimizer.sha256)?;
    let optimizer_state: OptimizerState = read_json(&optimizer_path)?;

    let scheduler_state = if let Some(record) = manifest.scheduler.as_ref() {
        let path = directory.join(&record.filename);
        validate_file(&path, &record.sha256)?;
        let state: SchedulerState = read_json(&path)?;
        Some(state)
    } else {
        None
    };

    let scaler_path = directory.join(&manifest.scaler.filename);
    validate_file(&scaler_path, &manifest.scaler.sha256)?;
    let scaler_state: GradientScalerState = read_json(&scaler_path)?;

    Ok(LoadOutcome {
        manifest,
        optimizer_state,
        scheduler_state,
        scaler_state,
        model_weights_path: model_path,
    })
}

pub fn apply_model_weights(model: &Model, weights_path: &Path) -> Result<(), TrainingError> {
    let device = model.config().device.clone();
    let tensors = load_safetensors(weights_path, &device).map_err(candle_to_training_error)?;
    let mut params_by_name: std::collections::HashMap<_, _> = tensors.into_iter().collect();

    for (name, var) in model.parameters() {
        let tensor = params_by_name.remove(&name).ok_or_else(|| {
            TrainingError::runtime(format!("checkpoint missing parameter {name}"))
        })?;
        let desired_dtype = var.as_tensor().dtype();
        let tensor = if tensor.dtype() == desired_dtype {
            tensor
        } else {
            tensor
                .to_dtype(desired_dtype)
                .map_err(candle_to_training_error)?
        };
        var.set(&tensor).map_err(candle_to_training_error)?;
    }

    if !params_by_name.is_empty() {
        let extra = params_by_name
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(TrainingError::runtime(format!(
            "checkpoint contains unused parameters: {extra}"
        )));
    }

    Ok(())
}

fn save_model_weights(model: &Model, path: &Path) -> Result<(), TrainingError> {
    let named_parameters = model.parameters();
    if named_parameters.is_empty() {
        return Err(TrainingError::runtime(
            "model contains no parameters to checkpoint",
        ));
    }
    let mut tensors = std::collections::HashMap::with_capacity(named_parameters.len());
    for (name, var) in named_parameters {
        tensors.insert(name, var.as_tensor().clone());
    }
    candle_core::safetensors::save(&tensors, path).map_err(|err| {
        TrainingError::runtime(format!(
            "failed to serialize model weights to {}: {err}",
            path.display()
        ))
    })
}

fn fingerprint_config(config: &TrainingConfig) -> Result<String, TrainingError> {
    let json = serde_json::to_vec(config)
        .map_err(|err| TrainingError::runtime(format!("failed to hash config: {err}")))?;
    Ok(hex_encode(Sha256::digest(json)))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn file_record(path: &Path) -> Result<FileRecord, TrainingError> {
    let sha = sha256_file(path)?;
    let bytes = path
        .metadata()
        .map_err(|err| {
            TrainingError::runtime(format!(
                "failed to stat checkpoint file {}: {err}",
                path.display()
            ))
        })?
        .len() as u64;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            TrainingError::runtime(format!(
                "checkpoint file name is not valid UTF-8: {}",
                path.display()
            ))
        })?
        .to_string();
    Ok(FileRecord {
        filename,
        sha256: sha,
        bytes,
    })
}

fn checkpoint_directories(base: &Path) -> Result<Vec<PathBuf>, TrainingError> {
    let mut dirs = Vec::new();
    if !base.exists() {
        return Ok(dirs);
    }
    for entry in fs::read_dir(base).map_err(|err| {
        TrainingError::runtime(format!(
            "failed to read checkpoint directory {}: {err}",
            base.display()
        ))
    })? {
        let entry = entry.map_err(|err| {
            TrainingError::runtime(format!("failed to read checkpoint entry: {err}"))
        })?;
        let file_type = entry.file_type().map_err(|err| {
            TrainingError::runtime(format!(
                "failed to inspect checkpoint entry {}: {err}",
                entry.path().display()
            ))
        })?;
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("step_") {
            dirs.push(entry.path());
        }
    }
    Ok(dirs)
}

fn load_manifest(directory: &Path) -> Result<CheckpointManifest, TrainingError> {
    let manifest_path = directory.join(MANIFEST_FILENAME);
    if !manifest_path.is_file() {
        return Err(TrainingError::runtime(format!(
            "checkpoint manifest not found at {}",
            manifest_path.display()
        )));
    }
    read_json(&manifest_path)
}

fn ensure_version_supported(version: u32) -> Result<(), TrainingError> {
    if version != CHECKPOINT_VERSION {
        return Err(TrainingError::runtime(format!(
            "unsupported checkpoint version {} (expected {})",
            version, CHECKPOINT_VERSION
        )));
    }
    Ok(())
}

fn validate_file(path: &Path, expected_sha: &str) -> Result<(), TrainingError> {
    let actual = sha256_file(path)?;
    if actual != expected_sha {
        return Err(TrainingError::runtime(format!(
            "checkpoint file {} failed checksum validation",
            path.display()
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, TrainingError> {
    let mut file = File::open(path).map_err(|err| {
        TrainingError::runtime(format!("failed to open {}: {err}", path.display()))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|err| {
            TrainingError::runtime(format!("failed to read {}: {err}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_encode(hasher.finalize()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), TrainingError> {
    let mut file = File::create(path).map_err(|err| {
        TrainingError::runtime(format!("failed to create {}: {err}", path.display()))
    })?;
    let data = serde_json::to_vec_pretty(value)
        .map_err(|err| TrainingError::runtime(format!("failed to serialize JSON: {err}")))?;
    file.write_all(&data).map_err(|err| {
        TrainingError::runtime(format!("failed to write {}: {err}", path.display()))
    })?;
    file.write_all(b"\n")
        .map_err(|err| TrainingError::runtime(format!("failed to write {}: {err}", path.display())))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, TrainingError> {
    let file = File::open(path).map_err(|err| {
        TrainingError::runtime(format!("failed to open {}: {err}", path.display()))
    })?;
    serde_json::from_reader(file).map_err(|err| {
        TrainingError::runtime(format!("failed to parse JSON {}: {err}", path.display()))
    })
}

fn prune_checkpoints(base: &Path, max_keep: Option<usize>) -> Result<(), TrainingError> {
    let Some(limit) = max_keep else {
        return Ok(());
    };
    if limit == 0 {
        return Ok(());
    }
    let mut dirs = checkpoint_directories(base)?;
    dirs.sort();
    while dirs.len() > limit {
        let victim = dirs.remove(0);
        fs::remove_dir_all(&victim).map_err(|err| {
            TrainingError::runtime(format!(
                "failed to prune checkpoint {}: {err}",
                victim.display()
            ))
        })?;
    }
    Ok(())
}

fn candle_to_training_error(err: candle_core::Error) -> TrainingError {
    TrainingError::runtime(err.to_string())
}
