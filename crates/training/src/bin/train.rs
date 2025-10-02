use std::{
    env,
    io::{self, BufWriter, Write},
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, SyncSender},
        Arc,
    },
    thread,
};

use clap::Parser;
use serde_json::{Number, Value};
use training::{Trainer, TrainingConfig, TrainingError};

fn main() {
    if let Err(err) = run() {
        eprintln!("training failed: {}", err);
        std::process::exit(1);
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Transformer training CLI", long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        value_name = "PATH",
        help = "Path to training config file"
    )]
    config: PathBuf,

    #[arg(
        long = "override",
        value_name = "KEY=VALUE",
        help = "Override configuration value using dot-separated paths"
    )]
    overrides: Vec<OverrideArg>,

    #[arg(long, help = "Resume from the latest checkpoint if available")]
    resume: bool,
}

#[derive(Debug, Clone)]
struct OverrideArg {
    path: String,
    value: String,
}

impl FromStr for OverrideArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (path, value) = s
            .split_once('=')
            .ok_or_else(|| "override must be in the form key=value".to_string())?;
        if path.trim().is_empty() {
            return Err("override key must not be empty".into());
        }
        Ok(Self {
            path: path.trim().to_string(),
            value: value.trim().to_string(),
        })
    }
}

fn run() -> Result<(), TrainingError> {
    let args = Args::parse();

    let mut config = TrainingConfig::load(&args.config)?;
    if !args.overrides.is_empty() {
        config = apply_overrides(config, &args.overrides)?;
    }

    config.validate()?;

    // Check if there's a streaming config in the same directory
    let config_dir = args.config.parent().unwrap_or(std::path::Path::new("."));
    let streaming_config_path = config_dir.join("streaming_config.json");

    if streaming_config_path.exists() {
        println!("🌊 Detected streaming configuration, using streaming mode");
        return run_streaming_training(config, &streaming_config_path);
    }

    let mut trainer = Trainer::new(config.clone())?;

    if args.resume {
        if let Some(descriptor) = trainer.resume_from_latest()? {
            println!(
                "resumed from checkpoint {} (step {})",
                descriptor.directory.display(),
                descriptor.manifest.progress.optimizer_step
            );
        }
    }

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let handler_flag = shutdown_flag.clone();
    ctrlc::set_handler(move || {
        handler_flag.store(true, Ordering::Relaxed);
    })
    .map_err(|err| TrainingError::runtime(format!("failed to install signal handler: {err}")))?;

    trainer.train_with_shutdown(|| shutdown_flag.load(Ordering::Relaxed))?;

    Ok(())
}

fn apply_overrides(
    config: TrainingConfig,
    overrides: &[OverrideArg],
) -> Result<TrainingConfig, TrainingError> {
    let mut value = serde_json::to_value(config).map_err(|err| {
        TrainingError::runtime(format!("failed to serialize config for overrides: {err}"))
    })?;

    for override_arg in overrides {
        let new_value = parse_override_value(&override_arg.value)?;
        set_value_at_path(&mut value, &override_arg.path, new_value)?;
    }

    serde_json::from_value(value).map_err(|err| {
        TrainingError::runtime(format!(
            "failed to deserialize config after overrides: {err}"
        ))
    })
}

fn parse_override_value(raw: &str) -> Result<Value, TrainingError> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("true") {
        return Ok(Value::Bool(true));
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return Ok(Value::Bool(false));
    }
    if trimmed.eq_ignore_ascii_case("null") {
        return Ok(Value::Null);
    }
    if let Ok(int_val) = trimmed.parse::<i64>() {
        return Ok(Value::Number(Number::from(int_val)));
    }
    if let Ok(float_val) = trimmed.parse::<f64>() {
        if let Some(number) = Number::from_f64(float_val) {
            return Ok(Value::Number(number));
        }
    }
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        if let Ok(json_val) = serde_json::from_str::<Value>(trimmed) {
            return Ok(json_val);
        }
    }
    Ok(Value::String(trimmed.to_string()))
}

fn set_value_at_path(value: &mut Value, path: &str, new_value: Value) -> Result<(), TrainingError> {
    let segments = parse_path(path)?;
    if segments.is_empty() {
        return Err(TrainingError::runtime("override path must not be empty"));
    }

    assign_at_path(value, &segments, new_value)
}

#[derive(Debug)]
struct PathSegment {
    key: String,
    index: Option<usize>,
}

fn parse_path(path: &str) -> Result<Vec<PathSegment>, TrainingError> {
    path.split('.')
        .map(|segment| {
            if let Some((base, idx_part)) = segment.split_once('[') {
                let idx_str = idx_part.trim_end_matches(']');
                let index = idx_str.parse::<usize>().map_err(|err| {
                    TrainingError::runtime(format!(
                        "invalid index in override path '{}': {}",
                        segment, err
                    ))
                })?;
                Ok(PathSegment {
                    key: base.to_string(),
                    index: Some(index),
                })
            } else {
                Ok(PathSegment {
                    key: segment.to_string(),
                    index: None,
                })
            }
        })
        .collect()
}

fn assign_at_path(
    target: &mut Value,
    segments: &[PathSegment],
    new_value: Value,
) -> Result<(), TrainingError> {
    if segments.is_empty() {
        *target = new_value;
        return Ok(());
    }

    if !target.is_object() {
        if target.is_null() {
            *target = Value::Object(serde_json::Map::new());
        } else {
            return Err(TrainingError::runtime(
                "override root must be a JSON object",
            ));
        }
    }

    let mut current = target;
    for (idx, segment) in segments.iter().enumerate() {
        let is_last = idx + 1 == segments.len();
        if !current.is_object() {
            if current.is_null() {
                *current = Value::Object(serde_json::Map::new());
            } else {
                return Err(TrainingError::runtime(format!(
                    "override path segment '{}' points to non-object value",
                    segment.key
                )));
            }
        }

        let map = current.as_object_mut().unwrap();
        let entry = map.entry(segment.key.clone()).or_insert(Value::Null);

        if let Some(array_index) = segment.index {
            if !entry.is_array() {
                if entry.is_null() {
                    *entry = Value::Array(Vec::new());
                } else {
                    return Err(TrainingError::runtime(format!(
                        "override path segment '{}' expects array value",
                        segment.key
                    )));
                }
            }
            let array = entry.as_array_mut().unwrap();
            while array.len() <= array_index {
                array.push(Value::Null);
            }
            if is_last {
                array[array_index] = new_value;
                return Ok(());
            } else {
                if array[array_index].is_null() {
                    array[array_index] = Value::Object(serde_json::Map::new());
                }
                current = &mut array[array_index];
            }
        } else if is_last {
            *entry = new_value;
            return Ok(());
        } else {
            if entry.is_null() {
                *entry = Value::Object(serde_json::Map::new());
            }
            current = entry;
        }
    }

    Ok(())
}

fn run_streaming_training(
    mut config: TrainingConfig,
    streaming_config_path: &std::path::Path,
) -> Result<(), TrainingError> {
    use pretraining_data::HuggingFaceStreamingCorpus;
    use pretraining_data::TextCorpus;
    use std::fs;

    // Load streaming config
    let streaming_json = fs::read_to_string(streaming_config_path)
        .map_err(|e| TrainingError::runtime(format!("Failed to read streaming config: {}", e)))?;

    let streaming_config: serde_json::Value = serde_json::from_str(&streaming_json)
        .map_err(|e| TrainingError::runtime(format!("Failed to parse streaming config: {}", e)))?;

    let dataset = streaming_config["dataset"]
        .as_str()
        .ok_or_else(|| TrainingError::runtime("Missing 'dataset' in streaming config"))?;
    let split = streaming_config["split"]
        .as_str()
        .ok_or_else(|| TrainingError::runtime("Missing 'split' in streaming config"))?;
    let max_samples = streaming_config["max_samples"].as_u64().map(|n| n as usize);
    let stream_batch_size = streaming_config["stream_batch_size"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(1000);

    println!("🌊 Streaming mode: {} ({})", dataset, split);

    // Create temporary shards by streaming and writing to disk
    // This is a hybrid approach: stream from HF, write temp files, use existing Trainer
    println!("📥 Streaming data from HuggingFace...");

    let temp_dir = env::var("STREAMING_TMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            streaming_config_path
                .parent()
                .map(|dir| dir.join("tmp_streaming_shards"))
                .unwrap_or_else(|| PathBuf::from("tmp_streaming_shards"))
        });
    fs::create_dir_all(&temp_dir).map_err(|err| {
        TrainingError::runtime(format!(
            "failed to create temp shard directory {}: {}",
            temp_dir.display(),
            err
        ))
    })?;

    let corpus = HuggingFaceStreamingCorpus::new(
        dataset.to_string(),
        split.to_string(),
        stream_batch_size,
        max_samples,
    )
    .map_err(|e| TrainingError::runtime(format!("Failed to create streaming corpus: {}", e)))?;

    let (job_tx, job_rx) = mpsc::sync_channel::<ShardJob>(4);
    let written_shards = Arc::new(AtomicUsize::new(0));
    let writer_progress = Arc::clone(&written_shards);
    let writer_handle = thread::spawn(move || -> Result<Vec<(usize, PathBuf)>, TrainingError> {
        let mut paths = Vec::new();
        while let Ok(job) = job_rx.recv() {
            write_shard(&job.path, &job.lines).map_err(|e| {
                TrainingError::runtime(format!("failed to write shard {:04}: {}", job.index, e))
            })?;
            writer_progress.fetch_add(1, Ordering::Relaxed);
            println!(
                "📝 Shard {:04} written ({} lines) -> {}",
                job.index,
                job.lines.len(),
                job.path.display()
            );
            paths.push((job.index, job.path));
        }
        Ok(paths)
    });

    let mut stream = corpus
        .stream()
        .map_err(|e| TrainingError::runtime(format!("Failed to start stream: {}", e)))?;

    let lines_per_shard = 100_000;
    let mut shard_index = 0usize;
    let mut total_samples = 0usize;
    let mut buffer: Vec<String> = Vec::with_capacity(lines_per_shard);

    while let Some(result) = stream.next() {
        let text = result.map_err(|e| TrainingError::runtime(format!("Stream error: {}", e)))?;
        if text.is_empty() {
            continue;
        }
        buffer.push(text);
        total_samples += 1;

        if total_samples % 100_000 == 0 {
            let written = written_shards.load(Ordering::Relaxed);
            println!(
                "🚚 Buffered {} samples (queued shards: {} | flushed shards: {})",
                total_samples, shard_index, written
            );
        }

        if buffer.len() >= lines_per_shard {
            enqueue_shard(shard_index, &temp_dir, std::mem::take(&mut buffer), &job_tx)?;
            shard_index += 1;
        }
    }

    if !buffer.is_empty() {
        enqueue_shard(shard_index, &temp_dir, buffer, &job_tx)?;
        shard_index += 1;
    }

    drop(job_tx);

    let mut shard_paths = writer_handle
        .join()
        .map_err(|_| TrainingError::runtime("shard writer thread panicked"))??;

    shard_paths.sort_by_key(|(idx, _)| *idx);
    let shard_paths: Vec<PathBuf> = shard_paths.into_iter().map(|(_, path)| path).collect();

    println!(
        "✅ Streamed {} samples into {} shards ({} written)",
        total_samples,
        shard_paths.len(),
        written_shards.load(Ordering::Relaxed)
    );

    // Update config to use temporary shards
    config.data.train_shards = shard_paths.clone();
    config.data.validation_shards = shard_paths; // Use same for validation (could split if needed)

    println!("🚀 Starting training...\n");

    // Use standard Trainer with the temporary shards
    let mut trainer = Trainer::new(config)?;

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let handler_flag = shutdown_flag.clone();
    ctrlc::set_handler(move || {
        handler_flag.store(true, Ordering::Relaxed);
    })
    .map_err(|err| TrainingError::runtime(format!("failed to install signal handler: {err}")))?;

    trainer.train_with_shutdown(|| shutdown_flag.load(Ordering::Relaxed))?;

    // Cleanup temporary shards
    // Cleaning up
    let _ = fs::remove_dir_all(&temp_dir);

    Ok(())
}

#[derive(Debug)]
struct ShardJob {
    index: usize,
    path: PathBuf,
    lines: Vec<String>,
}

fn enqueue_shard(
    index: usize,
    temp_dir: &std::path::Path,
    lines: Vec<String>,
    tx: &SyncSender<ShardJob>,
) -> Result<(), TrainingError> {
    let path = temp_dir.join(format!("shard_{:04}.txt", index));
    let job = ShardJob { index, path, lines };
    tx.send(job).map_err(|err| {
        TrainingError::runtime(format!("failed to queue shard {:04}: {}", index, err))
    })
}

fn write_shard(path: &PathBuf, lines: &[String]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for line in lines {
        writeln!(writer, "{}", line)?;
    }
    writer.flush()
}
