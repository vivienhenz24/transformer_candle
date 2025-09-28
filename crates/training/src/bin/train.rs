use std::{
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
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
