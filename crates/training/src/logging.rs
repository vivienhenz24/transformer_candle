use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use bytes::BytesMut;
use crc32fast::Hasher as Crc32;
use prost::Message;

use crate::{
    metrics::{EvaluationSummary, StepSnapshot},
    TrainingError,
};

#[derive(Clone, Debug)]
pub struct LoggingSettings {
    pub enable_stdout: bool,
    pub tensorboard_dir: Option<PathBuf>,
    pub tensorboard_flush_every_n: usize,
}

impl LoggingSettings {
    pub fn from_config(
        enable_stdout: bool,
        tensorboard_dir: Option<PathBuf>,
        flush_every: usize,
    ) -> Self {
        Self {
            enable_stdout,
            tensorboard_dir,
            tensorboard_flush_every_n: flush_every.max(1),
        }
    }
}

pub struct Logger {
    settings: LoggingSettings,
    tensorboard: Option<TensorBoardWriter>,
}

impl Logger {
    pub fn new(settings: LoggingSettings) -> Result<Self, TrainingError> {
        let tensorboard = if let Some(dir) = settings.tensorboard_dir.as_ref() {
            Some(TensorBoardWriter::create(
                dir,
                settings.tensorboard_flush_every_n,
            )?)
        } else {
            None
        };
        Ok(Self {
            settings,
            tensorboard,
        })
    }

    pub fn log_training_step(&mut self, step: usize, lr: f64, snapshot: &StepSnapshot) {
        if self.settings.enable_stdout {
            println!(
                "train step={} loss={:.4} tokens={} tok/s={:.1} grad_norm={:.3} lr={:.5e}",
                step,
                snapshot.step_loss,
                snapshot.tokens,
                snapshot.step_tokens_per_sec,
                snapshot.raw_grad_norm,
                lr
            );
        }

        if let Some(writer) = self.tensorboard.as_mut() {
            let step_i64 = step as i64;
            let _ = writer.write_scalar("train/loss", step_i64, snapshot.step_loss);
            let _ = writer.write_scalar("train/loss_ema", step_i64, snapshot.loss);
            let _ = writer.write_scalar("train/tokens_per_sec", step_i64, snapshot.tokens_per_sec);
            let _ = writer.write_scalar("train/grad_norm", step_i64, snapshot.raw_grad_norm);
            let _ = writer.write_scalar("train/learning_rate", step_i64, lr);
        }
    }

    pub fn log_evaluation(&mut self, step: usize, summary: &EvaluationSummary) {
        if self.settings.enable_stdout {
            println!(
                "eval step={} loss={:.4} ppl={:.4} acc={:.2}% tokens={}",
                step,
                summary.average_loss,
                summary.perplexity,
                summary.accuracy * 100.0,
                summary.tokens
            );
        }

        if let Some(writer) = self.tensorboard.as_mut() {
            let step_i64 = step as i64;
            let _ = writer.write_scalar("eval/loss", step_i64, summary.average_loss);
            let _ = writer.write_scalar("eval/perplexity", step_i64, summary.perplexity);
            let _ = writer.write_scalar("eval/accuracy", step_i64, summary.accuracy);
            let _ = writer.write_scalar("eval/tokens", step_i64, summary.tokens as f64);
        }
    }

    pub fn flush(&mut self) {
        if let Some(writer) = self.tensorboard.as_mut() {
            let _ = writer.flush();
        }
    }
}

struct TensorBoardWriter {
    writer: BufWriter<File>,
    flush_every: usize,
    pending: usize,
}

impl TensorBoardWriter {
    fn create(dir: &Path, flush_every: usize) -> Result<Self, TrainingError> {
        fs::create_dir_all(dir).map_err(|err| {
            TrainingError::runtime(format!(
                "failed to create tensorboard directory {}: {err}",
                dir.display()
            ))
        })?;
        let timestamp = current_unix_timestamp();
        let hostname = hostname();
        let filename = format!("events.out.tfevents.{}.{}", timestamp, hostname);
        let path = dir.join(filename);
        let file = File::create(&path).map_err(|err| {
            TrainingError::runtime(format!(
                "failed to create tensorboard file {}: {err}",
                path.display()
            ))
        })?;
        Ok(Self {
            writer: BufWriter::new(file),
            flush_every: flush_every.max(1),
            pending: 0,
        })
    }

    fn write_scalar(&mut self, tag: &str, step: i64, value: f64) -> Result<(), TrainingError> {
        let wall_time = current_wall_time();
        let summary = Summary {
            value: vec![summary::Value {
                tag: tag.to_string(),
                simple_value: Some(value as f32),
            }],
        };
        let event = Event {
            wall_time,
            step,
            summary: Some(summary),
        };
        self.write_event(&event)
    }

    fn write_event(&mut self, event: &Event) -> Result<(), TrainingError> {
        let mut buffer = BytesMut::with_capacity(128);
        event.encode(&mut buffer).map_err(|err| {
            TrainingError::runtime(format!("failed to encode tensorboard event: {err}"))
        })?;

        let data = buffer.freeze();
        let len = data.len() as u64;

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&len.to_le_bytes());
        let len_crc = masked_crc32(&len_bytes);
        let data_crc = masked_crc32(data.as_ref());

        let len_crc_bytes = len_crc.to_le_bytes();
        let data_crc_bytes = data_crc.to_le_bytes();

        self.writer
            .write_all(&len_bytes)
            .and_then(|_| self.writer.write_all(&len_crc_bytes))
            .and_then(|_| self.writer.write_all(&data))
            .and_then(|_| self.writer.write_all(&data_crc_bytes))
            .map_err(|err| {
                TrainingError::runtime(format!("failed to write tensorboard event: {err}"))
            })?;

        self.pending += 1;
        if self.pending >= self.flush_every {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), TrainingError> {
        self.writer.flush().map_err(|err| {
            TrainingError::runtime(format!("failed to flush tensorboard file: {err}"))
        })?;
        self.pending = 0;
        Ok(())
    }
}

impl Drop for TensorBoardWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

fn masked_crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    let crc = hasher.finalize();
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282_ead8)
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs_f64())
        .unwrap_or(0.0)
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "localhost".to_string())
}

#[derive(Clone, PartialEq, Message)]
struct Event {
    #[prost(double, tag = "1")]
    wall_time: f64,
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(message, optional, tag = "3")]
    summary: Option<Summary>,
}

#[derive(Clone, PartialEq, Message)]
struct Summary {
    #[prost(message, repeated, tag = "1")]
    value: Vec<summary::Value>,
}

mod summary {
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct Value {
        #[prost(string, tag = "7")]
        pub tag: String,
        #[prost(float, optional, tag = "2")]
        pub simple_value: Option<f32>,
    }
}
