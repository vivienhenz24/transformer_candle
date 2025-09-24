use candle_core::{Device, Result as CandleResult, Tensor, Var};
use candle_nn::{
    optim::{AdamW, Optimizer, ParamsAdamW},
    VarBuilder, VarMap,
};
use std::{f64::consts::PI, fs, path::PathBuf, time::Instant};

use crate::gpt::{GPTConfig, GPTLanguageModel};
use crate::tokenizer::{CharTokenizer, DataSplit};

/// Configuration struct containing all training hyperparameters
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer
    pub learning_rate: f64,
    /// Batch size for training and evaluation
    pub batch_size: usize,
    /// Block size (sequence length) for training
    pub block_size: usize,
    /// Maximum number of training iterations
    pub max_iters: usize,
    /// How often to evaluate loss on validation set
    pub eval_interval: usize,
    /// Number of evaluation batches to average over
    pub eval_iters: usize,
    /// Weight decay for AdamW optimizer
    pub weight_decay: f64,
    /// Beta1 parameter for AdamW
    pub beta1: f64,
    /// Beta2 parameter for AdamW
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
    /// Device to run training on (CPU/GPU)
    pub device: Device,
    /// How often to print training progress
    pub log_interval: usize,
    /// Whether to compile the model (for potential optimizations)
    pub compile: bool,
    /// Number of warmup steps for the learning-rate scheduler
    pub warmup_steps: usize,
    /// Interval (in iterations) for writing checkpoints
    pub checkpoint_interval: usize,
    /// Directory for checkpoint files
    pub checkpoint_dir: String,
    /// Gradient clipping threshold (set <= 0 to disable)
    pub grad_clip: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            learning_rate: 5e-4, // Base learning rate for balanced preset
            batch_size: 48,      // Micro-batch size per optimizer step
            block_size: 256,     // Context window length
            max_iters: 15000,    // Balanced training schedule
            eval_interval: 400,  // Evaluate periodically without too much overhead
            eval_iters: 200,     // Average validation over multiple batches
            weight_decay: 1e-2,  // Slight regularization for AdamW
            beta1: 0.9,          // AdamW beta1
            beta2: 0.95,         // AdamW beta2 (slightly lower than default)
            eps: 1e-8,           // Epsilon for numerical stability
            device: Device::Cpu, // Default to CPU
            log_interval: 1,     // Log every iteration by default
            compile: false,      // Don't compile by default
            warmup_steps: 800,   // Learning-rate warmup duration
            checkpoint_interval: 800,
            checkpoint_dir: "checkpoints".to_string(),
            grad_clip: 1.0, // Clip gradients to this L2 norm
        }
    }
}

/// Estimates the loss on a dataset without computing gradients
/// This is used for evaluation on train/validation sets
pub fn estimate_loss(
    model: &GPTLanguageModel,
    tokenizer: &CharTokenizer,
    split: DataSplit,
    config: &TrainingConfig,
) -> CandleResult<f32> {
    let data = match split {
        DataSplit::Train => tokenizer.train_tokens(),
        DataSplit::Val => tokenizer.val_tokens(),
    };

    let mut loader = DataLoader::new(
        data,
        config.block_size,
        config.batch_size,
        &config.device,
        split == DataSplit::Train,
    )?;

    let mut total_loss = 0.0f32;

    for _ in 0..config.eval_iters {
        let (inputs, targets) = loader.next_batch()?;
        let (_logits, loss) = model.forward(&inputs, Some(&targets), false)?;

        if let Some(loss_tensor) = loss {
            total_loss += loss_tensor.to_scalar::<f32>()?;
        }
    }

    Ok(total_loss / config.eval_iters as f32)
}

/// Training statistics to track progress
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub iteration: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub learning_rate: f64,
    pub tokens_per_sec: f32,
    pub elapsed_time: f32,
    pub train_perplexity: f32,
    pub val_perplexity: f32,
}

struct DataLoader<'a> {
    data: &'a [usize],
    block_size: usize,
    batch_size: usize,
    device: Device,
    shuffle: bool,
    positions: Vec<usize>,
    cursor: usize,
}

struct CosineWarmupScheduler {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    step: usize,
}

impl CosineWarmupScheduler {
    fn new(base_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            step: 0,
        }
    }

    fn next_lr(&mut self) -> f64 {
        let lr = if self.step < self.warmup_steps {
            let denom = self.warmup_steps.max(1) as f64;
            self.base_lr * ((self.step + 1) as f64 / denom)
        } else {
            let progress_steps = (self.total_steps.saturating_sub(self.warmup_steps)).max(1);
            let progress = ((self.step - self.warmup_steps + 1) as f64) / (progress_steps as f64);
            let progress = progress.clamp(0.0, 1.0);
            let cosine = (PI * progress).cos();
            self.base_lr * 0.5 * (1.0 + cosine)
        };
        self.step += 1;
        lr
    }
}

impl<'a> DataLoader<'a> {
    fn new(
        data: &'a [usize],
        block_size: usize,
        batch_size: usize,
        device: &Device,
        shuffle: bool,
    ) -> CandleResult<Self> {
        if data.len() <= block_size + 1 {
            return Err(candle_core::Error::Msg(
                "Dataset length must exceed block size".to_string(),
            ));
        }

        let mut positions: Vec<usize> = (0..data.len() - block_size - 1).collect();
        if shuffle {
            fastrand::shuffle(&mut positions);
        }

        Ok(Self {
            data,
            block_size,
            batch_size,
            device: device.clone(),
            shuffle,
            positions,
            cursor: 0,
        })
    }

    fn reset(&mut self) {
        self.cursor = 0;
        if self.shuffle {
            fastrand::shuffle(&mut self.positions);
        }
    }

    fn next_batch(&mut self) -> CandleResult<(Tensor, Tensor)> {
        if self.cursor + self.batch_size > self.positions.len() {
            self.reset();
        }

        let slice = &self.positions[self.cursor..self.cursor + self.batch_size];
        self.cursor += self.batch_size;

        let mut inputs = Vec::with_capacity(self.batch_size * self.block_size);
        let mut targets = Vec::with_capacity(self.batch_size * self.block_size);

        for &start in slice.iter() {
            let end = start + self.block_size;
            inputs.extend(self.data[start..end].iter().map(|&token| token as u32));
            targets.extend(
                self.data[start + 1..end + 1]
                    .iter()
                    .map(|&token| token as u32),
            );
        }

        let input_tensor =
            Tensor::from_vec(inputs, (self.batch_size, self.block_size), &self.device)?;
        let target_tensor =
            Tensor::from_vec(targets, (self.batch_size, self.block_size), &self.device)?;

        Ok((input_tensor, target_tensor))
    }
}

fn gradient_l2_norm(grads: &candle_core::backprop::GradStore, vars: &[Var]) -> CandleResult<f64> {
    let mut total = 0f64;
    for var in vars {
        if let Some(grad) = grads.get(var) {
            let norm_sq = grad
                .sqr()? // element-wise square
                .sum_all()? // sum over all elements
                .to_scalar::<f32>()? as f64;
            total += norm_sq;
        }
    }
    Ok(total.sqrt())
}

fn generate_preview(model: &GPTLanguageModel, tokenizer: &CharTokenizer) -> CandleResult<String> {
    let prompt = "ROMEO:";
    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        return Ok(String::new());
    }

    let prompt_tensor = tokenizer.indices_to_tensor(&prompt_tokens)?.unsqueeze(0)?;
    let generated = model.generate_with_sampling(&prompt_tensor, 200, 0.2, Some(10), Some(0.9))?;

    let flat = generated.get(0)?;
    let indices = flat.to_vec1::<u32>()?;
    let indices: Vec<usize> = indices.into_iter().map(|v| v as usize).collect();
    Ok(tokenizer.decode(&indices))
}

/// Main training function that runs the complete training loop
/// Uses a simplified SGD-like optimizer approach that works with Candle
pub fn train_model(
    model: &GPTLanguageModel,
    tokenizer: &CharTokenizer,
    varmap: &mut VarMap,
    config: &TrainingConfig,
) -> CandleResult<Vec<TrainingStats>> {
    println!("Starting GPT training...");
    println!("Training Configuration:");
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Batch size: {}", config.batch_size);
    println!("  Block size: {}", config.block_size);
    println!("  Max iterations: {}", config.max_iters);
    println!("  Weight decay: {}", config.weight_decay);
    println!("  Device: {:?}", config.device);
    println!("  Model parameters: ~{}", model.count_parameters());
    println!();

    let mut training_stats = Vec::new();

    if config.checkpoint_interval > 0 {
        let checkpoint_dir = PathBuf::from(&config.checkpoint_dir);
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Failed to create checkpoint directory {}: {e}",
                    checkpoint_dir.display()
                ))
            })?;
        }
    }

    let all_vars = varmap.all_vars();
    let mut optimizer = AdamW::new(
        all_vars.clone(),
        ParamsAdamW {
            lr: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            weight_decay: config.weight_decay,
            eps: config.eps,
        },
    )?;

    let mut lr_scheduler =
        CosineWarmupScheduler::new(config.learning_rate, config.warmup_steps, config.max_iters);

    let mut train_loader = DataLoader::new(
        tokenizer.train_tokens(),
        config.block_size,
        config.batch_size,
        &config.device,
        true,
    )?;

    // Timing and statistics
    let training_start = Instant::now();
    let mut last_log_time = training_start;
    let mut tokens_processed = 0;

    println!(" Training Progress:");
    println!(
        "{:>6} | {:>10} | {:>10} | {:>8} | {:>10} | {:>8} | {:>12}",
        "Iter", "Train Loss", "Val Loss", "LR", "Tok/sec", "Time", "PPL t/v"
    );
    println!("{}", "-".repeat(90));

    // Main training loop
    for iter in 0..config.max_iters {
        let current_lr = lr_scheduler.next_lr();
        optimizer.set_learning_rate(current_lr);

        // Get training batch
        let (inputs, targets) = train_loader
            .next_batch()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get training batch: {e}")))?;

        // Forward pass with loss computation
        let (_logits, loss) = model
            .forward(&inputs, Some(&targets), true)
            .map_err(|e| candle_core::Error::Msg(format!("Forward pass failed: {e}")))?;

        let loss_tensor = loss.ok_or_else(|| {
            candle_core::Error::Msg("No loss computed during training".to_string())
        })?;

        let grads = loss_tensor
            .backward()
            .map_err(|e| candle_core::Error::Msg(format!("Backward pass failed: {e}")))?;

        let mut effective_lr = current_lr;
        if config.grad_clip > 0.0 {
            let grad_norm = gradient_l2_norm(&grads, &all_vars)?;
            if grad_norm.is_finite() && grad_norm > config.grad_clip {
                let scale = config.grad_clip / (grad_norm + 1e-6);
                effective_lr *= scale;
                optimizer.set_learning_rate(effective_lr);
            }
        }

        optimizer
            .step(&grads)
            .map_err(|e| candle_core::Error::Msg(format!("Optimizer step failed: {e}")))?;

        let _batch_train_loss = loss_tensor
            .to_scalar::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to extract loss value: {e}")))?;

        // Update token count
        tokens_processed += config.batch_size * config.block_size;

        // Evaluate and log progress
        if iter % config.eval_interval == 0 || iter == config.max_iters - 1 {
            // Estimate losses on train and validation sets
            let train_loss_avg = estimate_loss(model, tokenizer, DataSplit::Train, config)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Train loss estimation failed: {}", e))
                })?;

            let val_loss_avg =
                estimate_loss(model, tokenizer, DataSplit::Val, config).map_err(|e| {
                    candle_core::Error::Msg(format!("Validation loss estimation failed: {}", e))
                })?;

            let train_perplexity = train_loss_avg.exp();
            let val_perplexity = val_loss_avg.exp();

            // Calculate timing statistics
            let current_time = Instant::now();
            let elapsed_since_last = current_time.duration_since(last_log_time).as_secs_f32();
            let total_elapsed = current_time.duration_since(training_start).as_secs_f32();
            let tokens_per_sec = if elapsed_since_last > 0.0 {
                (config.batch_size * config.block_size * config.eval_interval) as f32
                    / elapsed_since_last
            } else {
                0.0
            };

            // Create training statistics
            let stats = TrainingStats {
                iteration: iter,
                train_loss: train_loss_avg,
                val_loss: val_loss_avg,
                learning_rate: effective_lr,
                tokens_per_sec,
                elapsed_time: total_elapsed,
                train_perplexity,
                val_perplexity,
            };

            training_stats.push(stats.clone());

            // Log progress
            if iter % config.log_interval == 0 {
                println!(
                    "{:6} | {:10.4} | {:10.4} | {:8.2e} | {:10.0} | {:8.1}s | ppl {:.2}/{:.2}",
                    iter,
                    stats.train_loss,
                    stats.val_loss,
                    stats.learning_rate,
                    stats.tokens_per_sec,
                    stats.elapsed_time,
                    stats.train_perplexity,
                    stats.val_perplexity,
                );

                if let Ok(sample) = generate_preview(model, tokenizer) {
                    println!("--- preview ({} chars) ---", sample.len());
                    println!("{}", sample);
                    println!("---------------------------");
                }
            }

            last_log_time = current_time;
        }

        // Handle training interruption gracefully
        if config.checkpoint_interval > 0 && iter > 0 && iter % config.checkpoint_interval == 0 {
            let filename = format!("iter_{:05}.safetensors", iter);
            let path = PathBuf::from(&config.checkpoint_dir).join(filename);
            if let Err(err) = varmap.save(&path) {
                println!(
                    "⚠️ Failed to save checkpoint at iteration {}: {}",
                    iter, err
                );
            } else {
                println!("💾 Saved checkpoint: {}", path.display());
            }
        }

        // Restore scheduled learning-rate for the next iteration
        optimizer.set_learning_rate(current_lr);
    }

    let total_time = training_start.elapsed().as_secs_f32();
    let avg_tokens_per_sec = tokens_processed as f32 / total_time;

    println!("\n Training completed!");
    println!("Total time: {:.1}s", total_time);
    println!("Average tokens/sec: {:.0}", avg_tokens_per_sec);
    println!("Total tokens processed: {}", tokens_processed);

    if let Some(final_stats) = training_stats.last() {
        println!("Final train loss: {:.4}", final_stats.train_loss);
        println!("Final validation loss: {:.4}", final_stats.val_loss);
        println!(
            "Final perplexities -> train: {:.2}, val: {:.2}",
            final_stats.train_perplexity, final_stats.val_perplexity
        );
    }

    Ok(training_stats)
}

/// Convenience function to run training with default configuration
pub fn train_gpt_model(
    gpt_config: GPTConfig,
    tokenizer: &CharTokenizer,
    device: Device,
) -> CandleResult<(GPTLanguageModel, Vec<TrainingStats>)> {
    // Create variable map for model parameters
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    // Create model
    let model = GPTLanguageModel::new(gpt_config, vb)?;

    // Set up training configuration
    let mut training_config = TrainingConfig::default();
    training_config.device = device;
    training_config.block_size = model.config().block_size;

    // Reduce iterations for demo purposes
    training_config.max_iters = 100;
    training_config.eval_interval = 20;
    training_config.eval_iters = 10;

    // Run training
    let stats = train_model(&model, tokenizer, &mut varmap, &training_config)?;

    Ok((model, stats))
}

/// Helper function to create a smaller model configuration for quick training/testing
pub fn create_small_gpt_config(vocab_size: usize) -> GPTConfig {
    GPTConfig {
        vocab_size,
        block_size: 64,    // Smaller context for faster training
        n_embd: 64,        // Smaller embedding dimension
        n_head: 4,         // Fewer attention heads
        n_layer: 2,        // Fewer layers
        dropout_rate: 0.0, // No dropout for small model
    }
}

/// Helper function to create a medium model configuration for serious training
pub fn create_medium_gpt_config(vocab_size: usize) -> GPTConfig {
    GPTConfig {
        vocab_size,
        block_size: 256,   // Balanced context length
        n_embd: 384,       // Balanced embedding dimension
        n_head: 6,         // Attention heads
        n_layer: 6,        // 6 transformer layers
        dropout_rate: 0.0, // Disable dropout for faster convergence on character data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 5e-4);
        assert_eq!(config.batch_size, 48);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.max_iters, 15000);
        assert_eq!(config.eval_interval, 400);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.95);
        assert_eq!(config.weight_decay, 1e-2);
        assert_eq!(config.warmup_steps, 800);
        assert_eq!(config.checkpoint_interval, 800);
        assert!((config.grad_clip - 1.0).abs() < f64::EPSILON);
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
        assert_eq!(config.dropout_rate, 0.0);
    }

    #[test]
    fn test_training_stats_creation() {
        let stats = TrainingStats {
            iteration: 100,
            train_loss: 2.5,
            val_loss: 2.7,
            learning_rate: 5e-4,
            tokens_per_sec: 1000.0,
            elapsed_time: 120.0,
            train_perplexity: 12.2,
            val_perplexity: 14.9,
        };

        assert_eq!(stats.iteration, 100);
        assert_eq!(stats.train_loss, 2.5);
        assert_eq!(stats.val_loss, 2.7);
        assert_eq!(stats.learning_rate, 5e-4);
        assert_eq!(stats.tokens_per_sec, 1000.0);
        assert_eq!(stats.elapsed_time, 120.0);
        assert_eq!(stats.train_perplexity, 12.2);
        assert_eq!(stats.val_perplexity, 14.9);
    }
}
