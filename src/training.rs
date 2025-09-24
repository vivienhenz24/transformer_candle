use candle_core::{Device, Result as CandleResult};
use candle_nn::{
    VarBuilder,
    VarMap,
    optim::{AdamW, Optimizer, ParamsAdamW},
};
use std::time::Instant;

use crate::gpt::{GPTLanguageModel, GPTConfig};
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
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            learning_rate: 6e-4,          // Learning rate from GPT paper
            batch_size: 64,               // batch size for training
            block_size: 256,              // Context length
            max_iters: 5000,              // Number of training steps
            eval_interval: 100,           // Evaluate every 100 steps
            eval_iters: 200,              // Average over 200 eval batches
            weight_decay: 1e-1,           // Weight decay for regularization
            beta1: 0.9,                   // AdamW beta1
            beta2: 0.95,                  // AdamW beta2 (slightly lower than default)
            eps: 1e-8,                    // Epsilon for numerical stability
            device: Device::Cpu,          // Default to CPU
            log_interval: 1,              // Log every iteration by default
            compile: false,               // Don't compile by default
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
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;
    
    // Evaluation mode - no gradients needed
    for _ in 0..config.eval_iters {
        // Get a batch of data
        let (inputs, targets) = tokenizer.get_batch(
            split,
            config.batch_size,
            config.block_size,
        )?;
        
        // Forward pass with targets to compute loss
        let (_logits, loss) = model.forward(&inputs, Some(&targets), false)?;
        
        if let Some(loss_tensor) = loss {
            let loss_value = loss_tensor.to_scalar::<f32>()?;
            total_loss += loss_value;
            num_batches += 1;
        }
    }
    
    if num_batches > 0 {
        Ok(total_loss / num_batches as f32)
    } else {
        Err(candle_core::Error::Msg("No valid batches for loss estimation".to_string()))
    }
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

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            weight_decay: config.weight_decay,
            eps: config.eps,
        },
    )?;
    
    // Timing and statistics
    let training_start = Instant::now();
    let mut last_log_time = training_start;
    let mut tokens_processed = 0;
    
    println!(" Training Progress:");
    println!("{:>6} | {:>10} | {:>10} | {:>8} | {:>10} | {:>8}", 
             "Iter", "Train Loss", "Val Loss", "LR", "Tok/sec", "Time");
    println!("{}", "-".repeat(70));
    
    // Main training loop
    for iter in 0..config.max_iters {
        // Get training batch
        let (inputs, targets) = tokenizer.get_batch(
            DataSplit::Train,
            config.batch_size,
            config.block_size,
        ).map_err(|e| candle_core::Error::Msg(format!("Failed to get training batch: {}", e)))?;
        
        // Forward pass with loss computation
        let (_logits, loss) = model.forward(&inputs, Some(&targets), true)
            .map_err(|e| candle_core::Error::Msg(format!("Forward pass failed: {}", e)))?;

        let loss_tensor = loss.ok_or_else(|| {
            candle_core::Error::Msg("No loss computed during training".to_string())
        })?;

        optimizer.backward_step(&loss_tensor)
            .map_err(|e| candle_core::Error::Msg(format!("Optimizer step failed: {}", e)))?;

        let train_loss = loss_tensor.to_scalar::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to extract loss value: {}", e)))?;
        
        // Update token count
        tokens_processed += config.batch_size * config.block_size;
        
        // Evaluate and log progress
        if iter % config.eval_interval == 0 || iter == config.max_iters - 1 {
            // Estimate losses on train and validation sets
            let train_loss_avg = estimate_loss(model, tokenizer, DataSplit::Train, config)
                .map_err(|e| candle_core::Error::Msg(format!("Train loss estimation failed: {}", e)))?;
            
            let val_loss_avg = estimate_loss(model, tokenizer, DataSplit::Val, config)
                .map_err(|e| candle_core::Error::Msg(format!("Validation loss estimation failed: {}", e)))?;
            
            // Calculate timing statistics
            let current_time = Instant::now();
            let elapsed_since_last = current_time.duration_since(last_log_time).as_secs_f32();
            let total_elapsed = current_time.duration_since(training_start).as_secs_f32();
            let tokens_per_sec = if elapsed_since_last > 0.0 {
                (config.batch_size * config.block_size * config.eval_interval) as f32 / elapsed_since_last
            } else {
                0.0
            };
            
            // Create training statistics
            let stats = TrainingStats {
                iteration: iter,
                train_loss: train_loss_avg,
                val_loss: val_loss_avg,
                learning_rate: config.learning_rate,
                tokens_per_sec,
                elapsed_time: total_elapsed,
            };
            
            training_stats.push(stats.clone());
            
            // Log progress
            if iter % config.log_interval == 0 {
                println!("{:6} | {:10.4} | {:10.4} | {:8.2e} | {:10.0} | {:8.1}s",
                         iter,
                         stats.train_loss,
                         stats.val_loss,
                         stats.learning_rate,
                         stats.tokens_per_sec,
                         stats.elapsed_time);
            }
            
            last_log_time = current_time;
        }
        
        // Handle training interruption gracefully
        if iter > 0 && iter % 1000 == 0 {
            println!("Checkpoint at iteration {}: Train loss = {:.4}", iter, train_loss);
        }
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
        block_size: 64,          // Smaller context for faster training
        n_embd: 64,              // Smaller embedding dimension
        n_head: 4,               // Fewer attention heads
        n_layer: 2,              // Fewer layers
        dropout_rate: 0.0,       // No dropout for small model
    }
}

/// Helper function to create a medium model configuration for serious training
pub fn create_medium_gpt_config(vocab_size: usize) -> GPTConfig {
    GPTConfig {
        vocab_size,
        block_size: 256,         // Standard context length
        n_embd: 384,             // Medium embedding dimension
        n_head: 6,               // 6 attention heads
        n_layer: 6,              // 6 transformer layers
        dropout_rate: 0.1,       // Standard dropout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device};
    
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
}
