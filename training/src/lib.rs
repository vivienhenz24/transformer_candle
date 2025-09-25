use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{
    optim::{AdamW, Optimizer, ParamsAdamW},
    VarBuilder, VarMap,
};
use cascade_core::{AdaptiveSamplingConfig, CascadeTransformer, CascadeTransformerConfig};
use std::io::{self, Write};
use transformer_tokenization::{AdvancedTokenizer, DataSplit};
use utils::prompts::build_prompt;

pub mod memory_efficient;
pub mod monitoring;
pub mod optimizers;
pub mod regularization;
pub mod runtime;
pub mod schedulers;

pub use optimizers::custom_adamw::CustomAdamW;
pub use runtime::{check_available_backends, setup_device};

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub block_size: usize,
    pub max_iters: usize,
    pub eval_interval: usize,
    pub eval_iters: usize,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub device: candle_core::Device,
    pub log_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 5e-4,
            batch_size: 48,
            block_size: 256,
            max_iters: 12000,
            eval_interval: 400,
            eval_iters: 200,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            device: candle_core::Device::Cpu,
            log_interval: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub iteration: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub tokens_per_sec: f32,
    pub elapsed_time: f32,
}

pub fn create_model(
    config: CascadeTransformerConfig,
    vb: VarBuilder,
) -> candle_core::Result<CascadeTransformer> {
    CascadeTransformer::new(config, vb)
}

pub fn estimate_loss(
    model: &CascadeTransformer,
    tokenizer: &AdvancedTokenizer,
    split: DataSplit,
    config: &TrainingConfig,
) -> candle_core::Result<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for _ in 0..config.eval_iters {
        let (inputs, targets) = tokenizer.get_batch(split, config.batch_size, config.block_size)?;
        let (_, loss) = model.forward(&inputs, Some(&targets), false)?;
        if let Some(loss) = loss {
            total += loss.to_scalar::<f32>()?;
            count += 1;
        }
    }
    if count == 0 {
        Err(candle_core::Error::Msg(
            "no batches for loss estimate".into(),
        ))
    } else {
        Ok(total / count as f32)
    }
}

pub fn train_model(
    model: &CascadeTransformer,
    tokenizer: &AdvancedTokenizer,
    varmap: &mut VarMap,
    config: &TrainingConfig,
) -> candle_core::Result<Vec<TrainingStats>> {
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

    let mut stats = Vec::new();
    let started = std::time::Instant::now();
    let mut last_log = started;
    let mut tokens_processed = 0usize;

    for iter in 0..config.max_iters {
        let (inputs, targets) =
            tokenizer.get_batch(DataSplit::Train, config.batch_size, config.block_size)?;
        let (_logits, loss) = model.forward(&inputs, Some(&targets), true)?;
        let loss = loss.ok_or_else(|| candle_core::Error::Msg("no loss".into()))?;
        optimizer.backward_step(&loss)?;
        let loss_value = loss.to_scalar::<f32>()?;
        tokens_processed += config.batch_size * config.block_size;

        if config.log_interval > 0 && (iter + 1) % config.log_interval == 0 {
            println!(
                "iter {:5}/{:5} | train_loss {:.4}",
                iter + 1,
                config.max_iters,
                loss_value
            );
            let _ = io::stdout().flush();
        }

        if iter % config.eval_interval == 0 || iter == config.max_iters - 1 {
            let train_loss = estimate_loss(model, tokenizer, DataSplit::Train, config)?;
            let val_loss = estimate_loss(model, tokenizer, DataSplit::Val, config)?;
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(started).as_secs_f32();
            let since_last = now.duration_since(last_log).as_secs_f32();
            let tokens_per_sec = if since_last > 0.0 {
                (config.batch_size * config.block_size * config.eval_interval) as f32 / since_last
            } else {
                0.0
            };
            stats.push(TrainingStats {
                iteration: iter,
                train_loss,
                val_loss,
                tokens_per_sec,
                elapsed_time: elapsed,
            });
            last_log = now;

            println!(
                "{:6} | {:10.4} | {:10.4} | {:10.0} | {:8.1}s",
                iter, train_loss, val_loss, tokens_per_sec, elapsed
            );

            if let Ok(sample) = generate_preview(model, tokenizer) {
                println!("--- preview ---\n{}\n------------", sample);
            }
        }
        if iter > 0 && iter % 1000 == 0 {
            println!("checkpoint iter {iter}: loss={loss_value:.4}");
        }
    }

    println!(
        "Training complete in {:.1}s (avg {:.0} tok/s)",
        started.elapsed().as_secs_f32(),
        tokens_processed as f32 / started.elapsed().as_secs_f32()
    );
    Ok(stats)
}

fn generate_preview(model: &CascadeTransformer, tokenizer: &AdvancedTokenizer) -> Result<String> {
    let prompt = build_prompt("ROMEO:");
    let prompt_tokens = tokenizer.encode(&prompt)?;
    if prompt_tokens.is_empty() {
        return Ok(String::new());
    }
    let prompt_tensor = Tensor::from_vec(
        prompt_tokens.clone(),
        (1, prompt_tokens.len()),
        tokenizer.device(),
    )?;
    let generated = model.generate(&prompt_tensor, AdaptiveSamplingConfig::default())?;
    let flat = generated.get(0)?;
    let indices = flat.to_vec1::<u32>()?;
    let indices: Vec<u32> = indices;
    if indices.len() > prompt_tokens.len() as usize {
        tokenizer.decode(&indices[prompt_tokens.len()..])
    } else {
        Ok(String::new())
    }
}
