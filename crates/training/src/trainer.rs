use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use candle_core::{
    backprop::GradStore,
    utils::{cuda_is_available, metal_is_available},
    DType, Device, IndexOp, Tensor,
};
use model::Model;
use pretraining_data::corpora::StreamingCorpus;
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
use std::cmp::Ordering;
use tokenizer::{
    build_from_artifacts as build_bbpe_tokenizer, ArtifactsCfg as BbpeArtifactsCfg,
    ByteLevelCfg as BbpeByteLevelCfg, Config as BbpeConfig, ModelCfg as BbpeModelCfg,
    PostCfg as BbpePostCfg,
};
use tokenizers::{models::ModelWrapper, pre_tokenizers::PreTokenizerWrapper, Tokenizer};

use crate::{
    checkpoint::{self, LoadOutcome, RngSnapshot, SaveRequest, TrainingProgressSnapshot},
    config::TokenizerConfig as TrainingTokenizerConfig,
    data::{BlockingDataLoader, DataBatch, StreamingTextDataLoader},
    logging::{Logger, LoggingSettings},
    loss::{CrossEntropyLoss, LossMetrics, LossOutput},
    metrics::{EvaluationMetrics, EvaluationSummary, TrainingMetrics},
    optimizer::{GradientScaler, OptimizerConfig, TrainerOptimizer, TrainerOptimizerOptions},
    scheduler::{LRScheduler, SchedulerConfig},
    TrainingConfig, TrainingError,
};

const PREVIEW_PROMPT_TOKENS: usize = 64;
const PREVIEW_NEW_TOKENS: usize = 200;
const PREVIEW_TEMPERATURE: f32 = 0.8;
const PREVIEW_TOP_K: usize = 40;
const PREVIEW_TOP_P: f32 = 0.9;
const PREVIEW_REPETITION_PENALTY: f32 = 1.1;

pub struct Trainer {
    config: TrainingConfig,
    device: Device,
    tokenizer: Arc<Tokenizer>,
    data_loader: BlockingDataLoader<StreamingTextDataLoader>,
    model: Model,
    optimizer: TrainerOptimizer,
    scheduler: Option<Box<dyn LRScheduler>>,
    loss: CrossEntropyLoss,
    gradient_scaler: GradientScaler,
    parameter_tensors: Vec<Tensor>,
    pad_token_id: Option<u32>,
    max_steps: Option<usize>,
    max_epochs: Option<usize>,
    log_every: usize,
    sequence_length: usize,
    progress: TrainingProgressSnapshot,
    rng_snapshot: RngSnapshot,
    checkpoint: Option<CheckpointSettings>,
    optimizer_steps: usize,
    metrics: TrainingMetrics,
    logger: Logger,
    evaluation: Option<EvaluationSettings>,
    best_eval: Option<f64>,
}

#[derive(Debug, Clone)]
struct CheckpointSettings {
    directory: PathBuf,
    every_n_steps: Option<usize>,
    every_n_epochs: Option<usize>,
    max_keep: Option<usize>,
}

#[derive(Debug, Clone)]
struct EvaluationSettings {
    every_n_steps: Option<usize>,
    every_n_epochs: Option<usize>,
    max_batches: Option<usize>,
    best: Option<BestCheckpointSettings>,
}

#[derive(Debug, Clone)]
struct BestCheckpointSettings {
    directory: PathBuf,
    max_keep: Option<usize>,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Result<Self, TrainingError> {
        config.validate()?;

        let resolved = config.resolve_model_hyperparameters()?;

        let cuda_available = cuda_is_available();
        let metal_available = metal_is_available();
        println!(
            "device detection: cuda_available={} metal_available={}",
            cuda_available, metal_available
        );

        let device = if metal_available {
            match Device::new_metal(0) {
                Ok(device) => {
                    println!("device: using Metal GPU #0");
                    device
                }
                Err(err) => {
                    eprintln!(
                        "failed to initialize metal device, falling back to CPU: {}",
                        err
                    );
                    Device::Cpu
                }
            }
        } else if cuda_available {
            match Device::cuda_if_available(0) {
                Ok(device) => {
                    println!("device: using CUDA GPU #0");
                    device
                }
                Err(err) => {
                    eprintln!("cuda reported available but initialization failed: {err}");
                    Device::Cpu
                }
            }
        } else {
            eprintln!("no GPU backend available; using CPU");
            Device::Cpu
        };

        println!(
            "device selected: is_cuda={} is_metal={} is_cpu={}",
            device.is_cuda(),
            device.is_metal(),
            device.is_cpu()
        );
        if let Err(err) = device.set_seed(config.runtime.seed) {
            eprintln!("warning: failed to seed device RNG: {}", err);
        }

        let tokenizer = Arc::new(load_tokenizer(&config.tokenizer)?);
        let pad_token_id = tokenizer.get_padding().map(|params| params.pad_id as u32);

        let sequence_length = resolved.max_position_embeddings;
        let corpus = StreamingCorpus::new(config.data.train_shards.clone()).map_err(|err| {
            TrainingError::initialization(format!("failed to open training corpus: {err}"))
        })?;

        let data_loader = StreamingTextDataLoader::new(
            corpus,
            Arc::clone(&tokenizer),
            device.clone(),
            sequence_length,
            config.data.batch_size,
            config.data.gradient_accumulation_steps,
            config.runtime.seed,
            config.data.shuffle_buffer_size,
        )?;
        let data_loader = BlockingDataLoader::new(data_loader);
        println!(
            "[training crate] data loader ready (sequence_length={} batch_size={} grad_accum={})",
            sequence_length, config.data.batch_size, config.data.gradient_accumulation_steps
        );

        let mut model_config = resolved.model.clone();
        model_config.device = device.clone();
        let model = Model::new(model_config).map_err(to_runtime_error)?;
        println!("[training crate] model crate returned initialized model instance");

        let optimizer_config = OptimizerConfig::try_from(&config.optimizer)?;
        let named_parameters = model.parameters();
        if named_parameters.is_empty() {
            return Err(TrainingError::initialization(
                "model produced no trainable parameters",
            ));
        }
        println!(
            "[training crate] optimizer will track {} tensor(s)",
            named_parameters.len()
        );

        let parameter_tensors: Vec<Tensor> = named_parameters
            .iter()
            .map(|(_, var)| var.as_tensor().clone())
            .collect();

        let optimizer = TrainerOptimizer::new(
            named_parameters,
            optimizer_config,
            TrainerOptimizerOptions::default(),
        )?;

        let max_steps = config.scheduler.total_steps;
        let max_epochs = config.scheduler.total_epochs;
        let log_every = config.runtime.log_every_n_steps.max(1);

        let scheduler = if let Some(total_steps) = config.scheduler.total_steps {
            let scheduler_cfg = SchedulerConfig::from_training_config(
                &config.scheduler,
                optimizer.learning_rate(),
                total_steps,
            )?;
            Some(scheduler_cfg.build()?)
        } else {
            None
        };

        let mut loss = CrossEntropyLoss::new();
        loss = loss.with_ignore_index(pad_token_id);

        let gradient_scaler = GradientScaler::new(resolved.precision);

        let checkpoint = config
            .runtime
            .checkpoint
            .as_ref()
            .map(|cfg| CheckpointSettings {
                directory: cfg.directory.clone(),
                every_n_steps: cfg.every_n_steps,
                every_n_epochs: cfg.every_n_epochs,
                max_keep: cfg.max_keep,
            });

        let evaluation = build_evaluation_settings(&config)?;

        let logging_settings = LoggingSettings::from_config(
            config.runtime.logging.enable_stdout,
            config.runtime.logging.tensorboard.clone(),
            config.runtime.logging.tensorboard_flush_every_n,
        );
        let logger = Logger::new(logging_settings)?;

        let progress = TrainingProgressSnapshot {
            optimizer_step: 0,
            global_step: 0,
            epoch: 0,
            micro_batch_index: 0,
            micro_batches_per_step: config.data.gradient_accumulation_steps.max(1),
        };

        let rng_snapshot = RngSnapshot {
            master_seed: config.runtime.seed,
        };

        Ok(Self {
            config,
            device,
            tokenizer,
            data_loader,
            model,
            optimizer,
            scheduler,
            loss,
            gradient_scaler,
            parameter_tensors,
            pad_token_id,
            max_steps,
            max_epochs,
            log_every,
            sequence_length,
            progress,
            rng_snapshot,
            checkpoint,
            optimizer_steps: 0,
            metrics: TrainingMetrics::new(),
            logger,
            evaluation,
            best_eval: None,
        })
    }

    pub fn resume_from_latest(
        &mut self,
    ) -> Result<Option<checkpoint::CheckpointDescriptor>, TrainingError> {
        let Some(settings) = &self.checkpoint else {
            return Ok(None);
        };
        let Some(descriptor) = checkpoint::latest_checkpoint(&settings.directory)? else {
            return Ok(None);
        };
        println!(
            "resuming from checkpoint {} (manifest step {})",
            descriptor.directory.display(),
            descriptor.manifest.progress.optimizer_step
        );
        let outcome = checkpoint::load_checkpoint(&descriptor.directory)?;
        self.apply_checkpoint(outcome)?;
        Ok(Some(descriptor))
    }

    pub fn resume_from_path(
        &mut self,
        directory: &Path,
    ) -> Result<checkpoint::CheckpointDescriptor, TrainingError> {
        let outcome = checkpoint::load_checkpoint(directory)?;
        let manifest = outcome.manifest.clone();
        self.apply_checkpoint(outcome)?;
        Ok(checkpoint::CheckpointDescriptor {
            directory: directory.to_path_buf(),
            manifest,
        })
    }

    pub fn evaluate(
        &mut self,
        max_batches: Option<usize>,
    ) -> Result<EvaluationSummary, TrainingError> {
        self.evaluate_internal(max_batches)
    }

    pub fn train(&mut self) -> Result<(), TrainingError> {
        self.train_with_shutdown(|| false)
    }

    pub fn train_with_shutdown<F>(&mut self, mut should_stop: F) -> Result<(), TrainingError>
    where
        F: FnMut() -> bool,
    {
        self.model.set_training(true);
        let mut accumulated_grads: Option<GradStore> = None;
        let mut step_metrics = StepMetrics::default();
        let mut optimizer_steps: usize = self.optimizer_steps;

        println!(
            "starting training on {:?} (vocab={}, pad_id={:?})",
            self.device,
            self.tokenizer.get_vocab_size(true),
            self.pad_token_id
        );

        loop {
            if should_stop() {
                break;
            }
            if let Some(limit) = self.max_steps {
                if optimizer_steps >= limit {
                    break;
                }
            }

            let maybe_batch = self.data_loader.next_batch()?;
            let Some(batch) = maybe_batch else {
                break;
            };

            self.progress.micro_batches_per_step = batch.micro_batches_per_step;
            self.progress.micro_batch_index = batch.micro_batch_index;

            if let Some(epoch_limit) = self.max_epochs {
                if batch.epoch >= epoch_limit {
                    break;
                }
            }

            let dims = batch.input_ids.dims();
            if dims.len() < 2 {
                return Err(TrainingError::runtime(
                    "training batch requires [batch, seq] token tensor",
                ));
            }
            let seq_len = dims[1];
            if seq_len < 2 {
                continue;
            }

            let inputs = batch
                .input_ids
                .narrow(1, 0, seq_len - 1)
                .map_err(to_runtime_error)?;
            let targets = batch
                .input_ids
                .narrow(1, 1, seq_len - 1)
                .map_err(to_runtime_error)?;

            let logits = self.model.forward(&inputs).map_err(to_runtime_error)?;

            let LossOutput { loss, metrics } = self.loss.compute(&logits, &targets)?;
            step_metrics.accumulate(&metrics);

            let accumulation = batch.micro_batches_per_step.max(1) as f64;
            let normalized_loss = loss
                .affine(1.0 / accumulation, 0.0)
                .map_err(to_runtime_error)?;
            let scaled_loss = self.gradient_scaler.scale(&normalized_loss)?;

            let micro_grads = scaled_loss.backward().map_err(to_runtime_error)?;

            if let Some(existing) = accumulated_grads.as_mut() {
                self.merge_gradient_store(existing, micro_grads)?;
            } else {
                accumulated_grads = Some(micro_grads);
            }

            let is_step_boundary =
                batch.micro_batch_index + 1 == batch.micro_batches_per_step.max(1);
            if !is_step_boundary {
                continue;
            }

            let Some(grads) = accumulated_grads.as_mut() else {
                continue;
            };

            let (mut found_inf, grad_norm) = self.unscale_gradients(grads)?;

            if let Some(avg_loss) = step_metrics.average_loss() {
                if !avg_loss.is_finite() {
                    found_inf = true;
                }
            }

            self.gradient_scaler.update(found_inf);

            if found_inf {
                self.optimizer.zero_grad(grads);
                accumulated_grads = None;
                step_metrics.reset();
                continue;
            }

            // Apply gradient clipping if configured
            if let Some(max_norm) = self.config.optimizer.max_grad_norm {
                self.clip_gradients(grads, max_norm as f64)?;
            }

            let lr = if let Some(scheduler) = self.scheduler.as_mut() {
                let lr = scheduler.step();
                self.optimizer.set_learning_rate(lr);
                lr
            } else {
                self.optimizer.learning_rate()
            };

            self.optimizer.step(grads)?;
            optimizer_steps += 1;

            self.record_progress(&batch, optimizer_steps);
            self.optimizer_steps = optimizer_steps;

            if let Some(avg_loss) = step_metrics.average_loss() {
                let tokens = step_metrics.tokens() as u64;
                let snapshot = self.metrics.record_step(tokens, avg_loss, grad_norm);
                if optimizer_steps % self.log_every == 0 || optimizer_steps <= 5 {
                    self.logger
                        .log_training_step(optimizer_steps, lr, &snapshot);
                }
                if optimizer_steps % 5 == 0 {
                    println!(
                        "[progress] step {:>6} | seq {:>4} | tokens {:>6} | loss {:>8.4} | grad {:>7.4} | lr {:>9.3e}",
                        optimizer_steps,
                        seq_len,
                        tokens,
                        snapshot.step_loss,
                        grad_norm,
                        lr
                    );
                }
            }

            self.maybe_checkpoint(&batch, optimizer_steps)?;
            self.maybe_evaluate(&batch, optimizer_steps)?;

            accumulated_grads = None;
            step_metrics.reset();
        }

        self.logger.flush();
        Ok(())
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    fn merge_gradient_store(
        &self,
        accum: &mut GradStore,
        mut new_grads: GradStore,
    ) -> Result<(), TrainingError> {
        for tensor in &self.parameter_tensors {
            if let Some(grad) = new_grads.remove(tensor) {
                let combined = if let Some(existing) = accum.remove(tensor) {
                    existing.add(&grad).map_err(to_runtime_error)?
                } else {
                    grad
                };
                accum.insert(tensor, combined);
            }
        }

        // Silently ignore unknown tensors - they are cached tensors or RNG states
        let _unknown_count = new_grads.get_ids().count();
        // Note: Ignoring gradients for cached tensors (attention masks, position embeddings, etc.)

        Ok(())
    }

    fn unscale_gradients(&mut self, grads: &mut GradStore) -> Result<(bool, f64), TrainingError> {
        let mut found_inf = false;
        let mut sum_squares = 0.0f64;
        for tensor in &self.parameter_tensors {
            if let Some(grad) = grads.remove(tensor) {
                let unscaled = self.gradient_scaler.unscale(&grad)?;
                if !found_inf && contains_non_finite(&unscaled)? {
                    found_inf = true;
                }
                let sq = unscaled
                    .to_dtype(DType::F32)
                    .map_err(to_runtime_error)?
                    .sqr()
                    .map_err(to_runtime_error)?
                    .sum_all()
                    .map_err(to_runtime_error)?
                    .to_vec0::<f32>()
                    .map_err(to_runtime_error)? as f64;
                sum_squares += sq;
                grads.insert(tensor, unscaled);
            }
        }
        let norm = if sum_squares > 0.0 {
            sum_squares.sqrt()
        } else {
            0.0
        };
        Ok((found_inf, norm))
    }

    fn clip_gradients(&self, grads: &mut GradStore, max_norm: f64) -> Result<(), TrainingError> {
        // Calculate current gradient norm
        let mut sum_squares = 0.0f64;
        for tensor in &self.parameter_tensors {
            if let Some(grad) = grads.get(tensor) {
                let sq = grad
                    .to_dtype(DType::F32)
                    .map_err(to_runtime_error)?
                    .sqr()
                    .map_err(to_runtime_error)?
                    .sum_all()
                    .map_err(to_runtime_error)?
                    .to_vec0::<f32>()
                    .map_err(to_runtime_error)? as f64;
                sum_squares += sq;
            }
        }

        let total_norm = sum_squares.sqrt();

        // Only clip if norm exceeds max_norm
        if total_norm > max_norm {
            let clip_coef = max_norm / (total_norm + 1e-6);

            // Scale all gradients by clip_coef
            for tensor in &self.parameter_tensors {
                if let Some(grad) = grads.remove(tensor) {
                    let clipped = (grad * clip_coef).map_err(to_runtime_error)?;
                    grads.insert(tensor, clipped);
                }
            }
        }

        Ok(())
    }

    fn record_progress(&mut self, batch: &DataBatch, optimizer_steps: usize) {
        self.progress.optimizer_step = optimizer_steps;
        self.progress.global_step = batch.global_step.wrapping_add(1);
        self.progress.epoch = batch.epoch;
        self.progress.micro_batch_index = 0;
        self.progress.micro_batches_per_step = batch.micro_batches_per_step;
    }

    fn maybe_checkpoint(
        &mut self,
        batch: &DataBatch,
        optimizer_steps: usize,
    ) -> Result<(), TrainingError> {
        let Some(settings) = &self.checkpoint else {
            return Ok(());
        };

        let step_trigger = settings
            .every_n_steps
            .map_or(false, |n| n > 0 && optimizer_steps % n == 0);
        let epoch_trigger = settings
            .every_n_epochs
            .map_or(false, |n| n > 0 && (batch.epoch + 1) % n == 0);

        if !(step_trigger || epoch_trigger) {
            return Ok(());
        }

        let request = SaveRequest {
            base_dir: &settings.directory,
            config: &self.config,
            model: &self.model,
            optimizer: &self.optimizer,
            scheduler: self.scheduler.as_deref(),
            scaler: &self.gradient_scaler,
            progress: self.progress.clone(),
            rng: self.rng_snapshot.clone(),
            max_keep: settings.max_keep,
        };

        let descriptor = checkpoint::save_checkpoint(request)?;
        println!(
            "[training crate] checkpoint saved at step {} -> {}",
            optimizer_steps,
            descriptor.directory.display()
        );
        if let Some(sample) = self.sample_model_output(batch) {
            println!("[training crate] model sample: {}", sample);
        }

        Ok(())
    }

    fn maybe_evaluate(
        &mut self,
        batch: &DataBatch,
        optimizer_steps: usize,
    ) -> Result<(), TrainingError> {
        let Some(settings_ref) = &self.evaluation else {
            return Ok(());
        };
        let settings = settings_ref.clone();

        let step_trigger = settings
            .every_n_steps
            .map_or(false, |n| n > 0 && optimizer_steps % n == 0);
        let epoch_trigger = settings
            .every_n_epochs
            .map_or(false, |n| n > 0 && (batch.epoch + 1) % n == 0);

        if !(step_trigger || epoch_trigger) {
            return Ok(());
        }

        if self.config.data.validation_shards.is_empty() {
            return Ok(());
        }

        let summary = self.evaluate_internal(settings.max_batches)?;
        self.logger.log_evaluation(optimizer_steps, &summary);
        self.logger.flush();
        println!(
            "[training crate] evaluation summary at step {}: loss={:.4} ppl={:.4} tokens={}",
            optimizer_steps, summary.average_loss, summary.perplexity, summary.tokens
        );

        if let Some(best) = settings.best {
            let improved = match self.best_eval {
                Some(current_best) => summary.perplexity < current_best,
                None => true,
            };
            if improved {
                self.best_eval = Some(summary.perplexity);
                let request = SaveRequest {
                    base_dir: &best.directory,
                    config: &self.config,
                    model: &self.model,
                    optimizer: &self.optimizer,
                    scheduler: self.scheduler.as_deref(),
                    scaler: &self.gradient_scaler,
                    progress: self.progress.clone(),
                    rng: self.rng_snapshot.clone(),
                    max_keep: best.max_keep,
                };
                let descriptor = checkpoint::save_checkpoint(request)?;
                println!(
                    "[training crate] new best checkpoint saved at {} (ppl={:.4})",
                    descriptor.directory.display(),
                    summary.perplexity
                );
            }
        }

        Ok(())
    }

    fn evaluate_internal(
        &mut self,
        max_batches: Option<usize>,
    ) -> Result<EvaluationSummary, TrainingError> {
        let was_training = self.model.is_training();
        self.model.set_training(false);

        let result = (|| {
            let corpus = StreamingCorpus::new(self.config.data.validation_shards.clone()).map_err(
                |err| TrainingError::runtime(format!("failed to open validation corpus: {err}")),
            )?;

            let mut loader = BlockingDataLoader::new(StreamingTextDataLoader::new(
                corpus,
                Arc::clone(&self.tokenizer),
                self.device.clone(),
                self.sequence_length,
                self.config.data.batch_size,
                1,
                self.config.runtime.seed,
                Some(0),
            )?);

            let mut metrics = EvaluationMetrics::default();
            let mut processed_batches = 0usize;

            loop {
                if let Some(limit) = max_batches {
                    if processed_batches >= limit {
                        break;
                    }
                }
                let Some(batch) = loader.next_batch()? else {
                    break;
                };
                processed_batches += 1;

                let dims = batch.input_ids.dims();
                if dims.len() < 2 {
                    continue;
                }
                let seq_len = dims[1];
                if seq_len < 2 {
                    continue;
                }

                let inputs = batch
                    .input_ids
                    .narrow(1, 0, seq_len - 1)
                    .map_err(to_runtime_error)?;
                let targets = batch
                    .input_ids
                    .narrow(1, 1, seq_len - 1)
                    .map_err(to_runtime_error)?;

                let logits = self.model.forward(&inputs).map_err(to_runtime_error)?;

                let loss = self.loss.compute(&logits, &targets)?;
                let avg_loss = loss.metrics.average_loss() as f64;
                let tokens = loss.metrics.total_tokens() as u64;
                let correct = loss.metrics.correct_tokens() as u64;

                metrics.update(avg_loss, tokens, correct);
            }

            metrics
                .finalize()
                .ok_or_else(|| TrainingError::runtime("evaluation produced no tokens"))
        })();

        self.model.set_training(was_training);
        result
    }

    fn apply_checkpoint(&mut self, outcome: LoadOutcome) -> Result<(), TrainingError> {
        let LoadOutcome {
            manifest,
            optimizer_state,
            scheduler_state,
            scaler_state,
            model_weights_path,
        } = outcome;

        checkpoint::apply_model_weights(&self.model, &model_weights_path)?;
        self.optimizer.load_state(optimizer_state)?;

        if let Some(state) = scheduler_state {
            match self.scheduler.as_mut() {
                Some(scheduler) => {
                    scheduler.load_snapshot(&state)?;
                    self.optimizer.set_learning_rate(scheduler.learning_rate());
                }
                None => {
                    return Err(TrainingError::runtime(
                        "checkpoint includes scheduler state but trainer has no scheduler",
                    ));
                }
            }
        }

        self.gradient_scaler.load_state(scaler_state);

        self.fast_forward_data_loader(&manifest.progress)?;

        self.progress = manifest.progress;
        self.optimizer_steps = self.progress.optimizer_step;
        self.rng_snapshot = manifest.rng;

        Ok(())
    }

    fn fast_forward_data_loader(
        &mut self,
        progress: &TrainingProgressSnapshot,
    ) -> Result<(), TrainingError> {
        let micro_batches_per_step = progress.micro_batches_per_step.max(1);
        let total_micro_batches = progress
            .global_step
            .saturating_mul(micro_batches_per_step)
            .saturating_add(progress.micro_batch_index);

        if total_micro_batches == 0 {
            return Ok(());
        }

        println!(
            "fast-forwarding data loader: target global step {} ({} micro-batches)",
            progress.global_step, total_micro_batches
        );

        let mut remaining = total_micro_batches;
        let log_every = (micro_batches_per_step * 20).max(1000);

        while remaining > 0 {
            match self.data_loader.next_batch()? {
                Some(_) => {
                    remaining -= 1;
                    let processed = total_micro_batches - remaining;
                    if processed % log_every == 0 || remaining == 0 {
                        let percent = (processed as f64 / total_micro_batches as f64) * 100.0;
                        println!(
                            "fast-forwarding data loader: {}/{} micro-batches ({:.1}%)",
                            processed, total_micro_batches, percent
                        );
                    }
                }
                None => {
                    return Err(TrainingError::runtime(
                        "checkpoint references progress beyond available training data",
                    ));
                }
            }
        }

        Ok(())
    }

    fn sample_model_output(&self, batch: &DataBatch) -> Option<String> {
        let was_training = self.model.is_training();
        self.model.set_training(false);
        let result = self.sample_model_output_impl(batch);
        self.model.set_training(was_training);
        result
    }

    fn sample_model_output_impl(&self, batch: &DataBatch) -> Option<String> {
        let first_seq = batch.input_ids.narrow(0, 0, 1).ok()?;
        let ids_2d = first_seq.to_vec2::<u32>().ok()?;
        let tokens = ids_2d.into_iter().next()?;
        if tokens.is_empty() {
            return None;
        }

        let prompt_len = tokens.len().min(PREVIEW_PROMPT_TOKENS.max(1));
        let mut prompt_ids: Vec<u32> = tokens.iter().copied().take(prompt_len).collect();
        if prompt_ids.is_empty() {
            prompt_ids.push(*tokens.first()?);
        }

        let mut context = prompt_ids.clone();
        let mut generated: Vec<u32> = Vec::new();

        let mut rng = StdRng::seed_from_u64(
            self.config.runtime.seed
                ^ (self.progress.optimizer_step as u64)
                ^ (batch.global_step as u64),
        );

        for _ in 0..PREVIEW_NEW_TOKENS {
            if context.is_empty() {
                break;
            }

            // Convert u32 to i64 to avoid negative token ID issues
            let context_i64: Vec<i64> = context.iter().map(|&t| t as i64).collect();
            let input = match Tensor::from_slice(&context_i64, (1, context_i64.len()), &self.device)
            {
                Ok(t) => t,
                Err(_) => break,
            };
            let logits = match self.model.forward(&input) {
                Ok(t) => t,
                Err(_) => break,
            };
            let dims = logits.dims();
            if dims.len() != 3 {
                break;
            }
            let seq_len = dims[1];
            if seq_len == 0 {
                break;
            }

            let last = match logits.i((0, seq_len - 1)) {
                Ok(t) => t,
                Err(_) => break,
            };
            let scores = match last.to_vec1::<f32>() {
                Ok(v) => v,
                Err(_) => break,
            };
            if scores.is_empty() {
                break;
            }

            let next_id = match sample_next_token(
                &scores,
                PREVIEW_TEMPERATURE,
                PREVIEW_TOP_K,
                PREVIEW_TOP_P,
                PREVIEW_REPETITION_PENALTY,
                &context,
                &mut rng,
            ) {
                Some(id) => id,
                None => break,
            };

            if self.pad_token_id == Some(next_id) {
                break;
            }
            generated.push(next_id);
            context.push(next_id);
            if context.len() >= self.sequence_length {
                break;
            }
        }

        let prompt_text = self.tokenizer.decode(&prompt_ids, true).ok()?;
        let generated_text = if generated.is_empty() {
            "<no new tokens>".to_string()
        } else {
            self.tokenizer.decode(&generated, true).ok()?
        };

        Some(format!(
            "prompt=\"{}\" generated=\"{}\"",
            prompt_text, generated_text
        ))
    }
}

fn sample_next_token<R: rand::Rng + ?Sized>(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    history: &[u32],
    rng: &mut R,
) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    let mut adjusted = logits.to_vec();

    if repetition_penalty > 1.0 {
        for &token in history.iter() {
            let idx = token as usize;
            if idx >= adjusted.len() {
                continue;
            }
            let logit = &mut adjusted[idx];
            if *logit > 0.0 {
                *logit /= repetition_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
    }

    let temp = temperature.max(1e-4);
    let inv_temp = 1.0 / temp;
    for val in adjusted.iter_mut() {
        *val *= inv_temp;
    }

    let max_val = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = adjusted
        .iter()
        .map(|logit| (logit - max_val).exp())
        .collect();

    if !probs.iter().all(|p| p.is_finite()) {
        return None;
    }

    // Top-k filtering
    if top_k > 0 && top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices
            .sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal));
        for &idx in indices.iter().skip(top_k) {
            probs[idx] = 0.0;
        }
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let mut pairs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(idx, prob)| (idx, *prob))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut keep = vec![false; probs.len()];
        for (idx, prob) in pairs {
            cumulative += prob;
            keep[idx] = true;
            if cumulative >= top_p {
                break;
            }
        }
        for (idx, keep_flag) in keep.iter().enumerate() {
            if !*keep_flag {
                probs[idx] = 0.0;
            }
        }
    }

    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        if let Some((idx, _)) = adjusted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        {
            return Some(idx as u32);
        }
        return None;
    }

    for prob in probs.iter_mut() {
        *prob /= sum;
    }

    let dist = WeightedIndex::new(&probs).ok()?;
    let sampled = dist.sample(rng);
    Some(sampled as u32)
}

fn load_tokenizer(cfg: &TrainingTokenizerConfig) -> Result<Tokenizer, TrainingError> {
    let bbpe_cfg = build_tokenizer_config(cfg)?;
    build_bbpe_tokenizer(&bbpe_cfg).map_err(|err| {
        TrainingError::initialization(format!("failed to build tokenizer artifacts: {err}"))
    })
}

fn contains_non_finite(tensor: &Tensor) -> Result<bool, TrainingError> {
    if tensor.elem_count() == 0 {
        return Ok(false);
    }
    let sum = tensor
        .to_dtype(candle_core::DType::F32)
        .map_err(to_runtime_error)?
        .sqr()
        .map_err(to_runtime_error)?
        .sum_all()
        .map_err(to_runtime_error)?
        .to_vec0::<f32>()
        .map_err(to_runtime_error)?;
    Ok(!sum.is_finite())
}

fn build_tokenizer_config(cfg: &TrainingTokenizerConfig) -> Result<BbpeConfig, TrainingError> {
    let artifact_dir = infer_artifact_dir(cfg)?;

    let tokenizer_json = cfg.tokenizer_json.clone();
    let vocab = cfg.vocab.clone();
    let merges = cfg.merges.clone();

    let metadata = if let Some(json_path) = tokenizer_json.as_ref() {
        gather_metadata_from_json(json_path, cfg.special_tokens.as_deref())?
    } else {
        gather_metadata_from_vocab(cfg, cfg.special_tokens.as_deref())?
    };

    let artifacts = BbpeArtifactsCfg {
        dir: artifact_dir,
        tokenizer_json,
        vocab_json: vocab,
        merges_txt: merges,
        manifest: None,
    };

    Ok(BbpeConfig {
        model: BbpeModelCfg {
            vocab_size: metadata.vocab_size,
            min_frequency: metadata.min_frequency,
            dropout: metadata.dropout,
            special_tokens: metadata.special_tokens,
            byte_fallback_on_decode: metadata.byte_fallback,
        },
        pretokenizer: metadata.byte_level,
        postprocessor: metadata.postprocessor,
        artifacts,
        training: None,
    })
}

struct TokenizerMetadata {
    vocab_size: usize,
    min_frequency: u32,
    dropout: Option<f32>,
    special_tokens: Vec<String>,
    byte_fallback: bool,
    byte_level: BbpeByteLevelCfg,
    postprocessor: Option<BbpePostCfg>,
}

fn gather_metadata_from_json(
    tokenizer_path: &Path,
    special_tokens_path: Option<&Path>,
) -> Result<TokenizerMetadata, TrainingError> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|err| {
        TrainingError::initialization(format!(
            "failed to load tokenizer json {}: {}",
            tokenizer_path.display(),
            err
        ))
    })?;

    let vocab_size = tokenizer.get_vocab_size(false);
    let (dropout, byte_fallback) = match tokenizer.get_model() {
        ModelWrapper::BPE(bpe) => (bpe.dropout, bpe.byte_fallback),
        other => {
            return Err(TrainingError::initialization(format!(
                "unsupported tokenizer model type: {:?}",
                other
            )))
        }
    };

    let byte_level = match tokenizer.get_pre_tokenizer() {
        Some(PreTokenizerWrapper::ByteLevel(byte_level)) => BbpeByteLevelCfg {
            add_prefix_space: byte_level.add_prefix_space,
            trim_offsets: byte_level.trim_offsets,
            use_regex: byte_level.use_regex,
        },
        Some(other) => {
            return Err(TrainingError::initialization(format!(
                "training expects byte-level pretokenizer; found {:?}",
                other
            )))
        }
        None => {
            return Err(TrainingError::initialization(
                "tokenizer is missing a pretokenizer configuration",
            ))
        }
    };

    let mut special_tokens = if let Some(path) = special_tokens_path {
        read_special_tokens(path)?
    } else {
        collect_special_tokens(&tokenizer)
    };

    if special_tokens.is_empty() {
        special_tokens = collect_special_tokens(&tokenizer);
    }

    if special_tokens.is_empty() {
        return Err(TrainingError::initialization(
            "no special tokens provided or discoverable for tokenizer",
        ));
    }

    validate_special_tokens(&tokenizer, &special_tokens)?;

    Ok(TokenizerMetadata {
        vocab_size,
        min_frequency: 1,
        dropout,
        special_tokens,
        byte_fallback,
        byte_level,
        postprocessor: None,
    })
}

fn gather_metadata_from_vocab(
    cfg: &TrainingTokenizerConfig,
    special_tokens_path: Option<&Path>,
) -> Result<TokenizerMetadata, TrainingError> {
    let vocab_path = cfg.vocab.as_ref().ok_or_else(|| {
        TrainingError::initialization(
            "tokenizer configuration missing vocab path when tokenizer_json is absent",
        )
    })?;
    let merges_path = cfg.merges.as_ref().ok_or_else(|| {
        TrainingError::initialization(
            "tokenizer configuration missing merges path when tokenizer_json is absent",
        )
    })?;

    let (vocab_map, _merges) =
        tokenizers::models::bpe::BPE::read_file(path_str(vocab_path)?, path_str(merges_path)?)
            .map_err(|err| {
                TrainingError::initialization(format!(
                    "failed to read vocab/merges artifacts ({} / {}): {}",
                    vocab_path.display(),
                    merges_path.display(),
                    err
                ))
            })?;

    let special_tokens = special_tokens_path
        .map(read_special_tokens)
        .transpose()?
        .unwrap_or_default();

    Ok(TokenizerMetadata {
        vocab_size: vocab_map.len(),
        min_frequency: 1,
        dropout: None,
        special_tokens,
        byte_fallback: false,
        byte_level: BbpeByteLevelCfg {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
        postprocessor: None,
    })
}

fn infer_artifact_dir(cfg: &TrainingTokenizerConfig) -> Result<PathBuf, TrainingError> {
    let candidates = [
        cfg.tokenizer_json.as_ref(),
        cfg.vocab.as_ref(),
        cfg.merges.as_ref(),
    ];

    for candidate in candidates.into_iter().flatten() {
        if let Some(parent) = candidate.parent() {
            let dir = if parent.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                parent.to_path_buf()
            };
            if dir.is_dir() {
                return Ok(dir);
            }
        }
    }

    Err(TrainingError::initialization(
        "unable to infer tokenizer artifact directory; ensure tokenizer paths are set",
    ))
}

fn read_special_tokens(path: &Path) -> Result<Vec<String>, TrainingError> {
    let contents = fs::read_to_string(path).map_err(|err| {
        TrainingError::initialization(format!(
            "failed to read special tokens file {}: {}",
            path.display(),
            err
        ))
    })?;

    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }

    if let Ok(json_tokens) = serde_json::from_str::<Vec<String>>(&contents) {
        return Ok(deduplicate_tokens(json_tokens));
    }

    let tokens: Vec<String> = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect();

    Ok(deduplicate_tokens(tokens))
}

fn collect_special_tokens(tokenizer: &Tokenizer) -> Vec<String> {
    let added_vocab = tokenizer.get_added_vocabulary();
    let mut entries: Vec<(u32, String)> = added_vocab
        .get_vocab()
        .iter()
        .filter_map(|(token, id)| {
            if added_vocab.is_special_token(token) {
                Some((*id, token.clone()))
            } else {
                None
            }
        })
        .collect();
    entries.sort_by_key(|(id, _)| *id);
    deduplicate_tokens(entries.into_iter().map(|(_, token)| token).collect())
}

fn validate_special_tokens(tokenizer: &Tokenizer, tokens: &[String]) -> Result<(), TrainingError> {
    let mut missing = Vec::new();
    for token in tokens {
        if tokenizer.token_to_id(token).is_none() {
            missing.push(token.clone());
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(TrainingError::initialization(format!(
            "special tokens not present in tokenizer vocab: {}",
            missing.join(", ")
        )))
    }
}

fn path_str(path: &Path) -> Result<&str, TrainingError> {
    path.to_str().ok_or_else(|| {
        TrainingError::initialization(format!("path contains invalid UTF-8: {}", path.display()))
    })
}

fn deduplicate_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ordered = Vec::with_capacity(tokens.len());
    for token in tokens {
        if seen.insert(token.clone()) {
            ordered.push(token);
        }
    }
    ordered
}

fn to_runtime_error(err: candle_core::Error) -> TrainingError {
    TrainingError::runtime(err.to_string())
}

fn build_evaluation_settings(
    config: &TrainingConfig,
) -> Result<Option<EvaluationSettings>, TrainingError> {
    let eval_cfg = &config.runtime.evaluation;
    if eval_cfg.every_n_steps.is_none() && eval_cfg.every_n_epochs.is_none() {
        return Ok(None);
    }

    if config.data.validation_shards.is_empty() {
        return Err(TrainingError::initialization(
            "evaluation is configured but data.validation_shards is empty",
        ));
    }

    let best = eval_cfg.best.as_ref().map(|cfg| BestCheckpointSettings {
        directory: cfg.directory.clone(),
        max_keep: cfg.max_keep,
    });

    Ok(Some(EvaluationSettings {
        every_n_steps: eval_cfg.every_n_steps,
        every_n_epochs: eval_cfg.every_n_epochs,
        max_batches: eval_cfg.max_batches,
        best,
    }))
}

#[derive(Default)]
struct StepMetrics {
    loss_sum: f64,
    tokens: usize,
    correct: usize,
}

impl StepMetrics {
    fn accumulate(&mut self, metrics: &LossMetrics) {
        // Removed debug print for cleaner output
        self.loss_sum += metrics.average_loss() as f64 * metrics.total_tokens() as f64;
        self.tokens += metrics.total_tokens();
        self.correct += metrics.correct_tokens();
    }

    fn average_loss(&self) -> Option<f64> {
        if self.tokens == 0 {
            None
        } else {
            Some(self.loss_sum / self.tokens as f64)
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn tokens(&self) -> usize {
        self.tokens
    }
}
