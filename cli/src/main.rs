use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarMap;
use cascade_core::{
    CascadeTransformer, CascadeTransformerBuilder, CreativeMode, CreativePalette,
    ProgressiveGenerationConfig, ProgressiveRefiner,
};
use cascade_training::{check_available_backends, setup_device, train_model, TrainingConfig};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use transformer_tokenization::{AdvancedTokenizer, TokenizerConfig};
use utils::{prompts::build_prompt, wiki::preprocess_wikipedia_dump};

const TOKENIZER_DEFAULT_LIMIT_MIB: u64 = 512;

#[derive(Debug, Clone, Copy)]
enum TrainingPreset {
    Light,
    Balanced,
    Max,
}

impl TrainingPreset {
    fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "light" | "small" => Some(TrainingPreset::Light),
            "balanced" | "medium" | "default" => Some(TrainingPreset::Balanced),
            "max" | "heavy" | "large" => Some(TrainingPreset::Max),
            _ => None,
        }
    }
}

fn parse_training_preset() -> TrainingPreset {
    let mut preset = TrainingPreset::Balanced;
    if let Ok(env_value) = env::var("TRAINING_PRESET") {
        if let Some(parsed) = TrainingPreset::from_str(&env_value) {
            preset = parsed;
        }
    }
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--preset=") {
            if let Some(parsed) = TrainingPreset::from_str(value) {
                preset = parsed;
            }
        } else if arg == "--preset" {
            if let Some(value) = args.next() {
                if let Some(parsed) = TrainingPreset::from_str(&value) {
                    preset = parsed;
                }
            }
        }
    }
    preset
}

fn preset_label(preset: TrainingPreset) -> &'static str {
    match preset {
        TrainingPreset::Light => "light",
        TrainingPreset::Balanced => "balanced",
        TrainingPreset::Max => "max",
    }
}

fn configure_model(preset: TrainingPreset, vocab_size: usize) -> CascadeTransformerBuilder {
    match preset {
        TrainingPreset::Light => CascadeTransformerBuilder::new(vocab_size)
            .block_size(256)
            .model_width(320)
            .layers(4, 5)
            .dropout(0.05),
        TrainingPreset::Balanced => CascadeTransformerBuilder::new(vocab_size)
            .block_size(256)
            .model_width(384)
            .layers(6, 6)
            .dropout(0.1),
        TrainingPreset::Max => CascadeTransformerBuilder::new(vocab_size)
            .block_size(384)
            .model_width(512)
            .layers(8, 8)
            .dropout(0.1),
    }
}

fn configure_training(preset: TrainingPreset, device: candle_core::Device) -> TrainingConfig {
    let mut config = TrainingConfig::default();
    config.device = device;
    match preset {
        TrainingPreset::Light => {
            config.batch_size = 32;
            config.max_iters = 8000;
            config.eval_interval = 250;
        }
        TrainingPreset::Balanced => {
            config.batch_size = 48;
            config.max_iters = 12000;
        }
        TrainingPreset::Max => {
            config.batch_size = 96;
            config.max_iters = 20000;
            config.eval_interval = 500;
        }
    }
    config
}

fn main() -> Result<()> {
    println!("Cascade Transformer (Rust + Candle)");
    println!("==========================================\n");

    check_available_backends();
    println!();

    let device = setup_device()?;
    println!("Device: {:?}\n", device);

    let data_path = env::var("DATA_PATH")
        .unwrap_or_else(|_| "pt-data/enwiki-latest-pages-articles-multistream.xml.bz2".to_string());
    let mut corpus_path = PathBuf::from(&data_path);
    if !corpus_path.exists() {
        anyhow::bail!("Data file not found: {}", corpus_path.display());
    }
    println!("Found dataset at {}", corpus_path.display());

    if corpus_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("bz2"))
        .unwrap_or(false)
    {
        let cleaned_name = corpus_path
            .file_stem()
            .map(|stem| format!("{}-clean.txt", stem.to_string_lossy()))
            .unwrap_or_else(|| "wiki-clean.txt".to_string());
        let cleaned_path = corpus_path
            .parent()
            .map(|p| p.join(cleaned_name))
            .unwrap_or_else(|| PathBuf::from("pt-data/wiki-clean.txt"));
        println!(
            "Preparing cleaned corpus ({} -> {})... this may take several minutes.",
            corpus_path.display(),
            cleaned_path.display()
        );
        let preprocessing_started = Instant::now();
        preprocess_wikipedia_dump(&corpus_path, &cleaned_path)
            .with_context(|| "Failed to preprocess Wikipedia dump")?;
        println!(
            "Clean corpus ready after {:.1?} at {}",
            preprocessing_started.elapsed(),
            cleaned_path.display()
        );
        corpus_path = cleaned_path;
    } else {
        println!("Using text corpus at {}", corpus_path.display());
    }

    if let Ok(meta) = fs::metadata(&corpus_path) {
        let size_mib = meta.len() as f64 / (1024.0 * 1024.0);
        println!("Corpus size: {:.1} MiB", size_mib);
    }

    let tokenizer_cfg = TokenizerConfig::default();
    println!("Building tokenizer from corpus (may take a while for large datasets)...");
    let corpus_text = load_corpus_with_progress(&corpus_path)?;
    let mut tokenizer = AdvancedTokenizer::from_text(&corpus_text, device.clone(), tokenizer_cfg)
        .context("Failed to create tokenizer")?;
    println!("Using corpus: {}", corpus_path.display());
    println!(
        "Tokenizer ready: vocab {} | train tokens {} | val tokens {}",
        tokenizer.vocab_size(),
        tokenizer.train_len(),
        tokenizer.val_len()
    );

    let preset = parse_training_preset();
    println!("Using preset: {}", preset_label(preset));

    let builder = configure_model(preset, tokenizer.vocab_size());
    let model_config = builder.config().clone();
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut model = builder
        .build(vb)
        .context("Failed to build Cascade Transformer")?;
    println!("Model parameters: ~{}", model.count_parameters());

    let mut training_config = configure_training(preset, device.clone());
    training_config.block_size = model.config.block_size;
    apply_training_overrides(&mut training_config);
    println!(
        "Training configuration: batch={} max_iters={} block={}",
        training_config.batch_size, training_config.max_iters, training_config.block_size
    );

    println!("Starting training loop...");
    let stats = match train_model(&model, &tokenizer, &mut varmap, &training_config) {
        Ok(stats) => stats,
        Err(err) => {
            if device.is_metal() {
                println!(
                    "Metal backend training failed: {}. Falling back to CPU...",
                    err
                );
                let fallback_device = candle_core::Device::Cpu;
                let mut cpu_config = training_config.clone();
                cpu_config.device = fallback_device.clone();
                let cpu_tokenizer = tokenizer.clone().to_device(fallback_device.clone());
                let mut cpu_varmap = VarMap::new();
                let cpu_vb =
                    candle_nn::VarBuilder::from_varmap(&cpu_varmap, DType::F32, &fallback_device);
                let cpu_model = CascadeTransformer::new(model_config.clone(), cpu_vb)
                    .context("Failed to rebuild model on CPU")?;
                println!("Retrying training on CPU...");
                let cpu_stats =
                    train_model(&cpu_model, &cpu_tokenizer, &mut cpu_varmap, &cpu_config)
                        .context("Training failed after Metal fallback")?;
                model = cpu_model;
                tokenizer = cpu_tokenizer;
                println!(
                    "CPU training complete. Continuing with CPU backend for interactive phase."
                );
                cpu_stats
            } else {
                return Err(err).context("Training failed")?;
            }
        }
    };
    if let Some(last) = stats.last() {
        println!(
            "Final train loss: {:.4} | val loss: {:.4}",
            last.train_loss, last.val_loss
        );
    }

    interactive_generation_loop(&model, &tokenizer)?;
    Ok(())
}

fn load_corpus_with_progress(path: &Path) -> Result<String> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open corpus at {}", path.display()))?;
    let metadata = file.metadata().ok();
    let total_bytes = metadata.map(|m| m.len()).unwrap_or(0);
    let (explicit_limit, explicit_override) = tokenizer_limit_bytes();
    let mut limit_info = explicit_limit;
    if !explicit_override {
        if total_bytes
            > TOKENIZER_DEFAULT_LIMIT_MIB
                .saturating_mul(1024)
                .saturating_mul(1024)
        {
            let default_bytes = TOKENIZER_DEFAULT_LIMIT_MIB
                .saturating_mul(1024)
                .saturating_mul(1024);
            limit_info = Some((default_bytes, "default 512 MiB cap"));
        }
    }
    let limit_bytes = limit_info.map(|(bytes, _)| bytes);

    if total_bytes == 0 && limit_bytes.is_none() {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")
                .unwrap()
                .tick_strings(&["-", "\\", "|", "/"]),
        );
        spinner.enable_steady_tick(std::time::Duration::from_millis(120));
        spinner.set_message("Reading corpus...");
        let mut reader = io::BufReader::new(file);
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .with_context(|| format!("failed to read corpus at {}", path.display()))?;
        spinner.finish_with_message("Corpus loaded");
        return Ok(content);
    }

    let pb_total = limit_bytes.map_or(total_bytes, |limit| {
        if total_bytes == 0 {
            limit
        } else {
            limit.min(total_bytes)
        }
    });
    let pb = ProgressBar::new(pb_total.max(1));
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {bytes:>12}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("Loading corpus");

    let mut reader = io::BufReader::with_capacity(2 * 1024 * 1024, file);
    let capacity = limit_bytes.unwrap_or(total_bytes).min(usize::MAX as u64) as usize;
    let mut buffer = Vec::with_capacity(capacity);
    let mut chunk = vec![0u8; 2 * 1024 * 1024];
    let mut collected = 0u64;
    loop {
        let bytes_read = reader
            .read(&mut chunk)
            .with_context(|| format!("failed while reading corpus at {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        let mut take = bytes_read as u64;
        if let Some(limit) = limit_bytes {
            let remaining = limit.saturating_sub(collected);
            if remaining == 0 {
                break;
            }
            take = take.min(remaining);
        }
        buffer.extend_from_slice(&chunk[..take as usize]);
        collected += take;
        pb.inc(take);
        if let Some(limit) = limit_bytes {
            if collected >= limit {
                break;
            }
        }
    }
    pb.finish_with_message(if limit_bytes.is_some() {
        "Corpus sample loaded"
    } else {
        "Corpus loaded"
    });

    drop(pb);

    if let Some((limit, source)) = limit_info {
        if total_bytes > limit && total_bytes > 0 {
            let mib = limit as f64 / (1024.0 * 1024.0);
            if source.starts_with("default") {
                println!(
                    "Tokenizer corpus limited to first {:.1} MiB ({}). Override with TOKENIZER_MAX_BYTES=0 to disable.",
                    mib, source
                );
            } else {
                println!(
                    "Tokenizer corpus limited to first {:.1} MiB via {}.",
                    mib, source
                );
            }
        } else if total_bytes == 0 {
            let mib = limit as f64 / (1024.0 * 1024.0);
            println!(
                "Tokenizer corpus limited to approximately {:.1} MiB via {} (input size unknown).",
                mib, source
            );
        }
    }

    let valid_len = match std::str::from_utf8(&buffer) {
        Ok(_) => buffer.len(),
        Err(err) => {
            let valid = err.valid_up_to();
            if valid == 0 {
                return Err(anyhow!("corpus at {} is not valid UTF-8", path.display()));
            }
            valid
        }
    };
    buffer.truncate(valid_len);

    String::from_utf8(buffer)
        .with_context(|| format!("corpus at {} is not valid UTF-8", path.display()))
}

fn tokenizer_limit_bytes() -> (Option<(u64, &'static str)>, bool) {
    if let Ok(value) = env::var("TOKENIZER_MAX_BYTES") {
        if let Ok(parsed) = value.parse::<u64>() {
            if parsed == 0 {
                return (None, true);
            }
            return (Some((parsed, "TOKENIZER_MAX_BYTES")), true);
        }
    }
    if let Ok(value) = env::var("TOKENIZER_SAMPLE_MIB") {
        if let Ok(parsed) = value.parse::<u64>() {
            if parsed == 0 {
                return (None, true);
            }
            return (
                Some((
                    parsed.saturating_mul(1024).saturating_mul(1024),
                    "TOKENIZER_SAMPLE_MIB",
                )),
                true,
            );
        }
    }
    (None, false)
}

fn apply_training_overrides(config: &mut TrainingConfig) {
    if let Ok(value) = env::var("TRAINING_MAX_ITERS") {
        if let Ok(parsed) = value.parse::<usize>() {
            config.max_iters = parsed.max(1);
        }
    }
    if let Ok(value) = env::var("TRAINING_EVAL_INTERVAL") {
        if let Ok(parsed) = value.parse::<usize>() {
            config.eval_interval = parsed.max(1);
        }
    }
    if let Ok(value) = env::var("TRAINING_EVAL_ITERS") {
        if let Ok(parsed) = value.parse::<usize>() {
            config.eval_iters = parsed.max(1);
        }
    }
    if let Ok(value) = env::var("TRAINING_BATCH_SIZE") {
        if let Ok(parsed) = value.parse::<usize>() {
            config.batch_size = parsed.max(1);
        }
    }
    if let Ok(value) = env::var("TRAINING_LOG_INTERVAL") {
        if let Ok(parsed) = value.parse::<usize>() {
            config.log_interval = parsed.max(1);
        }
    }
}

fn interactive_generation_loop(
    model: &CascadeTransformer,
    tokenizer: &AdvancedTokenizer,
) -> Result<()> {
    println!("\nInteractive generation. Empty prompt to exit.\n");
    let palette = CreativePalette::new(CreativeMode::Dramatic);
    let progressive = ProgressiveRefiner::new(ProgressiveGenerationConfig::default());
    let stdin = io::stdin();
    loop {
        print!("Prompt> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        if stdin.read_line(&mut prompt)? == 0 {
            println!("\nEOF received. Bye.");
            break;
        }
        let prompt = prompt.trim();
        if prompt.is_empty() {
            println!("Empty prompt. Exiting.");
            break;
        }
        let composed = build_prompt(prompt);
        let prompt_tokens = tokenizer.encode(&composed)?;
        if prompt_tokens.is_empty() {
            println!("Prompt could not be encoded.");
            continue;
        }
        let prompt_tensor = Tensor::from_vec(
            prompt_tokens.clone(),
            (1, prompt_tokens.len()),
            tokenizer.device(),
        )?;
        let generated = progressive.generate(model, &prompt_tensor, &palette)?;
        let flat = generated.get(0)?;
        let indices = flat.to_vec1::<u32>()?;
        let indices: Vec<u32> = indices;
        let response = tokenizer.decode(&indices[prompt_tokens.len()..])?;
        println!("\n{}\n", "=".repeat(60));
        println!("{}", response);
        println!("{}\n", "=".repeat(60));
    }
    Ok(())
}
