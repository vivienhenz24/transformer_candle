use anyhow::{Context, Result};
use cascade_core::{
    CascadeTransformer,
    CascadeTransformerBuilder,
    CreativeMode,
    CreativePalette,
    ProgressiveGenerationConfig,
    ProgressiveRefiner,
};
use cascade_training::{train_model, TrainingConfig};
use candle_core::{DType, Tensor};
use candle_nn::VarMap;
use std::{
    env,
    io::{self, Write},
    path::Path,
};
use transformer_tokenization::{AdvancedTokenizer, TokenizerConfig};
use utils::prompts::build_prompt;

use transformer::{check_available_backends, setup_device};

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

    let data_path = "pt-data/input.txt";
    if !Path::new(data_path).exists() {
        anyhow::bail!("Data file not found: {data_path}");
    }

    let tokenizer_cfg = TokenizerConfig::default();
    let tokenizer = AdvancedTokenizer::from_file(data_path, device.clone(), tokenizer_cfg)
        .context("Failed to create tokenizer")?;
    println!(
        "Tokenizer ready: vocab {} | train tokens {} | val tokens {}",
        tokenizer.vocab_size(),
        tokenizer.train_len(),
        tokenizer.val_len()
    );

    let preset = parse_training_preset();
    println!("Using preset: {}", preset_label(preset));

    let builder = configure_model(preset, tokenizer.vocab_size());
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = builder
        .build(vb)
        .context("Failed to build Cascade Transformer")?;
    println!("Model parameters: ~{}", model.count_parameters());

    let mut training_config = configure_training(preset, device.clone());
    training_config.block_size = model.config.block_size;
    println!(
        "Training configuration: batch={} max_iters={} block={}",
        training_config.batch_size, training_config.max_iters, training_config.block_size
    );

    let stats = train_model(&model, &tokenizer, &mut varmap, &training_config)
        .context("Training failed")?;
    if let Some(last) = stats.last() {
        println!("Final train loss: {:.4} | val loss: {:.4}", last.train_loss, last.val_loss);
    }

    interactive_generation_loop(&model, &tokenizer)?;
    Ok(())
}

fn interactive_generation_loop(model: &CascadeTransformer, tokenizer: &AdvancedTokenizer) -> Result<()> {
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
        let prompt_tensor = Tensor::from_vec(prompt_tokens.clone(), (1, prompt_tokens.len()), tokenizer.device())?;
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
