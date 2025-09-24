mod gpt;
mod tokenizer;
mod training;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use std::{
    env,
    io::{self, Write},
    path::Path,
};

use gpt::{GPTConfig, GPTLanguageModel};
use tokenizer::{CharTokenizer, DataSplit};
use training::{create_medium_gpt_config, train_model, TrainingConfig};

#[derive(Debug, Clone, Copy)]
enum TrainingPreset {
    Light,
    Balanced,
    Max,
}

fn preset_from_str(value: &str) -> Option<TrainingPreset> {
    match value.trim().to_lowercase().as_str() {
        "light" | "small" => Some(TrainingPreset::Light),
        "balanced" | "medium" | "default" => Some(TrainingPreset::Balanced),
        "max" | "heavy" | "large" => Some(TrainingPreset::Max),
        _ => None,
    }
}

fn parse_training_preset() -> TrainingPreset {
    let mut preset = TrainingPreset::Light;

    if let Ok(env_value) = env::var("TRAINING_PRESET") {
        if let Some(parsed) = preset_from_str(&env_value) {
            preset = parsed;
        }
    }

    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--preset=") {
            if let Some(parsed) = preset_from_str(value) {
                preset = parsed;
            }
        } else if arg == "--preset" {
            if let Some(value) = args.next() {
                if let Some(parsed) = preset_from_str(&value) {
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

fn apply_training_preset(
    preset: TrainingPreset,
    gpt_config: &mut GPTConfig,
    training_config: &mut TrainingConfig,
) {
    match preset {
        TrainingPreset::Light => {
            gpt_config.block_size = 256;
            gpt_config.n_embd = 384;
            gpt_config.n_head = 6;
            gpt_config.n_layer = 4;

            training_config.batch_size = 32;
            training_config.block_size = gpt_config.block_size;
            training_config.learning_rate = 4e-4;
            training_config.max_iters = 12000;
            training_config.eval_interval = 300;
            training_config.warmup_steps = 500;
            training_config.checkpoint_interval = 600;
            training_config.eval_iters = 150;
        }
        TrainingPreset::Balanced => {
            gpt_config.block_size = 256;
            gpt_config.n_embd = 384;
            gpt_config.n_head = 6;
            gpt_config.n_layer = 6;

            training_config.batch_size = 48;
            training_config.block_size = gpt_config.block_size;
            training_config.learning_rate = 5e-4;
            training_config.max_iters = 15000;
            training_config.eval_interval = 400;
            training_config.warmup_steps = 800;
            training_config.checkpoint_interval = 800;
            training_config.eval_iters = 200;
        }
        TrainingPreset::Max => {
            gpt_config.block_size = 384;
            gpt_config.n_embd = 512;
            gpt_config.n_head = 8;
            gpt_config.n_layer = 6;

            training_config.batch_size = 96;
            training_config.block_size = gpt_config.block_size;
            training_config.learning_rate = 6e-4;
            training_config.max_iters = 25000;
            training_config.eval_interval = 500;
            training_config.warmup_steps = 1200;
            training_config.checkpoint_interval = 1000;
            training_config.eval_iters = 250;
        }
    }

    training_config.block_size = gpt_config.block_size;
    training_config.eval_interval = training_config
        .eval_interval
        .clamp(1, training_config.max_iters.max(1));
    training_config.checkpoint_interval = training_config
        .checkpoint_interval
        .max(200)
        .min(training_config.max_iters.max(200));
    training_config.eval_iters = training_config.eval_iters.max(50);
}

/// Main function that orchestrates the complete GPT training and text generation pipeline
fn main() -> Result<()> {
    println!("Transformer in Rust");
    println!("==========================================\n");

    // Step 1: Check available backends and system info
    check_available_backends();
    println!();

    // Step 2: Device setup with detailed debugging
    let device = setup_device()?;
    println!("📱 Final device selection: {:?}\n", device);

    // Step 3: Load and process Shakespeare text
    println!("Loading Shakespeare dataset");
    let data_path = "pt-data/input.txt";

    if !Path::new(data_path).exists() {
        return Err(anyhow::anyhow!(
            "Data file not found: {}. Please ensure the Shakespeare text file is available.",
            data_path
        ));
    }

    let tokenizer = CharTokenizer::from_file(data_path, device.clone())
        .context("Failed to create tokenizer from Shakespeare data")?;

    // Step 4: Display dataset statistics
    println!("✅ Dataset loaded successfully!");
    tokenizer.print_stats();

    // Step 5: Create GPT model with preset-specific hyperparameters
    println!("\n Creating GPT model...");
    let mut gpt_config = create_medium_gpt_config(tokenizer.vocab_size);
    let mut training_config = TrainingConfig::default();
    let preset = parse_training_preset();
    apply_training_preset(preset, &mut gpt_config, &mut training_config);

    // Create variable map for model parameters
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model =
        GPTLanguageModel::new(gpt_config.clone(), vb).context("Failed to create GPT model")?;

    // Step 5: Print model information
    let param_count = model.count_parameters();
    println!("GPT model created successfully!");
    println!("    Model Statistics:");
    println!("      - Vocabulary size: {}", gpt_config.vocab_size);
    println!("      - Block size (context): {}", gpt_config.block_size);
    println!("      - Embedding dimension: {}", gpt_config.n_embd);
    println!("      - Number of layers: {}", gpt_config.n_layer);
    println!("      - Number of attention heads: {}", gpt_config.n_head);
    println!("      - Total parameters: ~{}", param_count);
    println!("      - Dropout rate: {}", gpt_config.dropout_rate);

    // Step 6: Configure training parameters
    println!("\n Setting up training configuration...");
    training_config.device = device.clone();

    println!(" Training configuration:");
    println!("      - Preset: {}", preset_label(preset));
    println!("      - Learning rate: {}", training_config.learning_rate);
    println!("      - Batch size: {}", training_config.batch_size);
    println!("      - Block size: {}", training_config.block_size);
    println!("      - Max iterations: {}", training_config.max_iters);
    println!(
        "      - Evaluation interval: {}",
        training_config.eval_interval
    );
    println!("      - Warmup steps: {}", training_config.warmup_steps);
    println!(
        "      - Checkpoint interval: {}",
        training_config.checkpoint_interval
    );

    // Step 7: Validate data splits
    println!("\n Validating data splits...");
    match tokenizer.get_batch(DataSplit::Train, 4, training_config.block_size) {
        Ok((inputs, targets)) => {
            println!(
                " Training data: {} batches available",
                inputs.shape().dims()[0]
            );
            println!("   - Input shape: {:?}", inputs.shape());
            println!("   - Target shape: {:?}", targets.shape());
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Training data validation failed: {}", e));
        }
    }

    match tokenizer.get_batch(DataSplit::Val, 4, training_config.block_size) {
        Ok((inputs, targets)) => {
            println!(
                " Validation data: {} batches available",
                inputs.shape().dims()[0]
            );
            println!("   - Input shape: {:?}", inputs.shape());
            println!("   - Target shape: {:?}", targets.shape());
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Validation data validation failed: {}", e));
        }
    }

    // Step 8: Run training loop
    println!("\n Starting training...");
    println!("====================================");

    let training_stats = train_model(&model, &tokenizer, &mut varmap, &training_config)
        .context("Training failed")?;

    println!("====================================");
    println!(" Training completed successfully!");

    if let Some(final_stats) = training_stats.last() {
        println!(" Final Training Statistics:");
        println!("   - Final training loss: {:.4}", final_stats.train_loss);
        println!("   - Final validation loss: {:.4}", final_stats.val_loss);
        println!("   - Average tokens/sec: {:.0}", final_stats.tokens_per_sec);
        println!("   - Total training time: {:.1}s", final_stats.elapsed_time);
        println!(
            "   - Training iterations completed: {}",
            training_stats.len()
        );
    }

    // Step 9: Run interactive text generation loop
    println!("\n Starting interactive text generation...");
    println!("============================================");
    interactive_generation_loop(&model, &tokenizer)
        .context("Interactive text generation failed")?;

    // Step 10: Final summary
    println!("\n GPT Training and Generation Complete.");
    println!("========================================");
    println!(" Successfully completed all steps:");
    println!(
        "   1. Loaded Shakespeare dataset ({} characters)",
        tokenizer.vocab_size
    );
    println!(
        "   2. Created and configured GPT model ({} parameters)",
        param_count
    );
    println!(
        "   3. Trained model for {} iterations",
        training_config.max_iters
    );
    println!("   4. Generated text samples interactively with trained model");
    println!("\n Ready for production use or further training!");

    Ok(())
}

/// Setup compute device (prefer Metal on Apple, then CUDA, otherwise CPU)
fn setup_device() -> Result<Device> {
    println!("Starting device detection...");

    // Prefer the Metal backend when compiled with the Metal feature.
    #[cfg(feature = "metal")]
    {
        println!("Checking for Metal GPU support...");
        match Device::new_metal(0) {
            Ok(device) => {
                println!("Metal GPU detected successfully!");
                println!("   Device type: {:?}", device);
                println!("   Device info: {:?}", device);
                println!("Metal GPU will be used for training");
                return Ok(device);
            }
            Err(err) => {
                println!("Metal GPU detection failed: {err}");
                println!("Falling back to other backends...\n");
            }
        }
    }

    // When the Metal feature isn’t enabled we silently fall through to other backends.
    #[cfg(not(feature = "metal"))]
    {
        println!("Metal backend not enabled at compile time, skipping detection.\n");
    }

    // Check for CUDA support next.
    println!("🔍 Checking for CUDA GPU support...");
    match Device::cuda_if_available(0) {
        Ok(device) => {
            if device.is_cuda() {
                println!("CUDA GPU detected and will be used for training");
                return Ok(device);
            }

            println!("CUDA device handle returned but not reporting as CUDA, skipping.");
        }
        Err(err) => {
            println!("CUDA GPU detection failed: {err}");
        }
    }

    println!("💻 Using CPU for training (no GPU backend available)");
    Ok(Device::Cpu)
}

/// Check what backends are available and provide system information
fn check_available_backends() {
    println!(" System Information:");
    println!("   OS: {}", std::env::consts::OS);
    println!("   Architecture: {}", std::env::consts::ARCH);
    println!("   Family: {}", std::env::consts::FAMILY);

    // Check if we're on macOS (where Metal should be available)
    #[cfg(target_os = "macos")]
    {
        println!("   Platform: macOS (Metal should be available)");

        // Try to get more system info
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(&["SPHardwareDataType"])
            .output()
        {
            if let Ok(hardware_info) = String::from_utf8(output.stdout) {
                for line in hardware_info.lines() {
                    if line.contains("Chip") || line.contains("Processor") || line.contains("Model")
                    {
                        println!("   Hardware: {}", line.trim());
                    }
                }
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("   Platform: Non-macOS (Metal not available)");
    }

    // Check candle-core features
    println!("\n Candle-core backend availability:");
    println!("   CPU: Always available");
    println!("   Metal: Checking at runtime (feature detection not available)");
    println!("   CUDA: Checking at runtime (feature detection not available)");
}

/// Interactive loop allowing the user to provide prompts and control output length.
fn interactive_generation_loop(model: &GPTLanguageModel, tokenizer: &CharTokenizer) -> Result<()> {
    println!("Type a prompt and press Enter to generate text.");
    println!("Submit an empty prompt or press Ctrl+D to exit.\n");
    println!("Using sampling settings: max_tokens=200, temperature=0.2, top-k=10, top-p=0.9\n");

    let stdin = io::stdin();
    loop {
        print!("Prompt> ");
        io::stdout().flush().context("Failed to flush stdout")?;

        let mut prompt = String::new();
        let bytes_read = stdin
            .read_line(&mut prompt)
            .context("Failed to read prompt")?;

        if bytes_read == 0 {
            println!("\nEOF received. Exiting interactive session.");
            break;
        }

        let prompt = prompt.trim();
        if prompt.is_empty() {
            println!("Empty prompt detected. Exiting interactive session.");
            break;
        }

        let max_tokens = 200usize;
        let temperature = 0.2f64;
        let top_k = Some(10usize);
        let top_p = Some(0.9f64);

        match generate_single_sample(
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        ) {
            Ok(generated_text) => {
                println!("\n{}\n", "=".repeat(60));
                println!("{}", generated_text);
                println!("{}\n", "=".repeat(60));
            }
            Err(e) => println!("Failed to generate text: {}", e),
        }
    }

    Ok(())
}

/// Generate a single text sample from a prompt
fn generate_single_sample(
    model: &GPTLanguageModel,
    tokenizer: &CharTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<String> {
    // Encode the prompt
    let prompt_tokens = tokenizer.encode(prompt);

    if prompt_tokens.is_empty() {
        return Err(anyhow::anyhow!("Empty prompt tokens"));
    }

    // Convert to tensor and add batch dimension
    let prompt_tensor = tokenizer
        .indices_to_tensor(&prompt_tokens)
        .context("Failed to convert prompt to tensor")?;
    let prompt_tensor = prompt_tensor
        .unsqueeze(0)
        .context("Failed to add batch dimension")?;

    // Generate tokens
    let generated_tensor = model
        .generate_with_sampling(&prompt_tensor, max_tokens, temperature, top_k, top_p)
        .context("Failed to generate tokens")?;

    // Convert back to text
    let generated_flat = generated_tensor
        .get(0)
        .context("Failed to get generated sequence")?;
    let generated_indices = generated_flat
        .to_vec1::<u32>()
        .context("Failed to convert generated tensor to vector")?;

    let generated_indices: Vec<usize> = generated_indices.iter().map(|&x| x as usize).collect();

    let generated_text = tokenizer.decode(&generated_indices);

    Ok(generated_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_setup() {
        let device = setup_device();
        assert!(device.is_ok());
    }

    #[test]
    fn test_data_path_exists() {
        let data_path = "pt-data/input.txt";
        assert!(
            Path::new(data_path).exists(),
            "Shakespeare data file should exist"
        );
    }

    #[test]
    fn test_model_config_creation() {
        let config = create_medium_gpt_config(65);
        assert_eq!(config.vocab_size, 65);
        assert!(config.n_embd > 0);
        assert!(config.n_layer > 0);
        assert!(config.n_head > 0);
        assert!(config.block_size > 0);
    }

    #[test]
    fn test_tokenizer_integration() {
        if Path::new("pt-data/input.txt").exists() {
            let device = Device::Cpu;
            let tokenizer = CharTokenizer::from_file("pt-data/input.txt", device);
            assert!(tokenizer.is_ok());

            if let Ok(tokenizer) = tokenizer {
                assert!(tokenizer.vocab_size > 0);

                // Test encoding/decoding
                let test_text = "Hello";
                let encoded = tokenizer.encode(test_text);
                let decoded = tokenizer.decode(&encoded);
                assert_eq!(test_text, decoded);
            }
        }
    }
}
