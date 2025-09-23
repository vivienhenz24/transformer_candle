mod tokenizer;
mod gpt;
mod training;

use std::path::Path;
use candle_core::{Device, DType};
use candle_nn::VarMap;
use anyhow::{Result, Context};

use tokenizer::{CharTokenizer, DataSplit};
use gpt::GPTLanguageModel;
use training::{TrainingConfig, train_model, create_medium_gpt_config};

/// Main function that orchestrates the complete GPT training and text generation pipeline
fn main() -> Result<()> {
    println!("🚀 Character-level GPT Transformer in Rust");
    println!("==========================================\n");
    
    // Step 1: Device setup with CUDA detection
    let device = setup_device()?;
    println!("📱 Device: {:?}\n", device);
    
    // Step 2: Load and process Shakespeare text
    println!("📚 Loading Shakespeare dataset...");
    let data_path = "pt-data/input.txt";
    
    if !Path::new(data_path).exists() {
        return Err(anyhow::anyhow!(
            "Data file not found: {}. Please ensure the Shakespeare text file is available.",
            data_path
        ));
    }
    
    let tokenizer = CharTokenizer::from_file(data_path, device.clone())
        .context("Failed to create tokenizer from Shakespeare data")?;
    
    // Step 3: Display dataset statistics
    println!("✅ Dataset loaded successfully!");
    tokenizer.print_stats();
    
    // Step 4: Create GPT model with appropriate hyperparameters
    println!("\n🤖 Creating GPT model...");
    let gpt_config = create_medium_gpt_config(tokenizer.vocab_size);
    
    // Create variable map for model parameters
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let model = GPTLanguageModel::new(gpt_config.clone(), vb)
        .context("Failed to create GPT model")?;
    
    // Step 5: Print model information
    let param_count = model.count_parameters();
    println!("✅ GPT model created successfully!");
    println!("   📊 Model Statistics:");
    println!("      - Vocabulary size: {}", gpt_config.vocab_size);
    println!("      - Block size (context): {}", gpt_config.block_size);
    println!("      - Embedding dimension: {}", gpt_config.n_embd);
    println!("      - Number of layers: {}", gpt_config.n_layer);
    println!("      - Number of attention heads: {}", gpt_config.n_head);
    println!("      - Total parameters: ~{}", param_count);
    println!("      - Dropout rate: {}", gpt_config.dropout_rate);
    
    // Step 6: Configure training parameters
    println!("\n🏋️ Setting up training configuration...");
    let mut training_config = TrainingConfig::default();
    training_config.device = device.clone();
    training_config.block_size = gpt_config.block_size;
    training_config.batch_size = 32;  // Reasonable batch size
    training_config.max_iters = 1000; // Enough iterations for demonstration
    training_config.eval_interval = 100;
    training_config.eval_iters = 50;
    training_config.learning_rate = 3e-4; // Conservative learning rate
    
    println!("✅ Training configuration:");
    println!("      - Learning rate: {}", training_config.learning_rate);
    println!("      - Batch size: {}", training_config.batch_size);
    println!("      - Block size: {}", training_config.block_size);
    println!("      - Max iterations: {}", training_config.max_iters);
    println!("      - Evaluation interval: {}", training_config.eval_interval);
    
    // Step 7: Validate data splits
    println!("\n📊 Validating data splits...");
    match tokenizer.get_batch(DataSplit::Train, 4, training_config.block_size) {
        Ok((inputs, targets)) => {
            println!("✅ Training data: {} batches available", inputs.shape().dims()[0]);
            println!("   - Input shape: {:?}", inputs.shape());
            println!("   - Target shape: {:?}", targets.shape());
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Training data validation failed: {}", e));
        }
    }
    
    match tokenizer.get_batch(DataSplit::Val, 4, training_config.block_size) {
        Ok((inputs, targets)) => {
            println!("✅ Validation data: {} batches available", inputs.shape().dims()[0]);
            println!("   - Input shape: {:?}", inputs.shape());
            println!("   - Target shape: {:?}", targets.shape());
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Validation data validation failed: {}", e));
        }
    }
    
    // Step 8: Run training loop
    println!("\n🚀 Starting training...");
    println!("====================================");
    
    let training_stats = train_model(&model, &tokenizer, &mut varmap, &training_config)
        .context("Training failed")?;
    
    println!("====================================");
    println!("✅ Training completed successfully!");
    
    if let Some(final_stats) = training_stats.last() {
        println!("📊 Final Training Statistics:");
        println!("   - Final training loss: {:.4}", final_stats.train_loss);
        println!("   - Final validation loss: {:.4}", final_stats.val_loss);
        println!("   - Average tokens/sec: {:.0}", final_stats.tokens_per_sec);
        println!("   - Total training time: {:.1}s", final_stats.elapsed_time);
        println!("   - Training iterations completed: {}", training_stats.len());
    }
    
    // Step 9: Generate text samples
    println!("\n🎯 Generating text samples...");
    println!("=====================================");
    
    generate_text_samples(&model, &tokenizer, &device)
        .context("Text generation failed")?;
    
    // Step 10: Final summary
    println!("\n🎉 GPT Training and Generation Complete!");
    println!("========================================");
    println!("✅ Successfully completed all steps:");
    println!("   1. ✅ Loaded Shakespeare dataset ({} characters)", tokenizer.vocab_size);
    println!("   2. ✅ Created and configured GPT model ({} parameters)", param_count);
    println!("   3. ✅ Trained model for {} iterations", training_config.max_iters);
    println!("   4. ✅ Generated text samples with trained model");
    println!("\n🚀 Ready for production use or further training!");
    
    Ok(())
}

/// Setup compute device (CPU or CUDA if available)
fn setup_device() -> Result<Device> {
    // Try to use CUDA if available, otherwise fall back to CPU
    match Device::cuda_if_available(0) {
        Ok(device) => {
            if device.is_cuda() {
                println!("🚀 CUDA GPU detected and will be used for training");
                Ok(device)
            } else {
                println!("💻 Using CPU for training (CUDA not available)");
                Ok(Device::Cpu)
            }
        }
        Err(_) => {
            println!("💻 Using CPU for training");
            Ok(Device::Cpu)
        }
    }
}

/// Generate various text samples to demonstrate the trained model
fn generate_text_samples(
    model: &GPTLanguageModel,
    tokenizer: &CharTokenizer,
    _device: &Device,
) -> Result<()> {
    let prompts = vec![
        "ROMEO:",
        "JULIET:",
        "To be or not to be",
        "Once upon a time",
        "HAMLET:",
    ];
    
    println!("Generating 500 tokens for various prompts:\n");
    
    for (i, prompt) in prompts.iter().enumerate() {
        println!("📝 Sample {} - Prompt: \"{}\"", i + 1, prompt);
        println!("{}", "-".repeat(60));
        
        match generate_single_sample(model, tokenizer, prompt, 500) {
            Ok(generated_text) => {
                println!("{}", generated_text);
                println!("\n{}\n", "=".repeat(60));
            }
            Err(e) => {
                println!("❌ Failed to generate text for prompt '{}': {}", prompt, e);
                println!("{}\n", "=".repeat(60));
            }
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
) -> Result<String> {
    // Encode the prompt
    let prompt_tokens = tokenizer.encode(prompt);
    
    if prompt_tokens.is_empty() {
        return Err(anyhow::anyhow!("Empty prompt tokens"));
    }
    
    // Convert to tensor and add batch dimension
    let prompt_tensor = tokenizer.indices_to_tensor(&prompt_tokens)
        .context("Failed to convert prompt to tensor")?;
    let prompt_tensor = prompt_tensor.unsqueeze(0)
        .context("Failed to add batch dimension")?;
    
    // Generate tokens
    let generated_tensor = model.generate(&prompt_tensor, max_tokens)
        .context("Failed to generate tokens")?;
    
    // Convert back to text
    let generated_flat = generated_tensor.get(0)
        .context("Failed to get generated sequence")?;
    let generated_indices = generated_flat.to_vec1::<u32>()
        .context("Failed to convert generated tensor to vector")?;
    
    let generated_indices: Vec<usize> = generated_indices
        .iter()
        .map(|&x| x as usize)
        .collect();
    
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
        assert!(Path::new(data_path).exists(), "Shakespeare data file should exist");
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
