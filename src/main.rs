mod tokenizer;
mod gpt;

use candle_core::{Device, DType};
use candle_nn::VarMap;
use tokenizer::{CharTokenizer, DataSplit};
use gpt::{Head, MultiHeadAttention, FeedForward};

fn main() -> anyhow::Result<()> {
    println!("🚀 Character-level Transformer in Rust!");
    
    // Initialize device (CPU for now, could be CUDA if available)
    let device = Device::Cpu;
    
    // Create tokenizer from the Shakespeare data
    println!("📚 Loading Shakespeare data...");
    let tokenizer = CharTokenizer::from_file("pt-data/input.txt", device.clone())?;
    
    // Print tokenizer statistics
    tokenizer.print_stats();
    
    // Test encoding and decoding
    println!("\n🔤 Testing encode/decode:");
    let test_text = "Hello, world!";
    let encoded = tokenizer.encode(test_text);
    let decoded = tokenizer.decode(&encoded);
    println!("  Original: '{}'", test_text);
    println!("  Encoded:  {:?}", encoded);
    println!("  Decoded:  '{}'", decoded);
    
    // Test batch generation
    println!("\n📦 Testing batch generation:");
    let batch_size = 4;
    let block_size = 8;
    
    match tokenizer.get_batch(DataSplit::Train, batch_size, block_size) {
        Ok((inputs, targets)) => {
            println!("  Batch inputs shape: {:?}", inputs.shape());
            println!("  Batch targets shape: {:?}", targets.shape());
            
            // Show first sequence in the batch
            if let Ok(first_input) = inputs.get(0) {
                if let Ok(input_data) = first_input.to_vec1::<u32>() {
                    let input_chars = tokenizer.decode(&input_data.iter().map(|&x| x as usize).collect::<Vec<_>>());
                    println!("  First input sequence: '{}'", input_chars);
                }
            }
            
            if let Ok(first_target) = targets.get(0) {
                if let Ok(target_data) = first_target.to_vec1::<u32>() {
                    let target_chars = tokenizer.decode(&target_data.iter().map(|&x| x as usize).collect::<Vec<_>>());
                    println!("  First target sequence: '{}'", target_chars);
                }
            }
        }
        Err(e) => println!("  Error generating batch: {}", e),
    }
    
    // Test attention head implementation
    println!("\n🧠 Testing attention head implementation:");
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Create a single attention head
    let n_embd = 72;  // Embedding dimension (divisible by 6 for 6-head test)
    let head_size = 12; // Head size for single head test
    let block_size = 32; // Block size
    let dropout_rate = 0.1;
    
    match Head::new(n_embd, head_size, block_size, dropout_rate, vb.pp("test_head")) {
        Ok(head) => {
            println!("  ✅ Single attention head created successfully");
            println!("     - Embedding dim: {}", n_embd);
            println!("     - Head size: {}", head_size);
            println!("     - Block size: {}", block_size);
            
            // Test with dummy input
            let batch_size = 2;
            let seq_len = 8;
            let dummy_input = candle_core::Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)?.to_dtype(DType::F32)?;
            
            match head.forward(&dummy_input, false) {
                Ok(output) => {
                    println!("  ✅ Forward pass successful");
                    println!("     - Input shape: {:?}", dummy_input.shape());
                    println!("     - Output shape: {:?}", output.shape());
                }
                Err(e) => println!("  ❌ Forward pass failed: {}", e),
            }
        }
        Err(e) => println!("  ❌ Head creation failed: {}", e),
    }
    
    // Test multi-head attention with 6 heads (as requested)
    println!("\n🔄 Testing multi-head attention:");
    let n_head = 6;  // Testing with 6 heads as specifically requested
    let vb_mha = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    match MultiHeadAttention::new(n_embd, n_head, block_size, dropout_rate, vb_mha.pp("test_mha")) {
        Ok(mha) => {
            println!("  ✅ Multi-head attention created successfully");
            println!("     - Number of heads: {}", n_head);
            println!("     - Head size: {}", n_embd / n_head);
            
            let batch_size = 2;
            let seq_len = 8;
            let dummy_input = candle_core::Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)?.to_dtype(DType::F32)?;
            
            match mha.forward(&dummy_input, false) {
                Ok(output) => {
                    println!("  ✅ Multi-head forward pass successful");
                    println!("     - Input shape: {:?}", dummy_input.shape());
                    println!("     - Output shape: {:?}", output.shape());
                    
                    // Verify shape preservation: input and output should have same shape
                    if dummy_input.shape() == output.shape() {
                        println!("  ✅ Shape preservation confirmed: (batch, time, channels) preserved");
                    } else {
                        println!("  ❌ Shape mismatch detected");
                    }
                    
                    // Test with training mode (to test dropout)
                    match mha.forward(&dummy_input, true) {
                        Ok(train_output) => {
                            println!("  ✅ Training mode forward pass successful");
                            println!("     - Training output shape: {:?}", train_output.shape());
                        }
                        Err(e) => println!("  ❌ Training mode failed: {}", e),
                    }
                }
                Err(e) => println!("  ❌ Multi-head forward pass failed: {}", e),
            }
        }
        Err(e) => println!("  ❌ Multi-head attention creation failed: {}", e),
    }
    
    // Test feed-forward network
    println!("\n🧮 Testing feed-forward network:");
    let vb_ff = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    match FeedForward::new(n_embd, dropout_rate, vb_ff.pp("test_ff")) {
        Ok(ff) => {
            println!("  ✅ FeedForward network created successfully");
            println!("     - Input dimension: {}", n_embd);
            println!("     - Internal expansion: {} (4x)", 4 * n_embd);
            println!("     - Output dimension: {}", n_embd);
            
            let batch_size = 2;
            let seq_len = 8;
            let dummy_input = candle_core::Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)?.to_dtype(DType::F32)?;
            
            match ff.forward(&dummy_input, false) {
                Ok(output) => {
                    println!("  ✅ Forward pass successful");
                    println!("     - Input shape: {:?}", dummy_input.shape());
                    println!("     - Output shape: {:?}", output.shape());
                    
                    // Verify shape preservation
                    if dummy_input.shape() == output.shape() {
                        println!("  ✅ Shape preservation confirmed: (batch, time, channels) preserved");
                    } else {
                        println!("  ❌ Shape mismatch detected");
                    }
                    
                    // Test training mode with dropout
                    match ff.forward(&dummy_input, true) {
                        Ok(train_output) => {
                            println!("  ✅ Training mode (with dropout) successful");
                            println!("     - Training output shape: {:?}", train_output.shape());
                        }
                        Err(e) => println!("  ❌ Training mode failed: {}", e),
                    }
                }
                Err(e) => println!("  ❌ Forward pass failed: {}", e),
            }
        }
        Err(e) => println!("  ❌ FeedForward creation failed: {}", e),
    }
    
    println!("\n✅ Complete transformer components implemented!");
    println!("🔧 Ready for transformer block assembly!");
    Ok(())
}
