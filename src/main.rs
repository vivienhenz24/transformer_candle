mod tokenizer;

use candle_core::Device;
use tokenizer::{CharTokenizer, DataSplit};

fn main() -> anyhow::Result<()> {
    println!("🚀 Character-level Transformer in Rust!");
    
    // Initialize device (CPU for now, could be CUDA if available)
    let device = Device::Cpu;
    
    // Create tokenizer from the Shakespeare data
    println!("📚 Loading Shakespeare data...");
    let tokenizer = CharTokenizer::from_file("pt-data/input.txt", device)?;
    
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
    
    println!("\n✅ Tokenizer implementation complete!");
    Ok(())
}
