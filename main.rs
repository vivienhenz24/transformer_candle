use pretraining_data;
use tokenizer;

fn main() {
    println!("Transformer project initialized");
    
    // Initialize the crates
    pretraining_data::init();
    tokenizer::init();
}
