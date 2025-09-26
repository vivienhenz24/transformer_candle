//! Tokenizer crate

/*Re-exports Config, load_tokenizer, train_bbpe, build_from_artifacts.

Guards optional APIs behind features (#[cfg(feature = "train")]).

Ensures the final Tokenizer is Send + Sync.*/


// pub mod vocabulary;
// pub mod encoder;
// pub mod decoder;

/// Initialize the tokenizer module
pub fn init() {
    println!("Tokenizer module initialized");
}
