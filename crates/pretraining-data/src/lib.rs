//! Pretraining data handling crate

pub mod corpora;
pub mod preprocessing;
pub mod sharding;

// Re-export main types
pub use corpora::{HuggingFaceStreamingCorpus, StreamingCorpus, TextCorpus};

/// Initialize the pretraining data module
pub fn init() {
    println!("Pretraining data module initialized");
}
