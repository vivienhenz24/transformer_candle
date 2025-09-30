//! Pretraining data handling crate

pub mod corpora;
pub mod preprocessing;
pub mod sharding;

// Re-export main types
pub use corpora::{TextCorpus, StreamingCorpus, HuggingFaceStreamingCorpus};

/// Initialize the pretraining data module
pub fn init() {
    println!("Pretraining data module initialized");
}
