//! Pretraining data handling crate

pub mod corpora;
pub mod preprocessing;
pub mod sharding;

/// Initialize the pretraining data module
pub fn init() {
    println!("Pretraining data module initialized");
}
