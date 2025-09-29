//! Portable, exact implementations of causal self-attention.
//!
//! These paths favour clarity over absolute performance and serve as the
//! baseline for validating optimized kernels.

pub mod exact;

pub use exact::ExactAttention;
