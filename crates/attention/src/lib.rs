//! Exact causal self-attention primitives for the transformer project.
//!
//! The crate defines a portable API for computing causal self-attention over
//! tensors with layout `[batch, n_heads, seq_len, head_dim]`. The inputs `Q`,
//! `K`, and `V` share the same layout and dtype (bf16, f16, or f32). Reductions
//! are performed internally in `f32`, and the output tensor matches the input
//! dtype and shape.
//!
//! Dropout is an optional, train-only concern controlled via the public
//! configuration. Callers should disable it for evaluation or when deterministic
//! outputs are required.
//!
//! Causal masks are always enforced, ensuring each token attends only to prior
//! context. Additional padding masks can be supplied to ignore padded
//! positions; accepted mask shapes are documented alongside the [`Attention`]
//! trait.

pub mod core;
pub mod interop;
pub mod kv_cache;
pub mod masks;
pub mod reference;

#[cfg(feature = "fused")]
pub mod fused;

pub use core::{Attention, AttentionError, Config};
