//! Interfaces and helpers for key/value cache management.
//!
//! Attention implementations can use these utilities to persist incremental
//! context across decoding steps.

pub mod api;
pub mod layout;

pub use api::KeyValueCache;
pub use layout::CacheLayout;
