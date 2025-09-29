//! Interfaces and helpers for key/value cache management.
//!
//! Attention implementations can use these utilities to persist incremental
//! context across decoding steps.

pub mod api;
pub mod layout;
pub mod paged;

pub use api::{CacheStats, KeyValueCache};
pub use layout::CacheLayout;
pub use paged::{CacheConfig, PagedKeyValueCache};

#[cfg(test)]
mod tests;
