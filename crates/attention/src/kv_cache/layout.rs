//! Layout definitions for key/value caches.
//!
//! These helpers describe how cached tensors are ordered in memory across
//! batch items, heads, and sequence positions.

/// Declarative description of a cache layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    /// Number of attention heads stored in the cache.
    pub num_heads: usize,
    /// Maximum sequence length captured by the cache.
    pub max_seq_len: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
}
