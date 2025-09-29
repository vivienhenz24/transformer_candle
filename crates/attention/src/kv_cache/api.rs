//! Public traits describing key/value cache behaviour.

use candle_core::Tensor;

use crate::core::AttentionError;

/// Usage statistics emitted by cache implementations.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CacheStats {
    /// Total number of token entries stored across all layers.
    pub tokens: usize,
    /// Number of successful retrievals.
    pub hits: usize,
    /// Number of failed retrievals (e.g. out-of-range gather requests).
    pub misses: usize,
    /// Count of segment allocations or reallocations.
    pub segment_churn: usize,
}

/// Abstract cache capable of storing projected keys and values across steps.
pub trait KeyValueCache {
    /// Populate the cache with a prefill pass.
    fn prefill(&mut self, keys: &Tensor, values: &Tensor) -> Result<(), AttentionError>;

    /// Append a single decode step (one token) worth of keys and values for all layers.
    fn append_decode_step(&mut self, keys: &Tensor, values: &Tensor) -> Result<(), AttentionError>;

    /// Gather cached entries for the provided layer and token indices.
    fn gather(
        &mut self,
        layer: usize,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor), AttentionError>;

    /// Current length (number of cached tokens) recorded for the provided layer.
    fn len(&self, layer: usize) -> usize;

    /// Return the latest cache statistics.
    fn stats(&self) -> CacheStats;
}
