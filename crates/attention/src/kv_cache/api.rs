//! Public traits describing key/value cache behaviour.

use candle_core::Tensor;

/// Abstract cache capable of storing projected keys and values across steps.
pub trait KeyValueCache {
    /// Append a new slice of keys and values to the cache.
    fn append(&mut self, k: &Tensor, v: &Tensor);

    /// Retrieve cached projections for the supplied sequence position.
    fn view(&self, batch_index: usize) -> Option<(Tensor, Tensor)>;
}
