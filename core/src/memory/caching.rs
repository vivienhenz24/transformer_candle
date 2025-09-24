use std::collections::VecDeque;

use candle_core::Tensor;

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
}

#[derive(Debug)]
pub struct AttentionCache {
    capacity: usize,
    queue: VecDeque<Tensor>,
    stats: CacheStats,
}

impl AttentionCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            queue: VecDeque::with_capacity(capacity),
            stats: CacheStats::default(),
        }
    }

    pub fn push(&mut self, tensor: Tensor) {
        if self.queue.len() == self.capacity {
            self.queue.pop_front();
        }
        self.queue.push_back(tensor);
    }

    pub fn latest(&mut self) -> Option<Tensor> {
        if let Some(value) = self.queue.back() {
            self.stats.hits += 1;
            Some(value.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
}
