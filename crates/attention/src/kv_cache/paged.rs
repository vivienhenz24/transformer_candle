//! Paged key/value cache implementation.
//!
//! The cache stores keys and values with layout `[batch, layer, head, position, dim]`
//! and organises storage into fixed-capacity pages to avoid reallocating a single
//! monolithic buffer for long contexts. Prefill writes fill pages sequentially,
//! after which decode-time appends extend the cache one position at a time.

use std::cell::Cell;
use std::cmp::min;
use std::marker::PhantomData;

use candle_core::{DType, Device, Tensor};

use crate::core::AttentionError;
use crate::kv_cache::api::{CacheStats, KeyValueCache};

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub batch: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub dtype: DType,
    pub device: Device,
}

impl CacheConfig {
    pub fn new(
        batch: usize,
        layers: usize,
        heads: usize,
        head_dim: usize,
        page_size: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, AttentionError> {
        if page_size == 0 {
            return Err(AttentionError::InvalidShape {
                context: "page_size must be greater than zero".to_string(),
            });
        }
        let cfg = Self {
            batch,
            layers,
            heads,
            head_dim,
            page_size,
            dtype,
            device,
        };
        log::info!(
            "kv-cache init: batch={} layers={} heads={} head_dim={} page_size={} dtype={:?}",
            cfg.batch, cfg.layers, cfg.heads, cfg.head_dim, cfg.page_size, cfg.dtype
        );
        Ok(cfg)
    }
}

#[derive(Debug)]
struct CachePage {
    keys: Tensor,
    values: Tensor,
    len: usize,
    capacity: usize,
}

impl CachePage {
    fn new(config: &CacheConfig) -> Result<Self, AttentionError> {
        let shape = (
            config.batch,
            config.heads,
            config.page_size,
            config.head_dim,
        );
        let keys = Tensor::zeros(shape, config.dtype, &config.device).map_err(to_backend_err)?;
        let values = Tensor::zeros(shape, config.dtype, &config.device).map_err(to_backend_err)?;
        Ok(Self {
            keys,
            values,
            len: 0,
            capacity: config.page_size,
        })
    }

    fn remaining(&self) -> usize {
        self.capacity - self.len
    }

    fn store_chunk(
        &mut self,
        offset: usize,
        data_keys: &Tensor,
        data_values: &Tensor,
    ) -> Result<(), AttentionError> {
        let chunk = data_keys.dims()[2];
        self.keys = self
            .keys
            .slice_assign(
                &[
                    0..self.keys.dims()[0],
                    0..self.keys.dims()[1],
                    offset..offset + chunk,
                    0..self.keys.dims()[3],
                ],
                data_keys,
            )
            .map_err(to_backend_err)?;
        self.values = self
            .values
            .slice_assign(
                &[
                    0..self.values.dims()[0],
                    0..self.values.dims()[1],
                    offset..offset + chunk,
                    0..self.values.dims()[3],
                ],
                data_values,
            )
            .map_err(to_backend_err)?;
        self.len = offset + chunk;
        Ok(())
    }

    fn slice(&self, offset: usize, len: usize) -> Result<(Tensor, Tensor), AttentionError> {
        let keys = self.keys.narrow(2, offset, len).map_err(to_backend_err)?;
        let values = self.values.narrow(2, offset, len).map_err(to_backend_err)?;
        Ok((keys, values))
    }
}

#[derive(Debug)]
struct LayerCache {
    pages: Vec<CachePage>,
    len: usize,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
        }
    }
}

#[derive(Debug)]
pub struct PagedKeyValueCache {
    config: CacheConfig,
    layers: Vec<LayerCache>,
    stats: CacheStats,
    _not_sync: PhantomData<Cell<()>>,
}

impl PagedKeyValueCache {
    pub fn new(config: CacheConfig) -> Self {
        let layers = (0..config.layers).map(|_| LayerCache::new()).collect();
        Self {
            config,
            layers,
            stats: CacheStats::default(),
            _not_sync: PhantomData,
        }
    }

    fn reset(&mut self) {
        self.layers.iter_mut().for_each(|layer| {
            layer.pages.clear();
            layer.len = 0;
        });
        self.stats = CacheStats::default();
    }

    fn ensure_page(&mut self, layer: usize) -> Result<&mut CachePage, AttentionError> {
        if self.layers[layer]
            .pages
            .last()
            .map(|page| page.remaining() > 0)
            .unwrap_or(false)
        {
            return Ok(self.layers[layer].pages.last_mut().expect("page exists"));
        }
        let new_page = CachePage::new(&self.config)?;
        self.stats.segment_churn += 1;
        self.layers[layer].pages.push(new_page);
        Ok(self.layers[layer]
            .pages
            .last_mut()
            .expect("page freshly inserted"))
    }

    fn store_chunk(
        &mut self,
        layer: usize,
        keys: Tensor,
        values: Tensor,
    ) -> Result<(), AttentionError> {
        let total = keys.dims()[2];
        let mut remaining = total;
        let mut offset = 0;
        while remaining > 0 {
            let page = self.ensure_page(layer)?;
            let write = min(page.remaining(), remaining);
            let keys_chunk = keys.narrow(2, offset, write).map_err(to_backend_err)?;
            let values_chunk = values.narrow(2, offset, write).map_err(to_backend_err)?;
            page.store_chunk(page.len, &keys_chunk, &values_chunk)?;
            remaining -= write;
            offset += write;
        }
        self.layers[layer].len += total;
        Ok(())
    }
}

impl KeyValueCache for PagedKeyValueCache {
    fn prefill(&mut self, keys: &Tensor, values: &Tensor) -> Result<(), AttentionError> {
        self.reset();

        let dims = keys.dims();
        if dims.len() != 5 {
            return Err(AttentionError::InvalidShape {
                context: format!(
                    "keys must have shape [batch, layer, head, position, dim], got {:?}",
                    dims
                ),
            });
        }
        let (batch, layers, heads, positions, dim) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        if batch != self.config.batch
            || layers != self.config.layers
            || heads != self.config.heads
            || dim != self.config.head_dim
        {
            return Err(AttentionError::InvalidShape {
                context: format!(
                    "keys shape mismatch: expected [{}, {}, {}, position, {}], got {:?}",
                    self.config.batch,
                    self.config.layers,
                    self.config.heads,
                    self.config.head_dim,
                    dims
                ),
            });
        }
        if values.dims() != dims {
            return Err(AttentionError::InvalidShape {
                context: "values shape must match keys shape".to_string(),
            });
        }
        if keys.dtype() != self.config.dtype || values.dtype() != self.config.dtype {
            return Err(AttentionError::UnsupportedDType {
                requested: format!("keys/values must use {:?}", self.config.dtype),
            });
        }

        if positions == 0 {
            return Ok(());
        }

        for layer in 0..self.config.layers {
            let layer_keys = keys
                .narrow(1, layer, 1)
                .map_err(to_backend_err)?
                .squeeze(1)
                .map_err(to_backend_err)?;
            let layer_values = values
                .narrow(1, layer, 1)
                .map_err(to_backend_err)?
                .squeeze(1)
                .map_err(to_backend_err)?;

            let mut start = 0;
            while start < positions {
                let chunk = min(self.config.page_size, positions - start);
                let keys_chunk = layer_keys.narrow(2, start, chunk).map_err(to_backend_err)?;
                let values_chunk = layer_values
                    .narrow(2, start, chunk)
                    .map_err(to_backend_err)?;
                self.store_chunk(layer, keys_chunk, values_chunk)?;
                start += chunk;
            }
        }

        self.stats.tokens = positions * self.config.layers;
        Ok(())
    }

    fn append_decode_step(&mut self, keys: &Tensor, values: &Tensor) -> Result<(), AttentionError> {
        let dims = keys.dims();
        if dims.len() != 4 {
            return Err(AttentionError::InvalidShape {
                context: format!(
                    "decode keys must have shape [batch, layer, head, dim], got {:?}",
                    dims
                ),
            });
        }
        let (batch, layers, heads, dim) = (dims[0], dims[1], dims[2], dims[3]);
        if batch != self.config.batch
            || layers != self.config.layers
            || heads != self.config.heads
            || dim != self.config.head_dim
        {
            return Err(AttentionError::InvalidShape {
                context: format!(
                    "decode keys shape mismatch: expected [{}, {}, {}, {}], got {:?}",
                    self.config.batch,
                    self.config.layers,
                    self.config.heads,
                    self.config.head_dim,
                    dims
                ),
            });
        }
        if values.dims() != dims {
            return Err(AttentionError::InvalidShape {
                context: "decode values shape must match keys".to_string(),
            });
        }
        if keys.dtype() != self.config.dtype || values.dtype() != self.config.dtype {
            return Err(AttentionError::UnsupportedDType {
                requested: format!("decode tensors must use {:?}", self.config.dtype),
            });
        }

        for layer in 0..self.config.layers {
            let key_step = keys
                .narrow(1, layer, 1)
                .map_err(to_backend_err)?
                .squeeze(1)
                .map_err(to_backend_err)?
                .unsqueeze(2)
                .map_err(to_backend_err)?;
            let value_step = values
                .narrow(1, layer, 1)
                .map_err(to_backend_err)?
                .squeeze(1)
                .map_err(to_backend_err)?
                .unsqueeze(2)
                .map_err(to_backend_err)?;
            self.store_chunk(layer, key_step, value_step)?;
        }

        self.stats.tokens += self.config.layers;
        Ok(())
    }

    fn gather(
        &mut self,
        layer: usize,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor), AttentionError> {
        if layer >= self.config.layers {
            return Err(AttentionError::InvalidShape {
                context: format!("layer index {layer} out of range"),
            });
        }
        let layer_cache = &self.layers[layer];
        let len = layer_cache.len;
        if positions.iter().any(|&p| p >= len) {
            self.stats.misses += 1;
            return Err(AttentionError::InvalidShape {
                context: format!("gather position exceeds length {len}"),
            });
        }
        if positions.is_empty() {
            let empty = Tensor::zeros(
                (
                    self.config.batch,
                    self.config.heads,
                    0usize,
                    self.config.head_dim,
                ),
                self.config.dtype,
                &self.config.device,
            )
            .map_err(to_backend_err)?;
            return Ok((empty.clone(), empty));
        }

        let mut key_slices = Vec::with_capacity(positions.len());
        let mut value_slices = Vec::with_capacity(positions.len());
        for &pos in positions {
            let page_idx = pos / self.config.page_size;
            let offset = pos % self.config.page_size;
            let page = &layer_cache.pages[page_idx];
            let (keys, values) = page.slice(offset, 1)?;
            key_slices.push(keys);
            value_slices.push(values);
        }

        let key_refs: Vec<&Tensor> = key_slices.iter().collect();
        let value_refs: Vec<&Tensor> = value_slices.iter().collect();
        let keys = Tensor::cat(&key_refs, 2).map_err(to_backend_err)?;
        let values = Tensor::cat(&value_refs, 2).map_err(to_backend_err)?;
        self.stats.hits += positions.len();
        Ok((keys, values))
    }

    fn len(&self, layer: usize) -> usize {
        self.layers.get(layer).map(|l| l.len).unwrap_or(0)
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

fn to_backend_err(err: candle_core::Error) -> AttentionError {
    AttentionError::Backend {
        message: err.to_string(),
    }
}
