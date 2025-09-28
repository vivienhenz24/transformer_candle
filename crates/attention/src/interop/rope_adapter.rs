//! Integrations with rotary position embedding utilities.
//!
//! The adapter exposes helpers to (a) fetch sine/cosine tables covering a
//! contiguous span of positions, and (b) apply rotary embeddings to tensors
//! laid out as `[batch, num_heads, seq_len, head_dim]`. It defers to the
//! embedding crate for numerical kernels while reusing the shared sin/cos LRU
//! cache.

use candle_core::{bail, Device, Result, Tensor};
use embedding::positional::rope::{apply_rope_to_qk, get_sin_cos, RopeConfig};

#[derive(Debug, Clone)]
struct CachedTables {
    coverage: usize,
    sin: Tensor,
    cos: Tensor,
}

/// Adapter that bridges attention kernels with the positional embedding crate.
#[derive(Debug, Clone)]
pub struct RopeAdapter {
    config: RopeConfig,
    device: Device,
    tables: Option<CachedTables>,
}

impl RopeAdapter {
    /// Create a new adapter bound to the provided configuration and device.
    pub fn new(config: RopeConfig, device: Device) -> Self {
        Self {
            config,
            device,
            tables: None,
        }
    }

    /// Retrieve sine/cosine tables covering the contiguous span starting at
    /// `pos_start` and extending `seq_len` positions.
    ///
    /// # Preconditions
    /// - `seq_len` must be non-zero.
    /// - `pos_start + seq_len` must fit within the configured rotary geometry.
    pub fn sin_cos_slice(&mut self, pos_start: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if seq_len == 0 {
            bail!("seq_len must be non-zero");
        }
        self.ensure_tables(pos_start, seq_len)?;
        let cached = self.tables.as_ref().expect("tables present after ensure");
        let sin_slice = cached.sin.narrow(0, pos_start, seq_len)?;
        let cos_slice = cached.cos.narrow(0, pos_start, seq_len)?;
        Ok((sin_slice, cos_slice))
    }

    /// Apply rotary embeddings to query/key tensors shaped
    /// `[batch, num_heads, seq_len, head_dim]`.
    ///
    /// # Preconditions
    /// - `q` and `k` must reside on the adapter's device and share dtype/layout.
    /// - Their sequence positions must be contiguous, covering
    ///   `[pos_start, pos_start + seq_len)`.
    /// - Tensors must be contiguous in memory; violations surface as errors.
    pub fn apply(&mut self, q: &Tensor, k: &Tensor, pos_start: usize) -> Result<(Tensor, Tensor)> {
        if !self.device.same_device(q.device()) || !self.device.same_device(k.device()) {
            bail!("q and k must reside on the adapter's device");
        }
        if !q.is_contiguous() {
            bail!("q must be contiguous");
        }
        if !k.is_contiguous() {
            bail!("k must be contiguous");
        }

        let (_b, _h, seq_len, _d) = q.dims4()?;
        if seq_len == 0 {
            bail!("sequence length must be non-zero");
        }
        self.ensure_tables(pos_start, seq_len)?;

        let tables = self
            .tables
            .as_ref()
            .expect("tables guaranteed after ensure");
        apply_rope_to_qk(q, k, pos_start, &self.config, &tables.sin, &tables.cos)
    }

    fn ensure_tables(&mut self, pos_start: usize, seq_len: usize) -> Result<()> {
        let required = pos_start + seq_len;
        let current = self.tables.as_ref().map(|c| c.coverage).unwrap_or(0);
        if current >= required {
            return Ok(());
        }

        let needed = required.max(1);
        let (sin, cos) = get_sin_cos(needed, &self.config, &self.device);
        self.tables = Some(CachedTables {
            coverage: needed,
            sin,
            cos,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use embedding::positional::rope::RopeScaling;

    fn arange_tensor(
        total: usize,
        shape: (usize, usize, usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let data: Vec<f32> = (0..total).map(|v| v as f32).collect();
        Tensor::from_vec(data, shape, device)
    }

    fn allclose(a: &Tensor, b: &Tensor, tol: f32) -> Result<bool> {
        let diff = a.sub(b)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
        let max = diff.into_iter().fold(0.0_f32, |acc, val| acc.max(val));
        Ok(max <= tol)
    }

    #[test]
    fn adapter_matches_direct_application() -> Result<()> {
        let device = Device::Cpu;
        let cfg = RopeConfig {
            head_dim: 8,
            rope_theta: 10_000.0,
            rotate_dim: None,
            scaling: RopeScaling::None,
        };

        let batch = 2;
        let heads = 3;
        let seq_len = 4;
        let head_dim = cfg.head_dim;
        let total = batch * heads * seq_len * head_dim;
        let q = arange_tensor(total, (batch, heads, seq_len, head_dim), &device)?;
        let k = arange_tensor(total, (batch, heads, seq_len, head_dim), &device)?;

        let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
        let (direct_q, direct_k) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos)?;

        let mut adapter = RopeAdapter::new(cfg.clone(), device.clone());
        let (adapt_q, adapt_k) = adapter.apply(&q, &k, 0)?;

        assert!(allclose(&direct_q, &adapt_q, 1e-5)?);
        assert!(allclose(&direct_k, &adapt_k, 1e-5)?);
        Ok(())
    }

    #[test]
    fn adapter_handles_offset_positions() -> Result<()> {
        let device = Device::Cpu;
        let cfg = RopeConfig {
            head_dim: 8,
            rope_theta: 10_000.0,
            rotate_dim: None,
            scaling: RopeScaling::None,
        };

        let batch = 1;
        let heads = 2;
        let seq_len = 3;
        let head_dim = cfg.head_dim;
        let total = batch * heads * seq_len * head_dim;
        let q = arange_tensor(total, (batch, heads, seq_len, head_dim), &device)?;
        let k = arange_tensor(total, (batch, heads, seq_len, head_dim), &device)?;

        let pos_start = 5;
        let (sin, cos) = get_sin_cos(pos_start + seq_len, &cfg, &device);
        let (direct_q, direct_k) = apply_rope_to_qk(&q, &k, pos_start, &cfg, &sin, &cos)?;

        let mut adapter = RopeAdapter::new(cfg.clone(), device.clone());
        let (adapt_q, adapt_k) = adapter.apply(&q, &k, pos_start)?;

        assert!(allclose(&direct_q, &adapt_q, 1e-5)?);
        assert!(allclose(&direct_k, &adapt_k, 1e-5)?);

        // sin/cos slices align with direct narrow.
        let (slice_sin, slice_cos) = adapter.sin_cos_slice(pos_start, seq_len)?;
        let direct_slice_sin = sin.narrow(0, pos_start, seq_len)?;
        let direct_slice_cos = cos.narrow(0, pos_start, seq_len)?;
        assert!(allclose(&slice_sin, &direct_slice_sin, 1e-5)?);
        assert!(allclose(&slice_cos, &direct_slice_cos, 1e-5)?);
        Ok(())
    }

    #[test]
    fn adapter_extends_cached_tables_on_demand() -> Result<()> {
        let device = Device::Cpu;
        let cfg = RopeConfig {
            head_dim: 8,
            rope_theta: 10_000.0,
            rotate_dim: None,
            scaling: RopeScaling::None,
        };

        let mut adapter = RopeAdapter::new(cfg.clone(), device.clone());
        let (sin_short, _) = adapter.sin_cos_slice(0, 2)?;
        assert_eq!(sin_short.dims(), &[2, cfg.head_dim / 2]);
        let coverage_initial = adapter.tables.as_ref().map(|c| c.coverage).unwrap();
        assert!(coverage_initial >= 2);

        // Request a longer span that should grow the cached tables.
        let (sin_long, _) = adapter.sin_cos_slice(0, 8)?;
        assert_eq!(sin_long.dims(), &[8, cfg.head_dim / 2]);
        let coverage_after = adapter.tables.as_ref().map(|c| c.coverage).unwrap();
        assert!(coverage_after >= 8);

        // Parity for a longer apply.
        let batch = 1;
        let heads = 1;
        let seq_len = 8;
        let head_dim = cfg.head_dim;
        let total = batch * heads * seq_len * head_dim;
        let q = arange_tensor(total, (batch, heads, seq_len, head_dim), &device)?;
        let k = arange_tensor(total, (batch, heads, seq_len, head_dim), &device)?;
        let (sin, cos) = get_sin_cos(seq_len, &cfg, &device);
        let (direct_q, direct_k) = apply_rope_to_qk(&q, &k, 0, &cfg, &sin, &cos)?;
        let (adapt_q, adapt_k) = adapter.apply(&q, &k, 0)?;
        assert!(allclose(&direct_q, &adapt_q, 1e-5)?);
        assert!(allclose(&direct_k, &adapt_k, 1e-5)?);
        Ok(())
    }
}
