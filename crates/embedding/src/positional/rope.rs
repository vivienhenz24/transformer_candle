//! Rotary positional embedding scaffolding.

use candle_core::{bail, DType, Device, DeviceLocation, Result, Tensor};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const SIN_COS_CACHE_CAPACITY: usize = 16;

static SIN_COS_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static SIN_COS_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);

/// Return the current `(hits, misses)` counters for the shared sin/cos cache.
pub fn sin_cos_cache_counters() -> (usize, usize) {
    (
        SIN_COS_CACHE_HITS.load(Ordering::Relaxed),
        SIN_COS_CACHE_MISSES.load(Ordering::Relaxed),
    )
}

/// Reset the shared sin/cos cache counters.
pub fn reset_sin_cos_cache_stats() {
    SIN_COS_CACHE_HITS.store(0, Ordering::Relaxed);
    SIN_COS_CACHE_MISSES.store(0, Ordering::Relaxed);
}

struct SinCosCache {
    capacity: usize,
    order: Vec<String>,
    entries: HashMap<String, (Tensor, Tensor)>,
}

impl SinCosCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: Vec::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
        }
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            if pos + 1 == self.order.len() {
                return;
            }
            let key_owned = self.order.remove(pos);
            self.order.push(key_owned);
        }
    }

    fn get(&mut self, key: &str) -> Option<(Tensor, Tensor)> {
        let value = self.entries.get(key)?;
        let sin = value.0.clone();
        let cos = value.1.clone();
        self.touch(key);
        Some((sin, cos))
    }

    fn insert(&mut self, key: String, value: (Tensor, Tensor)) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), value);
            self.touch(&key);
            return;
        }

        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.first().cloned() {
                self.order.remove(0);
                self.entries.remove(&oldest);
            }
        }

        self.order.push(key.clone());
        self.entries.insert(key, value);
    }
}

fn global_sin_cos_cache() -> &'static Mutex<SinCosCache> {
    static CACHE: OnceLock<Mutex<SinCosCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SinCosCache::new(SIN_COS_CACHE_CAPACITY)))
}

/// Configuration for building rotary positional embeddings.
///
/// Environment overrides (`ROPE_*`) may be layered by callers before constructing this config:
/// - `ROPE_MODE` (`"none" | "ntk" | "pi"`) selects the scaling variant.
/// - `ROPE_THETA` overrides `rope_theta` (parsed as f32).
/// - `ROPE_ROTATE_DIM` overrides `rotate_dim` (parsed as usize; `0` maps to `None`).
/// - `ROPE_ALPHA` supplies the NTK-aware `alpha` parameter when `ROPE_MODE=ntk`.
/// - `ROPE_PI_SCALE` supplies the position interpolation scale when `ROPE_MODE=pi`.
///
/// Implementations should log the resolved `scaling_fingerprint` once per run so downstream logs capture
/// the effective configuration used for reproducibility and debugging.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    /// Per-head dimensionality of the representations being rotated (commonly 64 or 128).
    pub head_dim: usize,
    /// Base angle parameter θ controlling the RoPE frequency spectrum (defaults to 10k).
    pub rope_theta: f32,
    /// Optional override for how many dimensions to rotate; `None` implies the full `head_dim`.
    pub rotate_dim: Option<usize>,
    /// Strategy used to scale RoPE angles for extended context lengths; advanced variants require the `long-context` feature.
    pub scaling: RopeScaling,
}

/// Available scaling strategies for rotary positional embeddings.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RopeScaling {
    /// Baseline RoPE behaviour without any context extension tweaks.
    #[default]
    None,
    /// (requires `long-context`) NTK-aware scaling controlled by α (typically ≥ 1.0) to stabilise longer contexts.
    #[cfg(feature = "long-context")]
    NTKAware { alpha: f32 },
    /// (requires `long-context`) Position interpolation that compresses positions by the provided factor (commonly between 1.0 and 8.0).
    #[cfg(feature = "long-context")]
    PositionInterpolation { scale: f32 },
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 0,
            rope_theta: 10_000.0,
            rotate_dim: None,
            scaling: RopeScaling::default(),
        }
    }
}

/// Build a stable cache key derived from geometry and device inputs.
///
/// Cache entries must be keyed by the tuple `{max_seq_len, rotate_dim, rope_theta, scaling fingerprint, device id}`
/// to ensure compatibility across configurations and hardware targets.
pub fn rope_cache_key(max_seq_len: usize, cfg: &RopeConfig, device: &Device) -> String {
    let rotate_dim_effective = cfg.rotate_dim.unwrap_or(cfg.head_dim);

    let device_id = match device.location() {
        DeviceLocation::Cpu => "cpu".to_owned(),
        DeviceLocation::Cuda { gpu_id } => format!("cuda{gpu_id}"),
        DeviceLocation::Metal { gpu_id } => format!("metal{gpu_id}"),
    };

    let fingerprint = scaling_fingerprint(cfg, max_seq_len);

    format!(
        "seq={};rot={};theta={:.6};dev={};{}",
        max_seq_len, rotate_dim_effective, cfg.rope_theta, device_id, fingerprint
    )
}

/// Retrieve (or lazily build) the per-device sine/cosine tables for RoPE.
///
/// The cache stores **f32** tensors shaped `[max_seq_len, rotate_dim/2]`, with `rotate_dim` defaulting to
/// the full `head_dim` when unspecified. Callers may downcast to bf16/fp16 at the usage site if desired.
/// The underlying cache must be thread-safe and bounded (e.g. LRU) to avoid unbounded memory growth.
pub fn get_sin_cos(max_seq_len: usize, cfg: &RopeConfig, device: &Device) -> (Tensor, Tensor) {
    assert!(max_seq_len > 0, "max_seq_len must be non-zero");

    let rotate_dim = cfg.rotate_dim.unwrap_or(cfg.head_dim);
    assert!(rotate_dim >= 2, "rotate_dim must be at least 2");
    assert!(
        rotate_dim % 2 == 0,
        "rotate_dim must be even to pair dimensions"
    );

    let cache_key = rope_cache_key(max_seq_len, cfg, device);
    let cache = global_sin_cos_cache();
    {
        let mut guard = cache.lock().expect("sin/cos cache lock poisoned");
        if let Some((sin, cos)) = guard.get(&cache_key) {
            SIN_COS_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            log::debug!("rope sin/cos cache hit: {}", cache_key);
            return (sin, cos);
        }
        log::debug!("rope sin/cos cache miss: {}", cache_key);
        SIN_COS_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    }

    let half_dim = rotate_dim / 2;
    let base = cfg.rope_theta as f64;
    let mut inv_freqs = Vec::with_capacity(half_dim);
    for idx in 0..half_dim {
        let exponent = (2 * idx) as f64 / rotate_dim as f64;
        inv_freqs.push(base.powf(-exponent));
    }

    let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);
    let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);

    for pos in 0..max_seq_len {
        let pos_f = pos as f64;
        for &inv_freq in &inv_freqs {
            let angle = pos_f * inv_freq;
            sin_data.push(angle.sin() as f32);
            cos_data.push(angle.cos() as f32);
        }
    }

    let sin = Tensor::from_vec(sin_data, (max_seq_len, half_dim), device)
        .expect("failed to build RoPE sine table");
    let cos = Tensor::from_vec(cos_data, (max_seq_len, half_dim), device)
        .expect("failed to build RoPE cosine table");

    let mut guard = cache.lock().expect("sin/cos cache lock poisoned");
    if let Some((sin_cached, cos_cached)) = guard.get(&cache_key) {
        return (sin_cached, cos_cached);
    }
    guard.insert(cache_key, (sin.clone(), cos.clone()));

    (sin, cos)
}

/// Compute the effective sequence positions after applying the configured scaling strategy.
///
/// * `None` returns the identity positions starting at `pos_start` (i.e. `pos_start + i`).
/// * `PositionInterpolation { scale }` maps positions via `floor((pos_start + i) / scale)` and clamps the result into `[0, train_ctx_len)`.
/// * `NTKAware { alpha }` leaves the positions unchanged but signals downstream frequency adjustments; the base derived from `rope_theta` must stay constant across the decode stream.
///
/// Callers must pass the training context length (`train_ctx_len`) used when fitting the model so scaling modes can honour the intended cap.
pub fn effective_positions(
    pos_start: usize,
    seq_len: usize,
    cfg: &RopeConfig,
    train_ctx_len: usize,
) -> Vec<u32> {
    debug_assert!(seq_len > 0, "sequence length must be non-zero");
    debug_assert!(
        train_ctx_len > 0,
        "training context length must be non-zero"
    );

    #[cfg(feature = "long-context")]
    if let RopeScaling::PositionInterpolation { scale } = cfg.scaling {
        debug_assert!(scale >= 1.0, "position interpolation scale must be >= 1.0");
    }

    let mut positions = Vec::with_capacity(seq_len);

    match cfg.scaling {
        RopeScaling::None => {
            for offset in 0..seq_len {
                positions.push((pos_start + offset) as u32);
            }
        }
        #[cfg(feature = "long-context")]
        RopeScaling::PositionInterpolation { scale } => {
            let cap = train_ctx_len - 1;
            let scale_f64 = scale as f64;
            for offset in 0..seq_len {
                let raw_pos = pos_start + offset;
                let scaled = ((raw_pos as f64) / scale_f64).floor() as usize;
                let clamped = scaled.min(cap);
                positions.push(clamped as u32);
            }
            debug_assert!(
                positions.iter().all(|&p| (p as usize) < train_ctx_len),
                "clamped positions must stay within training context"
            );
        }
        #[cfg(feature = "long-context")]
        RopeScaling::NTKAware { .. } => {
            // NTK-aware scaling keeps positional indices unchanged; NTK adjustments modify base frequencies.
            for offset in 0..seq_len {
                positions.push((pos_start + offset) as u32);
            }
        }
    }

    positions
}

/// Produce a stable fingerprint describing the scaling behaviour for cache keying.
///
/// The fingerprint must uniquely capture the scaling mode, any associated parameters, and the provided `train_ctx_len`
/// to prevent cache collisions when switching context regimes. This string is incorporated into cache keys alongside
/// geometry/device details to avoid cross-contamination between scaling strategies.
///
/// Environment variables parsed by the caller (`ROPE_MODE`, `ROPE_THETA`, `ROPE_ROTATE_DIM`, `ROPE_ALPHA`, `ROPE_PI_SCALE`) feed into
/// this fingerprint. Emit the fingerprint once during start-up so logs capture the effective configuration used.
pub fn scaling_fingerprint(cfg: &RopeConfig, train_ctx_len: usize) -> String {
    let mode = match cfg.scaling {
        RopeScaling::None => "none",
        #[cfg(feature = "long-context")]
        RopeScaling::PositionInterpolation { .. } => "pi",
        #[cfg(feature = "long-context")]
        RopeScaling::NTKAware { .. } => "ntk",
    };

    let rotate_segment = cfg
        .rotate_dim
        .map(|d| d.to_string())
        .unwrap_or_else(|| "all".to_owned());

    #[cfg_attr(not(feature = "long-context"), allow(unused_mut))]
    let mut fingerprint = format!(
        "mode={mode};theta={:.6};rot={};train_ctx={}",
        cfg.rope_theta, rotate_segment, train_ctx_len
    );

    match cfg.scaling {
        RopeScaling::None => {}
        #[cfg(feature = "long-context")]
        RopeScaling::PositionInterpolation { scale } => {
            fingerprint.push_str(&format!(";pi.scale={:.6}", scale));
        }
        #[cfg(feature = "long-context")]
        RopeScaling::NTKAware { alpha } => {
            fingerprint.push_str(&format!(";ntk.alpha={:.6}", alpha));
        }
    }

    fingerprint
}

/// Apply rotary embeddings to query/key tensors with explicit shape guarantees.
///
/// * All tensors must be contiguous in memory and shaped `[batch, n_heads, seq_len, head_dim]`.
/// * Only the leading `rotate_dim` components (defaulting to `head_dim`) are rotated; the remainder is copied through untouched.
/// * The rotation operates on `[seq_len, rotate_dim]` blocks without per-token loops, avoiding layout-breaking transposes or per-call allocations.
/// * Sine/cosine tables are f32 tensors shaped `[max_seq_len, rotate_dim/2]`; they remain in f32 while `q`/`k` may be bf16/fp16/f32, and the outputs mirror the dtype of the inputs.
///
/// Example: with `batch=2`, `n_heads=4`, `seq_len=16`, `head_dim=64`, `rotate_dim=32`, the function rotates the first 32 features of each head across the 16 positions while leaving the remaining 32 features untouched.
pub fn apply_rope_to_qk(
    q: &Tensor,
    k: &Tensor,
    pos_start: usize,
    cfg: &RopeConfig,
    sin: &Tensor,
    cos: &Tensor,
) -> Result<(Tensor, Tensor)> {
    debug_assert!(q.is_contiguous(), "query tensor must be contiguous");
    debug_assert!(k.is_contiguous(), "key tensor must be contiguous");

    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    let (kb, kh, kt, kd) = k.dims4()?;
    if (kb, kh, kt, kd) != (batch, heads, seq_len, head_dim) {
        bail!(
            "q/k shape mismatch: q={:?} k={:?}",
            q.shape().dims(),
            k.shape().dims()
        );
    }

    let rotate_dim = cfg.rotate_dim.unwrap_or(cfg.head_dim);
    if rotate_dim == 0 || rotate_dim > head_dim {
        bail!(
            "invalid rotate_dim {} for head_dim {}",
            rotate_dim,
            head_dim
        );
    }
    debug_assert!(head_dim >= rotate_dim, "head_dim must be >= rotate_dim");
    debug_assert_eq!(rotate_dim % 2, 0, "rotate_dim must be even");

    let half_dim = rotate_dim / 2;

    let (sin_rows, sin_dim) = sin.dims2()?;
    let (cos_rows, cos_dim) = cos.dims2()?;
    debug_assert!(
        sin_rows >= pos_start + seq_len,
        "sin table too short for positions"
    );
    debug_assert!(
        cos_rows >= pos_start + seq_len,
        "cos table too short for positions"
    );
    debug_assert_eq!(sin_dim, half_dim, "sin table dimension mismatch");
    debug_assert_eq!(cos_dim, half_dim, "cos table dimension mismatch");

    let sin_slice = sin.narrow(0, pos_start, seq_len)?;
    let cos_slice = cos.narrow(0, pos_start, seq_len)?;
    let sin_b = sin_slice
        .reshape((1, 1, seq_len, half_dim))?
        .broadcast_as((batch, heads, seq_len, half_dim))?;
    let cos_b = cos_slice
        .reshape((1, 1, seq_len, half_dim))?
        .broadcast_as((batch, heads, seq_len, half_dim))?;

    let apply_one = |tensor: &Tensor| -> Result<Tensor> {
        let main = tensor.narrow(3, 0, rotate_dim)?;
        let tail_dim = head_dim - rotate_dim;
        let dtype = tensor.dtype();

        let main_f32 = main.to_dtype(DType::F32)?;
        let main_pairs = main_f32.reshape((batch, heads, seq_len, half_dim, 2))?;
        let chunks = main_pairs.chunk(2, 4)?;
        let even = chunks[0].squeeze(4)?;
        let odd = chunks[1].squeeze(4)?;

        let even_cos = even.mul(&cos_b)?;
        let odd_sin = odd.mul(&sin_b)?;
        let even_sin = even.mul(&sin_b)?;
        let odd_cos = odd.mul(&cos_b)?;

        let rotated_even = even_cos.sub(&odd_sin)?;
        let rotated_odd = odd_cos.add(&even_sin)?;

        let rotated_even = rotated_even.unsqueeze(4)?;
        let rotated_odd = rotated_odd.unsqueeze(4)?;
        let rotated = Tensor::cat(&[&rotated_even, &rotated_odd], 4)?
            .reshape((batch, heads, seq_len, rotate_dim))?
            .to_dtype(dtype)?;

        if tail_dim == 0 {
            Ok(rotated)
        } else {
            let tail = tensor.narrow(3, rotate_dim, tail_dim)?;
            Tensor::cat(&[&rotated, &tail], 3)
        }
    };

    let q_rot = apply_one(q)?;
    let k_rot = apply_one(k)?;

    Ok((q_rot, k_rot))
}

/// Precomputed cosine/sine tables for RoPE.
#[derive(Debug, Clone)]
pub struct RopeCache {
    config: RopeConfig,
    device: Device,
    max_cached: usize,
}

impl PartialEq for RopeCache {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.device.same_device(&other.device)
            && self.max_cached == other.max_cached
    }
}

impl RopeCache {
    fn new(config: RopeConfig, device: Device) -> Self {
        Self {
            config,
            device,
            max_cached: 0,
        }
    }

    /// Prepare the cache for a maximum sequence length.
    pub fn precompute(&mut self, max_positions: usize, head_dim: usize) -> Result<()> {
        let rotate_dim = self.config.rotate_dim.unwrap_or(self.config.head_dim);
        debug_assert!(
            head_dim >= rotate_dim,
            "head_dim must accommodate rotate_dim"
        );
        let _ = get_sin_cos(max_positions, &self.config, &self.device);
        if max_positions > self.max_cached {
            self.max_cached = max_positions;
        }
        Ok(())
    }

    /// Fetch cached cos/sin angles for the provided positions.
    pub fn slice(&self, positions: &Tensor) -> Result<Tensor> {
        let positions_vec = positions.to_vec1::<u32>()?;
        if positions_vec.is_empty() {
            bail!("positions tensor must not be empty");
        }
        let max_pos = positions_vec.iter().copied().max().unwrap() as usize;
        let required = max_pos + 1;
        let fetch_len = required.max(self.max_cached.max(1));
        let (sin, cos) = get_sin_cos(fetch_len, &self.config, &self.device);
        let len = positions_vec.len();
        let indices = Tensor::from_vec(positions_vec, (len,), &self.device)?;
        let sin_rows = sin.index_select(&indices, 0)?;
        let cos_rows = cos.index_select(&indices, 0)?;
        let sin_rows = sin_rows.unsqueeze(0)?;
        let cos_rows = cos_rows.unsqueeze(0)?;
        Tensor::cat(&[&sin_rows, &cos_rows], 0)
    }
}

/// Rotary positional embedding helper exposing construction and application entry points.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    config: RopeConfig,
}

impl Rope {
    /// Construct the rotary embedding helper from a configuration.
    pub fn new(config: RopeConfig) -> Result<Self> {
        if config.head_dim == 0 {
            bail!("head_dim must be non-zero");
        }
        if let Some(rotate_dim) = config.rotate_dim {
            if rotate_dim == 0 || rotate_dim > config.head_dim {
                bail!(
                    "invalid rotate_dim {} for head_dim {}",
                    rotate_dim,
                    config.head_dim
                );
            }
            if rotate_dim % 2 != 0 {
                bail!("rotate_dim must be even");
            }
        }
        Ok(Self { config })
    }

    /// Apply rotary positional embeddings to query/key tensors.
    pub fn apply_rotary_embeddings(
        &self,
        query: &Tensor,
        key: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if !query.device().same_device(key.device()) {
            bail!("query and key must live on the same device");
        }

        let (_b, _h, seq_len, head_dim) = query.dims4()?;
        let rotate_dim = self.config.rotate_dim.unwrap_or(self.config.head_dim);
        if head_dim < rotate_dim {
            bail!("query head_dim {} < rotate_dim {}", head_dim, rotate_dim);
        }

        if positions.rank() != 1 {
            bail!("positions tensor must be 1-D");
        }
        let pos_vec = positions.to_vec1::<u32>()?;
        if pos_vec.len() != seq_len {
            bail!(
                "positions length {} does not match sequence length {}",
                pos_vec.len(),
                seq_len
            );
        }
        if pos_vec.is_empty() {
            bail!("positions tensor must not be empty");
        }

        let pos_start = *pos_vec.iter().min().unwrap() as usize;
        for (offset, &position) in pos_vec.iter().enumerate() {
            let expected = pos_start + offset;
            if position as usize != expected {
                bail!(
                    "positions must form a contiguous range starting at {}",
                    pos_start
                );
            }
        }

        let needed = pos_start + seq_len;
        let (sin, cos) = get_sin_cos(needed, &self.config, query.device());
        apply_rope_to_qk(query, key, pos_start, &self.config, &sin, &cos)
    }

    /// Access a cache of angles to reuse across steps.
    pub fn cached_angles(&self) -> Result<RopeCache> {
        Ok(RopeCache::new(self.config.clone(), Device::Cpu))
    }
}
