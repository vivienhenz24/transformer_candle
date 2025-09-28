//! Configuration options shared by all attention implementations.
//!
//! The [`Config`] struct captures run-time knobs such as dropout that callers
//! can tune without swapping implementations.

use std::env;

/// Selection hints for choosing the attention backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendSelection {
    /// Defer to runtime heuristics when deciding between reference and fused paths.
    Auto,
    /// Force the portable reference implementation.
    ReferenceOnly,
    /// Force the fused implementation when available (falls back to reference otherwise).
    FusedOnly,
}

impl Default for BackendSelection {
    fn default() -> Self {
        Self::Auto
    }
}

/// Controls how intermediate precision is handled during attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionPolicy {
    /// Preserve the input dtype for intermediate work where possible.
    Inherit,
    /// Promote intermediates to `f32` before casting back to the requested dtype.
    ForceF32,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::ForceF32
    }
}

/// Indicates how rotary embeddings should be handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeMode {
    /// Disable RoPE application inside the attention backend.
    Off,
    /// Assume inputs were pre-rotated upstream (e.g. in blocks) and skip internal work.
    Preapply,
    /// Apply RoPE on-the-fly using the configured adapter.
    OnTheFly,
}

impl Default for RopeMode {
    fn default() -> Self {
        Self::OnTheFly
    }
}

/// Key/Value cache behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvConfig {
    /// Whether incremental decode should populate and read from the cache.
    pub enabled: bool,
    /// Optional page size override (defaults to implementation choice when `None`).
    pub page_size: Option<usize>,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            page_size: None,
        }
    }
}

/// Configuration driving attention behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Probability for dropout applied to attention weights during training.
    ///
    /// When `None`, dropout is disabled and the computation is deterministic.
    pub dropout_p: Option<f32>,
    /// Override policy for choosing between reference and fused execution paths.
    pub backend: BackendSelection,
    /// Intermediate precision handling.
    pub precision: PrecisionPolicy,
    /// Whether padding masks should be honoured (set `false` when inputs are already trimmed).
    pub use_padding_mask: bool,
    /// RoPE application mode for this invocation.
    pub rope_mode: RopeMode,
    /// Key/value cache controls.
    pub kv: KvConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dropout_p: None,
            backend: BackendSelection::default(),
            precision: PrecisionPolicy::default(),
            use_padding_mask: true,
            rope_mode: RopeMode::default(),
            kv: KvConfig::default(),
        }
    }
}

impl Config {
    /// Apply environment overrides (e.g. `ATTN_BACKEND=reference`).
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = env::var("ATTN_BACKEND") {
            self.backend = match val.to_ascii_lowercase().as_str() {
                "reference" | "ref" => BackendSelection::ReferenceOnly,
                "fused" => BackendSelection::FusedOnly,
                _ => BackendSelection::Auto,
            };
        }
        if let Ok(val) = env::var("ATTN_PRECISION") {
            self.precision = match val.to_ascii_lowercase().as_str() {
                "inherit" | "input" => PrecisionPolicy::Inherit,
                _ => PrecisionPolicy::ForceF32,
            };
        }
        if let Ok(val) = env::var("ATTN_USE_PADDING_MASK") {
            self.use_padding_mask = matches!(val.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Ok(val) = env::var("ATTN_ROPE_MODE") {
            self.rope_mode = match val.to_ascii_lowercase().as_str() {
                "off" | "disabled" => RopeMode::Off,
                "preapply" | "pre" => RopeMode::Preapply,
                _ => RopeMode::OnTheFly,
            };
        }
        if let Ok(val) = env::var("ATTN_KV_ENABLED") {
            self.kv.enabled = matches!(val.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Ok(val) = env::var("ATTN_KV_PAGE_SIZE") {
            self.kv.page_size = val.parse::<usize>().ok();
        }
        if let Ok(val) = env::var("ATTN_DROPOUT_P") {
            self.dropout_p = val.parse::<f32>().ok();
        }
    }

    /// Construct a configuration honouring environment overrides.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.apply_env_overrides();
        cfg
    }
}
