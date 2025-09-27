//! Optimised fused attention kernels.
//!
//! These implementations are built for hardware-specific acceleration and are
//! only compiled when the `fused` feature is enabled.

/// Placeholder marker to ensure the module compiles until fused kernels land.
pub struct FusedPlaceholder;
