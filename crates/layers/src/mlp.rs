//! Position-wise feed-forward blocks built on top of projections and activations.
//!
//! MLPs operate on hidden states shaped `(batch, seq, hidden)` and return the
//! same layout. Intermediate projections expand the hidden dimension to
//! `config.intermediate_size`, apply an activation, then contract back to the
//! model hidden size. Gated variants expect fused projections in
//! `(batch, seq, 2 * intermediate_size)` before splitting along the last axis.

use candle_core::{Result, Tensor};

use crate::{activations::ActivationKind, dtypes::PrecisionPolicy};

/// Configuration shared by transformer feed-forward networks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedForwardConfig {
    /// Model hidden size.
    pub hidden_size: usize,
    /// Width of the activation space.
    pub intermediate_size: usize,
    /// Activation applied between projections.
    pub activation: ActivationKind,
    /// Whether the MLP uses a gated projection (e.g. SwiGLU).
    pub gated: bool,
}

impl FeedForwardConfig {
    /// Creates a standard two-projection MLP configuration.
    pub fn new(hidden_size: usize, intermediate_size: usize, activation: ActivationKind) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation,
            gated: false,
        }
    }
}

/// Shared interface for feed-forward stacks.
pub trait FeedForwardLayer: Send + Sync {
    /// Configuration metadata used during block assembly.
    fn config(&self) -> &FeedForwardConfig;

    /// Performs the forward pass through the MLP.
    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor>;
}
