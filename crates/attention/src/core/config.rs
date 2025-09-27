//! Configuration options shared by all attention implementations.
//!
//! The [`Config`] struct captures run-time knobs such as dropout that callers
//! can tune without swapping implementations.

/// Configuration driving attention behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Probability for dropout applied to attention weights during training.
    ///
    /// When `None`, dropout is disabled and the computation is deterministic.
    pub dropout_p: Option<f32>,
}

impl Default for Config {
    fn default() -> Self {
        Self { dropout_p: None }
    }
}
