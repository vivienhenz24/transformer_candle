//! Error types emitted by attention implementations.

/// Attention-specific error category.
#[derive(Debug)]
pub enum AttentionError {
    /// The supplied tensor shapes do not align with the documented contract.
    InvalidShape { context: &'static str },
    /// The kernel does not support the requested data type.
    UnsupportedDType { requested: &'static str },
    /// A backend-specific failure propagated to the caller.
    Backend { message: String },
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionError::InvalidShape { context } => {
                write!(f, "invalid tensor shape for {context}")
            }
            AttentionError::UnsupportedDType { requested } => {
                write!(f, "unsupported dtype {requested}")
            }
            AttentionError::Backend { message } => f.write_str(message),
        }
    }
}

impl std::error::Error for AttentionError {}
