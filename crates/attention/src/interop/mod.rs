//! Interoperability helpers bridging attention with upstream components.

pub mod rope_adapter;
pub mod types;

pub use rope_adapter::RopeAdapter;
pub use types::{AttentionDType, AttentionShape};
