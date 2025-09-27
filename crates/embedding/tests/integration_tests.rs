//! Integration smoke tests for the embedding crate.

use embedding::positional::rope::{RopeConfig, RopeScaling};

#[test]
fn positional_module_exposes_defaults() {
    let config = RopeConfig::default();
    assert_eq!(config.head_dim, 0);
    assert_eq!(config.rope_theta, 10_000.0);
    assert_eq!(config.rotate_dim, None);
    assert!(matches!(config.scaling, RopeScaling::None));
}
