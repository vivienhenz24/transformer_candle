#[derive(Debug, Clone)]
pub struct SmartDropoutConfig {
    pub base_prob: f32,
}

impl Default for SmartDropoutConfig {
    fn default() -> Self {
        Self { base_prob: 0.1 }
    }
}
