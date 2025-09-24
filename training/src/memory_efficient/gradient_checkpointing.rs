#[derive(Debug, Clone)]
pub struct GradientCheckpointingConfig {
    pub segments: usize,
}

impl Default for GradientCheckpointingConfig {
    fn default() -> Self {
        Self { segments: 4 }
    }
}
