#[derive(Debug, Clone)]
pub struct ActivationCheckpointingConfig {
    pub enabled: bool,
}

impl Default for ActivationCheckpointingConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}
