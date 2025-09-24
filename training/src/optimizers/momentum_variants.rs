#[derive(Debug, Clone)]
pub struct MomentumSchedule {
    pub beta1: f64,
    pub beta2: f64,
}

impl Default for MomentumSchedule {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.95,
        }
    }
}
