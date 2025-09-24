#[derive(Debug, Clone)]
pub struct AdaptiveLearningRate {
    pub base: f64,
    pub min: f64,
}

impl AdaptiveLearningRate {
    pub fn scale(&self, loss: f32) -> f64 {
        let scale = (loss as f64).clamp(0.1, 2.0);
        (self.base / scale).max(self.min)
    }
}
