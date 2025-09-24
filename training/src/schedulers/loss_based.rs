#[derive(Debug, Clone)]
pub struct LossBasedScheduler {
    pub factor: f64,
    pub patience: usize,
}

impl LossBasedScheduler {
    pub fn new(factor: f64, patience: usize) -> Self {
        Self { factor, patience }
    }
}
