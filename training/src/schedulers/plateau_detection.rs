#[derive(Debug, Clone)]
pub struct PlateauDetector {
    pub threshold: f32,
    pub patience: usize,
}

impl PlateauDetector {
    pub fn triggered(&self, history: &[f32]) -> bool {
        if history.len() < self.patience {
            return false;
        }
        let recent = &history[history.len() - self.patience..];
        let mut deltas = recent.windows(2).map(|w| (w[1] - w[0]).abs());
        deltas.all(|delta| delta < self.threshold)
    }
}
