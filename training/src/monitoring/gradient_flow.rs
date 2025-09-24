#[derive(Debug, Default)]
pub struct GradientFlowTracker {
    pub norms: Vec<f32>,
}

impl GradientFlowTracker {
    pub fn record(&mut self, norm: f32) {
        self.norms.push(norm);
    }
}
