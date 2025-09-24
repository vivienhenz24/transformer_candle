#[derive(Debug, Clone)]
pub struct GradientPenalty {
    pub scale: f32,
}

impl GradientPenalty {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}
