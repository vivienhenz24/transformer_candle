#[derive(Debug, Clone)]
pub struct NoiseInjection {
    pub stddev: f32,
}

impl NoiseInjection {
    pub fn new(stddev: f32) -> Self {
        Self { stddev }
    }
}
