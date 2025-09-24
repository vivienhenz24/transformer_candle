#[derive(Debug, Default)]
pub struct LossHistory {
    pub values: Vec<f32>,
}

impl LossHistory {
    pub fn push(&mut self, value: f32) {
        self.values.push(value);
    }
}
