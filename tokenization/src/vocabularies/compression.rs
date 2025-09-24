#[derive(Debug, Clone)]
pub struct VocabularyCompressor {
    pub keep_fraction: f32,
}

impl VocabularyCompressor {
    pub fn new(keep_fraction: f32) -> Self {
        Self {
            keep_fraction: keep_fraction.clamp(0.1, 1.0),
        }
    }

    pub fn target_size(&self, current: usize) -> usize {
        ((current as f32) * self.keep_fraction) as usize
    }
}
