#[derive(Debug, Clone, Default)]
pub struct AdaptiveVocabulary {
    pub min_frequency: u32,
}

impl AdaptiveVocabulary {
    pub fn should_keep(&self, frequency: u32) -> bool {
        frequency >= self.min_frequency
    }
}
