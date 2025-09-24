#[derive(Debug, Clone, Default)]
pub struct AugmentationStrategies;

impl AugmentationStrategies {
    pub fn maybe_augment(&self, text: &str) -> String {
        text.to_string()
    }
}
