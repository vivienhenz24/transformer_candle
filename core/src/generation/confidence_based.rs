use candle_core::{Result as CandleResult, Tensor};
use candle_nn::ops::softmax_last_dim;

#[derive(Debug, Clone)]
pub struct ConfidenceSettings {
    pub min_confidence: f32,
    pub max_confidence: f32,
    pub patience: usize,
}

impl Default for ConfidenceSettings {
    fn default() -> Self {
        Self {
            min_confidence: 0.4,
            max_confidence: 0.95,
            patience: 12,
        }
    }
}

#[derive(Debug)]
pub struct ConfidenceGuidedSampler {
    settings: ConfidenceSettings,
    stable_steps: usize,
}

impl ConfidenceGuidedSampler {
    pub fn new(settings: ConfidenceSettings) -> Self {
        Self {
            settings,
            stable_steps: 0,
        }
    }

    pub fn confidence(&self, logits: &Tensor) -> CandleResult<f32> {
        let probs = softmax_last_dim(logits)?;
        let mut best = 0.0f32;
        let rows = probs.to_vec2::<f32>()?;
        if let Some(row) = rows.last() {
            best = row
                .iter()
                .cloned()
                .fold(0.0f32, |acc, value| acc.max(value));
        }
        Ok(best)
    }

    pub fn should_stop(&mut self, logits: &Tensor) -> CandleResult<bool> {
        let confidence = self.confidence(logits)?;
        if confidence >= self.settings.max_confidence {
            self.stable_steps += 1;
        } else if confidence >= self.settings.min_confidence {
            self.stable_steps = self.stable_steps.saturating_sub(1);
        } else {
            self.stable_steps = 0;
        }
        Ok(self.stable_steps >= self.settings.patience)
    }
}
