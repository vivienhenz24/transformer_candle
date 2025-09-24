use super::adaptive_sampling::{AdaptiveSampler, AdaptiveSamplingConfig};

#[derive(Debug, Clone, Copy)]
pub enum CreativeMode {
    Faithful,
    Dramatic,
    Experimental,
    Whisper,
}

#[derive(Debug, Clone)]
pub struct CreativePalette {
    pub mode: CreativeMode,
}

impl CreativePalette {
    pub fn new(mode: CreativeMode) -> Self {
        Self { mode }
    }

    pub fn sampler(&self) -> AdaptiveSampler {
        let config = match self.mode {
            CreativeMode::Faithful => AdaptiveSamplingConfig {
                max_tokens: 160,
                min_tokens: 32,
                base_temperature: 0.6,
                min_temperature: 0.2,
                top_k: Some(30),
                top_p: Some(0.9),
            },
            CreativeMode::Dramatic => AdaptiveSamplingConfig {
                max_tokens: 220,
                min_tokens: 40,
                base_temperature: 0.85,
                min_temperature: 0.35,
                top_k: Some(50),
                top_p: Some(0.95),
            },
            CreativeMode::Experimental => AdaptiveSamplingConfig {
                max_tokens: 240,
                min_tokens: 24,
                base_temperature: 1.1,
                min_temperature: 0.5,
                top_k: Some(80),
                top_p: Some(0.97),
            },
            CreativeMode::Whisper => AdaptiveSamplingConfig {
                max_tokens: 120,
                min_tokens: 12,
                base_temperature: 0.4,
                min_temperature: 0.15,
                top_k: Some(20),
                top_p: Some(0.85),
            },
        };
        AdaptiveSampler::new(config)
    }
}
