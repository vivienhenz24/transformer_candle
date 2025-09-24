use candle_core::{Result as CandleResult, Tensor};

use crate::architectures::cascade::CascadeTransformer;
use crate::generation::adaptive_sampling::{AdaptiveSampler, AdaptiveSamplingConfig};
use crate::generation::creative_modes::CreativePalette;

#[derive(Debug, Clone)]
pub struct ProgressiveGenerationConfig {
    pub passes: usize,
    pub refinement_decay: f32,
    pub base_sampling: AdaptiveSamplingConfig,
}

impl Default for ProgressiveGenerationConfig {
    fn default() -> Self {
        Self {
            passes: 3,
            refinement_decay: 0.6,
            base_sampling: AdaptiveSamplingConfig::default(),
        }
    }
}

#[derive(Debug)]
pub struct ProgressiveRefiner {
    config: ProgressiveGenerationConfig,
}

impl ProgressiveRefiner {
    pub fn new(config: ProgressiveGenerationConfig) -> Self {
        Self { config }
    }

    pub fn generate(
        &self,
        model: &CascadeTransformer,
        context: &Tensor,
        palette: &CreativePalette,
    ) -> CandleResult<Tensor> {
        let mut output = context.clone();
        let mut sampling = self.config.base_sampling.clone();
        let mut sampler = palette.sampler();

        for pass in 0..self.config.passes.max(1) {
            let decay = self.config.refinement_decay.powi(pass as i32);
            sampling.base_temperature = (sampling.base_temperature * decay as f64).max(sampling.min_temperature);
            sampling.max_tokens = ((sampler.config().max_tokens as f32) * decay).max(sampling.min_tokens as f32) as usize;
            sampler = AdaptiveSampler::new(sampling.clone());
            output = model.generate_with_sampler(&output, &sampler, sampling.max_tokens)?;
        }

        Ok(output)
    }
}
