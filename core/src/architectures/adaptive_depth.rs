use candle_core::{Result as CandleResult, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum DepthDecision {
    Execute,
    Skip,
}

#[derive(Debug, Clone)]
pub struct TokenDepthUsage {
    counters: Vec<usize>,
    max_layers: usize,
    seq_len: usize,
}

impl TokenDepthUsage {
    pub fn new(batch: usize, seq_len: usize, max_layers: usize) -> Self {
        Self {
            counters: vec![0; batch * seq_len],
            max_layers,
            seq_len,
        }
    }

    pub fn increment(&mut self, mask: &[bool]) {
        for (count, active) in self.counters.iter_mut().zip(mask.iter()) {
            if *active {
                *count += 1;
            }
        }
    }

    pub fn layer_budget_met(&self) -> Vec<bool> {
        self.counters.iter().map(|&c| c >= self.max_layers).collect()
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveDepthConfig {
    pub min_layers: usize,
    pub max_layers: usize,
    pub base_threshold: f32,
    pub sensitivity: f32,
}

impl Default for AdaptiveDepthConfig {
    fn default() -> Self {
        Self {
            min_layers: 2,
            max_layers: 8,
            base_threshold: 0.12,
            sensitivity: 1.5,
        }
    }
}

#[derive(Debug)]
pub struct AdaptiveDepthController {
    config: AdaptiveDepthConfig,
    running_complexity: f32,
}

impl AdaptiveDepthController {
    pub fn new(config: AdaptiveDepthConfig) -> Self {
        let running_complexity = config.base_threshold;
        Self {
            config,
            running_complexity,
        }
    }

    fn estimate_complexity(&self, activations: &Tensor) -> CandleResult<(Vec<f32>, f32)> {
        let (batch, seq_len, n_embd) = activations.dims3()?;
        let data = activations.to_vec3::<f32>()?;
        let mut complexities = Vec::with_capacity(batch * seq_len);
        for sample in data {
            for token in sample {
                let mut energy = 0.0f32;
                for value in &token {
                    energy += value * value;
                }
                complexities.push((energy / n_embd as f32).sqrt());
            }
        }
        let mean = if complexities.is_empty() {
            self.config.base_threshold
        } else {
            complexities.iter().copied().sum::<f32>() / complexities.len() as f32
        };
        Ok((complexities, mean))
    }

    pub fn layer_mask(
        &mut self,
        layer_idx: usize,
        activations: &Tensor,
        usage: &TokenDepthUsage,
    ) -> CandleResult<(Tensor, Vec<bool>)> {
        let (_, seq_len, _) = activations.dims3()?;
        let min_layers = self.config.min_layers.min(self.config.max_layers);
        if layer_idx < min_layers {
            let mask = Tensor::ones((activations.dim(0)?, seq_len, 1), activations.dtype(), activations.device())?;
            let flags = vec![true; usage.counters.len()];
            return Ok((mask, flags));
        }

        let (complexities, batch_mean) = self.estimate_complexity(activations)?;
        self.running_complexity = 0.9 * self.running_complexity + 0.1 * batch_mean;
        let dynamic_threshold = self.running_complexity * self.config.sensitivity;

        let mut active_flags = Vec::with_capacity(complexities.len());
        for (complexity, budget_met) in complexities.iter().zip(usage.layer_budget_met()) {
            if budget_met {
                active_flags.push(false);
            } else {
                active_flags.push(*complexity >= dynamic_threshold);
            }
        }

        let mut mask_tensor = Vec::with_capacity(active_flags.len());
        for flag in &active_flags {
            mask_tensor.push(if *flag { 1.0f32 } else { 0.0 });
        }

        let mask = Tensor::from_vec(mask_tensor, (activations.dim(0)?, seq_len, 1), activations.device())?;
        Ok((mask, active_flags))
    }
}
