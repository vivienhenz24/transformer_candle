use candle_core::{Result as CandleResult, Tensor};
use candle_nn::ops::softmax_last_dim;

#[derive(Debug, Clone)]
pub struct AdaptiveSamplingConfig {
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub base_temperature: f64,
    pub min_temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
}

impl Default for AdaptiveSamplingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            min_tokens: 16,
            base_temperature: 0.8,
            min_temperature: 0.2,
            top_k: Some(40),
            top_p: Some(0.95),
        }
    }
}

#[derive(Debug)]
pub struct AdaptiveSampler {
    config: AdaptiveSamplingConfig,
}

impl AdaptiveSampler {
    pub fn new(config: AdaptiveSamplingConfig) -> Self {
        Self { config }
    }

    fn adaptive_temperature(&self, probs_vec: &[Vec<f32>]) -> f64 {
        let mut entropy = 0.0f64;
        let mut count = 0usize;
        for row in probs_vec {
            for &p in row {
                if p > 0.0 {
                    entropy -= (p as f64) * (p as f64).ln();
                }
            }
            count += 1;
        }
        if count == 0 {
            return self.config.base_temperature;
        }
        entropy /= count as f64;
        let scaled = (entropy / 5.0).clamp(0.0, 1.0);
        self.config.min_temperature + (self.config.base_temperature - self.config.min_temperature) * scaled
    }

    pub fn sample_next_token(&self, logits: &Tensor) -> CandleResult<Tensor> {
        let probs = softmax_last_dim(logits)?;
        let probs_vec = probs.to_vec2::<f32>()?;
        let temperature = self.adaptive_temperature(&probs_vec);
        let mut sampled = Vec::with_capacity(probs_vec.len());
        for row in probs_vec.iter() {
            let idx = sample_from_distribution(row, temperature, self.config.top_k, self.config.top_p);
            sampled.push(idx as u32);
        }
        Tensor::from_vec(sampled, (probs_vec.len(), 1), logits.device())
    }

    pub fn config(&self) -> &AdaptiveSamplingConfig {
        &self.config
    }
}

fn sample_from_distribution(
    logits: &[f32],
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> usize {
    if logits.is_empty() {
        return 0;
    }

    let inv_temp = if temperature <= 0.0 { 1.0 } else { (1.0 / temperature).max(1e-4) } as f32;
    let mut adjusted: Vec<f32> = logits.iter().map(|&logit| logit * inv_temp).collect();

    if let Some(mut k) = top_k {
        if k == 0 {
            k = 1;
        }
        if k < adjusted.len() {
            let mut indices: Vec<usize> = (0..adjusted.len()).collect();
            indices.sort_unstable_by(|a, b| adjusted[*b].partial_cmp(&adjusted[*a]).unwrap_or(std::cmp::Ordering::Equal));
            for &idx in indices.iter().skip(k) {
                adjusted[idx] = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = Vec::with_capacity(adjusted.len());
    let mut sum = 0.0f32;
    for &logit in &adjusted {
        let value = if logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0
        };
        exp_values.push(value);
        sum += value;
    }

    if sum <= f32::EPSILON {
        return fastrand::usize(0..adjusted.len());
    }

    let mut probabilities: Vec<f32> = exp_values.iter().map(|value| value / sum).collect();

    if let Some(p_threshold) = top_p {
        let mut pairs: Vec<(usize, f32)> = probabilities.iter().cloned().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut allowed = vec![false; probabilities.len()];
        for (idx, prob) in pairs {
            cumulative += prob;
            allowed[idx] = true;
            if cumulative >= p_threshold as f32 {
                break;
            }
        }

        for (idx, prob) in probabilities.iter_mut().enumerate() {
            if !allowed[idx] {
                *prob = 0.0;
            }
        }

        let renorm: f32 = probabilities.iter().sum();
        if renorm > f32::EPSILON {
            for prob in probabilities.iter_mut() {
                *prob /= renorm;
            }
        }
    }

    let mut cumulative = 0.0f32;
    let sample = fastrand::f32();
    for (idx, prob) in probabilities.iter().enumerate() {
        cumulative += *prob;
        if sample <= cumulative {
            return idx;
        }
    }

    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
