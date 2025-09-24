use candle_core::{Device, Result as CandleResult, Tensor};

use super::{AttentionConfig, MultiHeadSelfAttention};

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveWindowConfig {
    pub min_window: usize,
    pub max_window: usize,
    pub responsiveness: f32,
}

impl Default for AdaptiveWindowConfig {
    fn default() -> Self {
        Self {
            min_window: 16,
            max_window: 256,
            responsiveness: 0.75,
        }
    }
}

#[derive(Debug)]
pub struct AdaptiveWindowAttention {
    base: MultiHeadSelfAttention,
    config: AdaptiveWindowConfig,
}

impl AdaptiveWindowAttention {
    pub fn new(attn_config: AttentionConfig, window_config: AdaptiveWindowConfig, vb: candle_nn::VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            base: MultiHeadSelfAttention::new(attn_config, vb)?,
            config: window_config,
        })
    }

    pub fn determine_window(&self, x: &Tensor) -> CandleResult<usize> {
        let (_batch, seq_len, _n_embd) = x.dims3()?;
        if seq_len <= 1 {
            return Ok(self.config.min_window.min(seq_len));
        }

        let left = x.narrow(1, 0, seq_len - 1)?;
        let right = x.narrow(1, 1, seq_len - 1)?;
        let diffs = right.sub(&left)?.abs()?;
        let mean = diffs.mean_all()?.to_scalar::<f32>().unwrap_or(0.0);
        let scaled = (mean * self.config.responsiveness).tanh();
        let window = self.config.min_window as f32
            + (self.config.max_window.saturating_sub(self.config.min_window) as f32)
                * (0.5 * (scaled + 1.0));
        let window = window.round() as usize;
        let window = window.clamp(self.config.min_window, self.config.max_window);
        Ok(window.min(seq_len).max(1))
    }

    fn build_window_mask(&self, device: &Device, seq_len: usize, window: usize) -> CandleResult<Tensor> {
        let mut data = vec![-1e9f32; seq_len * seq_len];
        for i in 0..seq_len {
            let start = i.saturating_sub(window - 1);
            for j in start..=i {
                data[i * seq_len + j] = 0.0;
            }
        }
        Tensor::from_vec(data, (1, seq_len, seq_len), device)
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        let window = self.determine_window(x)?;
        let mask = self.build_window_mask(x.device(), x.dim(1)?, window)?;
        self.base.forward(x, Some(&mask), train)
    }

    pub fn base(&self) -> &MultiHeadSelfAttention {
        &self.base
    }
}
