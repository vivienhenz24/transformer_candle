use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder};

use super::activation::ActivationRegistry;

#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    pub n_embd: usize,
    pub expansion: usize,
    pub dropout: f32,
    pub activation: ActivationRegistry,
    pub gated: bool,
}

impl Default for FeedForwardConfig {
    fn default() -> Self {
        Self {
            n_embd: 384,
            expansion: 4,
            dropout: 0.1,
            activation: ActivationRegistry::default(),
            gated: true,
        }
    }
}

#[derive(Debug)]
pub struct CascadeFeedForward {
    c_fc: Linear,
    c_gate: Option<Linear>,
    c_proj: Linear,
    dropout: Dropout,
    activation: ActivationRegistry,
    config: FeedForwardConfig,
}

impl CascadeFeedForward {
    pub fn new(config: FeedForwardConfig, vb: VarBuilder) -> CandleResult<Self> {
        let inner = config.expansion.max(1) * config.n_embd;
        let c_fc = candle_nn::linear(config.n_embd, inner, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear(inner, config.n_embd, vb.pp("c_proj"))?;
        let c_gate = if config.gated {
            Some(candle_nn::linear(config.n_embd, inner, vb.pp("c_gate"))?)
        } else {
            None
        };
        let dropout = Dropout::new(config.dropout);

        Ok(Self {
            c_fc,
            c_gate,
            c_proj,
            dropout,
            activation: config.activation.clone(),
            config,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        let mut hidden = self.c_fc.forward(x)?;
        let gate_tensor = if let Some(gate) = &self.c_gate {
            Some(gate.forward(x)?)
        } else {
            None
        };

        hidden = if self.config.gated {
            self.activation
                .primary
                .apply(&hidden, gate_tensor.as_ref())?
        } else {
            self.activation.apply(&hidden)?
        };

        let projected = self.c_proj.forward(&hidden)?;
        if train {
            self.dropout.forward(&projected, train)
        } else {
            Ok(projected)
        }
    }

    pub fn config(&self) -> &FeedForwardConfig {
        &self.config
    }
}
