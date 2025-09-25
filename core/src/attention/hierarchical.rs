use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{AttentionConfig, MultiHeadSelfAttention};

#[derive(Debug)]
pub struct HierarchicalAttention {
    local: MultiHeadSelfAttention,
    global_mixer: Linear,
    gate: Linear,
}

impl HierarchicalAttention {
    pub fn new(config: AttentionConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            local: MultiHeadSelfAttention::new(config, vb.pp("local"))?,
            global_mixer: candle_nn::linear(config.n_embd, config.n_embd, vb.pp("global_mixer"))?,
            gate: candle_nn::linear(config.n_embd, config.n_embd, vb.pp("gate"))?,
        })
    }

    fn global_summary(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (_batch, seq_len, _n_embd) = x.dims3()?;
        let summary = x.sum(1)?.affine(1.0 / seq_len as f64, 0.0)?;
        summary.unsqueeze(1)
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> CandleResult<Tensor> {
        let local = self.local.forward(x, mask, train)?;
        let summary = self.global_summary(x)?;
        let blended = self
            .global_mixer
            .forward(&summary.broadcast_as(local.shape())?)?;
        let gate_raw = self.gate.forward(&local)?;
        let gate = candle_nn::ops::sigmoid(&gate_raw)?;
        let inverse_gate = gate.affine(-1.0, 1.0)?;
        local.mul(&gate)?.add(&blended.mul(&inverse_gate)?)
    }

    pub fn local_attention(&self) -> &MultiHeadSelfAttention {
        &self.local
    }
}
