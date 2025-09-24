use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};

#[derive(Debug, Clone, Copy)]
pub enum NormStrategy {
    Layer,
    Rms,
    Scale(f32),
}

impl Default for NormStrategy {
    fn default() -> Self {
        NormStrategy::Layer
    }
}

#[derive(Debug)]
pub struct CascadeNorm {
    layer: LayerNorm,
    rms: Option<candle_nn::RmsNorm>,
    strategy: NormStrategy,
}

impl CascadeNorm {
    pub fn new(n_embd: usize, eps: f64, strategy: NormStrategy, vb: VarBuilder) -> CandleResult<Self> {
        let layer = candle_nn::layer_norm(n_embd, eps, vb.pp("ln"))?;
        let rms = if let NormStrategy::Rms = strategy {
            Some(candle_nn::rms_norm(n_embd, eps, vb.pp("rms"))?)
        } else {
            None
        };

        Ok(Self { layer, rms, strategy })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self.strategy {
            NormStrategy::Layer => self.layer.forward(x),
            NormStrategy::Rms => {
                if let Some(rms) = &self.rms {
                    rms.forward(x)
                } else {
                    self.layer.forward(x)
                }
            }
            NormStrategy::Scale(scale) => self.layer.forward(x)?.affine(scale as f64, 0.0),
        }
    }
}
