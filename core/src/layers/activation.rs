use candle_core::{Result as CandleResult, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum ActivationStrategy {
    Relu,
    Silu,
    GeLU,
    Swish,
    SqRelu,
    GatedSilu,
}

impl Default for ActivationStrategy {
    fn default() -> Self {
        ActivationStrategy::Silu
    }
}

impl ActivationStrategy {
    pub fn apply(&self, x: &Tensor, gate: Option<&Tensor>) -> CandleResult<Tensor> {
        match self {
            ActivationStrategy::Relu => x.relu(),
            ActivationStrategy::Silu => x.silu(),
            ActivationStrategy::GeLU => x.gelu(),
            ActivationStrategy::Swish => x.silu(),
            ActivationStrategy::SqRelu => {
                let relu = x.relu()?;
                relu.mul(&relu)
            }
            ActivationStrategy::GatedSilu => {
                let gate = gate.ok_or_else(|| candle_core::Error::Msg("GatedSilu requires gate tensor".into()))?;
                x.silu()?.mul(gate)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActivationRegistry {
    pub primary: ActivationStrategy,
    pub secondary: Option<ActivationStrategy>,
}

impl Default for ActivationRegistry {
    fn default() -> Self {
        Self {
            primary: ActivationStrategy::Silu,
            secondary: Some(ActivationStrategy::SqRelu),
        }
    }
}

impl ActivationRegistry {
    pub fn apply(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base = self.primary.apply(x, None)?;
        if let Some(extra) = self.secondary {
            extra.apply(&base, None)
        } else {
            Ok(base)
        }
    }
}
