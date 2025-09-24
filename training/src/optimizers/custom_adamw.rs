use candle_core::{Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

#[derive(Debug)]
pub struct CustomAdamW {
    inner: AdamW,
}

impl CustomAdamW {
    pub fn new(params: Vec<Var>, settings: ParamsAdamW) -> candle_core::Result<Self> {
        Ok(Self {
            inner: AdamW::new(params, settings)?,
        })
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> candle_core::Result<()> {
        self.inner.backward_step(loss)
    }
}
