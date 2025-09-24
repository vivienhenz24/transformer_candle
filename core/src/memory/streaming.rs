use candle_core::{Result as CandleResult, Tensor};

#[derive(Debug, Clone)]
pub struct StreamWindow {
    pub start: usize,
    pub end: usize,
}

impl StreamWindow {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

#[derive(Debug, Default)]
pub struct StreamingState {
    pub window: Option<StreamWindow>,
}

impl StreamingState {
    pub fn update(&mut self, seq_len: usize, block_size: usize) {
        if seq_len <= block_size {
            self.window = Some(StreamWindow { start: 0, end: seq_len });
        } else {
            let start = seq_len - block_size;
            self.window = Some(StreamWindow { start, end: seq_len });
        }
    }

    pub fn apply(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        if let Some(window) = &self.window {
            tensor.narrow(1, window.start, window.len())
        } else {
            Ok(tensor.clone())
        }
    }
}
