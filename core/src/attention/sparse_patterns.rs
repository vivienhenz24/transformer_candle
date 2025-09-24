use candle_core::{Device, Result as CandleResult, Tensor};

#[derive(Debug, Clone)]
pub enum PatternSpec {
    Local { window: usize },
    Dilated { step: usize, width: usize },
    Block { size: usize },
    Strided { stride: usize },
    Landmarks { anchors: usize },
    Custom(Vec<(usize, usize)>),
}

impl Default for PatternSpec {
    fn default() -> Self {
        PatternSpec::Local { window: 64 }
    }
}

#[derive(Debug, Clone)]
pub struct SparseAttentionMask {
    pattern: PatternSpec,
    penalty: f32,
}

impl SparseAttentionMask {
    pub fn new(pattern: PatternSpec) -> Self {
        Self {
            pattern,
            penalty: -1e9,
        }
    }

    pub fn with_penalty(mut self, penalty: f32) -> Self {
        self.penalty = penalty;
        self
    }

    pub fn pattern(&self) -> &PatternSpec {
        &self.pattern
    }

    pub fn as_tensor(&self, device: &Device, seq_len: usize) -> CandleResult<Tensor> {
        let mut mask = vec![self.penalty; seq_len * seq_len];

        match &self.pattern {
            PatternSpec::Local { window } => {
                let window = (*window).max(1).min(seq_len);
                for i in 0..seq_len {
                    let start = i.saturating_sub(window - 1);
                    for j in start..=i {
                        mask[i * seq_len + j] = 0.0;
                    }
                }
            }
            PatternSpec::Dilated { step, width } => {
                let step = (*step).max(1);
                let width = (*width).max(1);
                for i in 0..seq_len {
                    let mut count = 0usize;
                    let mut j = i;
                    while count < width {
                        mask[i * seq_len + j] = 0.0;
                        count += 1;
                        if j < step {
                            mask[i * seq_len] = 0.0;
                            break;
                        }
                        j -= step;
                    }
                }
            }
            PatternSpec::Block { size } => {
                let size = (*size).max(1);
                for block_start in (0..seq_len).step_by(size) {
                    let block_end = (block_start + size).min(seq_len);
                    for i in block_start..block_end {
                        for j in block_start..=i {
                            mask[i * seq_len + j] = 0.0;
                        }
                    }
                }
            }
            PatternSpec::Strided { stride } => {
                let stride = (*stride).max(1);
                for i in 0..seq_len {
                    mask[i * seq_len + i] = 0.0;
                    let mut j = i;
                    while j >= stride {
                        j -= stride;
                        mask[i * seq_len + j] = 0.0;
                        if j == 0 {
                            break;
                        }
                    }
                    mask[i * seq_len] = 0.0;
                }
            }
            PatternSpec::Landmarks { anchors } => {
                let anchors = (*anchors).max(1).min(seq_len);
                let stride = (seq_len / anchors.max(1)).max(1);
                let mut landmark_positions = Vec::new();
                let mut pos = 0usize;
                while pos < seq_len {
                    landmark_positions.push(pos);
                    pos += stride;
                }
                landmark_positions.push(seq_len - 1);
                landmark_positions.sort_unstable();
                landmark_positions.dedup();

                for i in 0..seq_len {
                    mask[i * seq_len + i] = 0.0;
                    for &anchor in &landmark_positions {
                        if anchor <= i {
                            mask[i * seq_len + anchor] = 0.0;
                        }
                    }
                }
            }
            PatternSpec::Custom(pairs) => {
                for &(i, j) in pairs {
                    if i < seq_len && j <= i {
                        mask[i * seq_len + j] = 0.0;
                    }
                }
            }
        }

        Tensor::from_vec(mask, (1, seq_len, seq_len), device)
    }
}
