use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{
    adaptive_window::{AdaptiveWindowAttention, AdaptiveWindowConfig},
    hierarchical::HierarchicalAttention,
    sparse_patterns::{PatternSpec, SparseAttentionMask},
    AttentionConfig, MultiHeadSelfAttention,
};

#[derive(Debug, Default, Clone)]
pub struct CrossLayerState {
    pub global_context: Option<Tensor>,
    pub residual_attn: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct CascadeAttentionConfig {
    pub attention: AttentionConfig,
    pub window: AdaptiveWindowConfig,
    pub sparse: Option<PatternSpec>,
    pub fusion_dropout: f32,
}

impl Default for CascadeAttentionConfig {
    fn default() -> Self {
        Self {
            attention: AttentionConfig {
                n_embd: 384,
                n_head: 6,
                block_size: 256,
                dropout: 0.1,
            },
            window: AdaptiveWindowConfig::default(),
            sparse: Some(PatternSpec::Local { window: 128 }),
            fusion_dropout: 0.1,
        }
    }
}

#[derive(Debug)]
pub struct CascadeAttentionComposer {
    adaptive: AdaptiveWindowAttention,
    hierarchical: HierarchicalAttention,
    residual: MultiHeadSelfAttention,
    fusion: Linear,
    dropout: candle_nn::Dropout,
    sparse_mask: Option<SparseAttentionMask>,
}

impl CascadeAttentionComposer {
    pub fn new(config: CascadeAttentionConfig, vb: VarBuilder) -> CandleResult<Self> {
        let adaptive = AdaptiveWindowAttention::new(config.attention, config.window, vb.pp("adaptive"))?;
        let hierarchical = HierarchicalAttention::new(config.attention, vb.pp("hierarchical"))?;
        let residual = MultiHeadSelfAttention::new(config.attention, vb.pp("residual"))?;
        let fusion = candle_nn::linear(config.attention.n_embd * 3, config.attention.n_embd, vb.pp("fusion"))?;
        let dropout = candle_nn::Dropout::new(config.fusion_dropout);
        let sparse_mask = config.sparse.map(SparseAttentionMask::new);

        Ok(Self {
            adaptive,
            hierarchical,
            residual,
            fusion,
            dropout,
            sparse_mask,
        })
    }

    fn build_sparse_mask(&self, device: &Device, seq_len: usize) -> CandleResult<Option<Tensor>> {
        if let Some(mask) = &self.sparse_mask {
            mask.as_tensor(device, seq_len).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        train: bool,
        shared_state: Option<&CrossLayerState>,
    ) -> CandleResult<(Tensor, CrossLayerState)> {
        let seq_len = x.dim(1)?;
        let mask = self.build_sparse_mask(x.device(), seq_len)?;

        let adaptive = self.adaptive.forward(x, train)?;
        let hierarchical = self.hierarchical.forward(x, mask.as_ref(), train)?;
        let residual = self.residual.forward(x, mask.as_ref(), train)?;

        let fused = Tensor::cat(&[adaptive.clone(), hierarchical.clone(), residual.clone()], 2)?;
        let mut fused = self.fusion.forward(&fused)?;
        fused = if train {
            self.dropout.forward(&fused, train)?
        } else {
            fused
        };

        let merged = fused.add(x)?;

        let global_context = if let Some(state) = shared_state {
            if let Some(context) = &state.global_context {
                Some(context.add(&hierarchical)?.affine(0.5, 0.0)?)
            } else {
                Some(hierarchical.clone())
            }
        } else {
            Some(hierarchical.clone())
        };

        let residual_attn = if let Some(state) = shared_state {
            if let Some(prev) = &state.residual_attn {
                Some(prev.add(&adaptive)?.affine(0.5, 0.0)?)
            } else {
                Some(adaptive.clone())
            }
        } else {
            Some(adaptive.clone())
        };

        Ok((merged, CrossLayerState { global_context, residual_attn }))
    }

    pub fn adaptive(&self) -> &AdaptiveWindowAttention {
        &self.adaptive
    }

    pub fn hierarchical(&self) -> &HierarchicalAttention {
        &self.hierarchical
    }
}
