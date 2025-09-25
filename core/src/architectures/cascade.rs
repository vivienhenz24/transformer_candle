use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{loss, Linear, Module, VarBuilder};

use crate::architectures::adaptive_depth::{
    AdaptiveDepthConfig, AdaptiveDepthController, TokenDepthUsage,
};
use crate::attention::cascade_attention::{
    CascadeAttentionComposer, CascadeAttentionConfig, CrossLayerState,
};
use crate::generation::adaptive_sampling::{AdaptiveSampler, AdaptiveSamplingConfig};
use crate::layers::activation::ActivationRegistry;
use crate::layers::embedding::{CascadeEmbeddingConfig, CascadeEmbeddings};
use crate::layers::feed_forward::{CascadeFeedForward, FeedForwardConfig};
use crate::layers::normalization::{CascadeNorm, NormStrategy};
use crate::memory::streaming::StreamingState;

#[derive(Debug, Clone)]
pub struct CascadeTransformerConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub dropout: f32,
    pub embedding: CascadeEmbeddingConfig,
    pub attention: CascadeAttentionConfig,
    pub feed_forward: FeedForwardConfig,
    pub depth: AdaptiveDepthConfig,
}

impl Default for CascadeTransformerConfig {
    fn default() -> Self {
        let n_embd = 384;
        Self {
            vocab_size: 65,
            block_size: 256,
            n_embd,
            n_head: 6,
            n_layer: 6,
            dropout: 0.1,
            embedding: CascadeEmbeddingConfig {
                vocab_size: 65,
                block_size: 256,
                n_embd,
                rotary_scaling: Some(0.01),
                learned_frequency: true,
            },
            attention: CascadeAttentionConfig::default(),
            feed_forward: FeedForwardConfig {
                n_embd,
                expansion: 4,
                dropout: 0.1,
                activation: ActivationRegistry::default(),
                gated: true,
            },
            depth: AdaptiveDepthConfig::default(),
        }
    }
}

#[derive(Debug)]
pub struct CascadeBlock {
    attn_norm: CascadeNorm,
    attention: CascadeAttentionComposer,
    ff_norm: CascadeNorm,
    feed_forward: CascadeFeedForward,
}

impl CascadeBlock {
    pub fn new(config: &CascadeTransformerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let attn_norm =
            CascadeNorm::new(config.n_embd, 1e-5, NormStrategy::Layer, vb.pp("ln_attn"))?;
        let mut attention_cfg = config.attention.clone();
        attention_cfg.attention.n_embd = config.n_embd;
        attention_cfg.attention.n_head = config.n_head;
        attention_cfg.attention.block_size = config.block_size;
        attention_cfg.attention.dropout = config.dropout;
        let attention = CascadeAttentionComposer::new(attention_cfg, vb.pp("attn"))?;

        let ff_norm = CascadeNorm::new(config.n_embd, 1e-5, NormStrategy::Rms, vb.pp("ln_ff"))?;
        let mut ff_cfg = config.feed_forward.clone();
        ff_cfg.n_embd = config.n_embd;
        let feed_forward = CascadeFeedForward::new(ff_cfg, vb.pp("ff"))?;

        Ok(Self {
            attn_norm,
            attention,
            ff_norm,
            feed_forward,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        train: bool,
        layer_idx: usize,
        depth_controller: &mut AdaptiveDepthController,
        usage: &mut TokenDepthUsage,
        shared_state: Option<&CrossLayerState>,
    ) -> CandleResult<(Tensor, CrossLayerState)> {
        let normed = self.attn_norm.forward(x)?;
        let (attn_out, attn_state) = self.attention.forward(&normed, train, shared_state)?;
        let (mask, flags) = depth_controller.layer_mask(layer_idx, &attn_out, usage)?;
        usage.increment(&flags);
        let mask = mask.broadcast_as(attn_out.shape())?;
        let attn_out = attn_out.mul(&mask)?;
        let residual = x.add(&attn_out)?;

        let ff_normed = self.ff_norm.forward(&residual)?;
        let ff_out = self.feed_forward.forward(&ff_normed, train)?;
        let ff_out = ff_out.mul(&mask)?;
        let output = residual.add(&ff_out)?;

        Ok((output, attn_state))
    }
}

#[derive(Debug)]
pub struct CascadeTransformer {
    pub config: CascadeTransformerConfig,
    embeddings: CascadeEmbeddings,
    blocks: Vec<CascadeBlock>,
    norm: CascadeNorm,
    head: Linear,
}

impl CascadeTransformer {
    pub fn new(config: CascadeTransformerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let embeddings = CascadeEmbeddings::new(config.embedding.clone(), vb.pp("embed"))?;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for idx in 0..config.n_layer {
            let block = CascadeBlock::new(&config, vb.pp(&format!("block_{idx}")))?;
            blocks.push(block);
        }
        let norm = CascadeNorm::new(config.n_embd, 1e-5, NormStrategy::Layer, vb.pp("ln_f"))?;
        let head = candle_nn::linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            config,
            embeddings,
            blocks,
            norm,
            head,
        })
    }

    pub fn forward(
        &self,
        idx: &Tensor,
        targets: Option<&Tensor>,
        train: bool,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len) = idx.dims2()?;
        if seq_len > self.config.block_size {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds block size {}",
                seq_len, self.config.block_size
            )));
        }

        let mut x = self.embeddings.forward(idx)?;

        let mut depth = AdaptiveDepthController::new(self.config.depth.clone());
        let mut usage = TokenDepthUsage::new(batch_size, seq_len, self.config.depth.max_layers);
        let mut shared_state: Option<CrossLayerState> = None;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let (next, state) = block.forward(
                &x,
                train,
                layer_idx,
                &mut depth,
                &mut usage,
                shared_state.as_ref(),
            )?;
            x = next;
            shared_state = Some(state);
        }

        x = self.norm.forward(&x)?;
        let logits = self.head.forward(&x)?;

        let loss = if let Some(targets) = targets {
            let logits_flat = logits.reshape(&[batch_size * seq_len, self.config.vocab_size])?;
            let targets_flat = targets.reshape(&[batch_size * seq_len])?;
            Some(loss::cross_entropy(&logits_flat, &targets_flat)?)
        } else {
            None
        };

        Ok((logits, loss))
    }

    pub fn generate_with_sampler(
        &self,
        context: &Tensor,
        sampler: &AdaptiveSampler,
        max_tokens: usize,
    ) -> CandleResult<Tensor> {
        let mut current = context.clone();
        let mut streaming = StreamingState::default();

        for _ in 0..max_tokens {
            let (_, seq_len) = current.dims2()?;
            streaming.update(seq_len, self.config.block_size);
            let model_input = streaming.apply(&current)?;
            let (logits, _) = self.forward(&model_input, None, false)?;
            let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
            let next_token = sampler.sample_next_token(&last_logits)?;
            current = Tensor::cat(&[current.clone(), next_token], 1)?;
        }

        Ok(current)
    }

    pub fn generate(
        &self,
        context: &Tensor,
        config: AdaptiveSamplingConfig,
    ) -> CandleResult<Tensor> {
        let sampler = AdaptiveSampler::new(config.clone());
        let max_tokens = config.max_tokens.max(config.min_tokens);
        self.generate_with_sampler(context, &sampler, max_tokens)
    }

    pub fn count_parameters(&self) -> usize {
        let embed = self.config.vocab_size * self.config.n_embd;
        let pos = self.config.block_size * self.config.n_embd;
        let per_block = self.config.n_layer
            * (12 * self.config.n_embd * self.config.n_embd + 2 * self.config.n_embd);
        let head = self.config.n_embd * self.config.vocab_size;
        embed + pos + per_block + head
    }
}

pub struct CascadeTransformerBuilder {
    config: CascadeTransformerConfig,
}

impl CascadeTransformerBuilder {
    pub fn new(vocab_size: usize) -> Self {
        let mut config = CascadeTransformerConfig::default();
        config.vocab_size = vocab_size;
        config.embedding.vocab_size = vocab_size;
        Self { config }
    }

    pub fn block_size(mut self, block_size: usize) -> Self {
        self.config.block_size = block_size;
        self.config.embedding.block_size = block_size;
        self
    }

    pub fn model_width(mut self, n_embd: usize) -> Self {
        self.config.n_embd = n_embd;
        self.config.embedding.n_embd = n_embd;
        self.config.feed_forward.n_embd = n_embd;
        self
    }

    pub fn layers(mut self, n_layer: usize, n_head: usize) -> Self {
        self.config.n_layer = n_layer;
        self.config.n_head = n_head;
        self
    }

    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = dropout;
        self.config.attention.attention.dropout = dropout;
        self.config.feed_forward.dropout = dropout;
        self
    }

    pub fn depth(mut self, depth: AdaptiveDepthConfig) -> Self {
        self.config.depth = depth;
        self
    }

    pub fn build(self, vb: VarBuilder) -> CandleResult<CascadeTransformer> {
        CascadeTransformer::new(self.config, vb)
    }

    pub fn config(&self) -> &CascadeTransformerConfig {
        &self.config
    }
}
