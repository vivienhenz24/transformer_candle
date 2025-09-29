use std::fmt;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use attention::core::{Config as AttentionConfig, RopeMode};
use attention::fused::FusedAttention;
use attention::Attention;
use candle_core::{bail, Error, Result, Tensor, Var};
use embedding::positional::rope::{Rope, RopeConfig};
use layers::{
    activations::ActivationKind,
    checks,
    dtypes::PrecisionPolicy,
    linear::{Linear, LinearConfig, LinearInit, LinearLayer},
    mlp::{FeedForward, FeedForwardConfig, FeedForwardLayer},
    norm::{LayerNorm, NormConfig, NormKind, NormalizationLayer, RmsNorm},
    residual::{Residual, ResidualConfig},
};

use crate::config::ModelConfig;

pub(crate) fn build_norm(
    kind: NormKind,
    hidden: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<Arc<dyn NormalizationLayer>> {
    let config = NormConfig::new(hidden, kind);
    match kind {
        NormKind::LayerNorm => {
            let weight = Tensor::ones(hidden, dtype, device)?;
            let bias = Tensor::zeros(hidden, dtype, device)?;
            Ok(Arc::new(LayerNorm::new(weight, bias, config)?))
        }
        NormKind::RmsNorm => {
            let weight = Tensor::ones(hidden, dtype, device)?;
            Ok(Arc::new(RmsNorm::new(weight, config)?))
        }
        other => Err(Error::Msg(format!("unsupported norm kind {:?}", other))),
    }
}

fn default_rope_config(head_dim: usize) -> RopeConfig {
    let mut cfg = RopeConfig::default();
    cfg.head_dim = head_dim;
    cfg
}

/// Decoder block implementing the project pre-norm residual layout.
pub struct DecoderBlock {
    hidden_dim: usize,
    heads: usize,
    head_dim: usize,
    policy: PrecisionPolicy,
    norm_attn: Arc<dyn NormalizationLayer>,
    norm_mlp: Arc<dyn NormalizationLayer>,
    qkv_proj: Arc<Linear>,
    out_proj: Arc<Linear>,
    mlp: Arc<FeedForward>,
    attention: FusedAttention,
    attention_config: AttentionConfig,
    residual_attn: Residual,
    residual_mlp: Residual,
    rope_preapply: Option<Rope>,
    rope_mode: RopeMode,
    training: AtomicBool,
}

impl fmt::Debug for DecoderBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderBlock")
            .field("hidden_dim", &self.hidden_dim)
            .field("heads", &self.heads)
            .field("head_dim", &self.head_dim)
            .field("rope_mode", &self.rope_mode)
            .finish()
    }
}

impl DecoderBlock {
    /// Construct a decoder block from the shared [`ModelConfig`].
    pub fn new(
        index: usize,
        model_cfg: &ModelConfig,
        mut attn_cfg: AttentionConfig,
    ) -> Result<Self> {
        let policy = PrecisionPolicy::from_parameter_dtype(model_cfg.dtype);

        let norm_attn = build_norm(
            model_cfg.norm_kind,
            model_cfg.hidden_dim,
            model_cfg.dtype,
            &model_cfg.device,
        )?;
        let norm_mlp = build_norm(
            model_cfg.norm_kind,
            model_cfg.hidden_dim,
            model_cfg.dtype,
            &model_cfg.device,
        )?;

        let mut qkv_config = LinearConfig::new(model_cfg.hidden_dim, model_cfg.hidden_dim);
        qkv_config.bias = true;
        qkv_config.fused_projections = 3;
        let qkv_proj = Arc::new(Linear::with_init(
            qkv_config,
            &LinearInit::XavierUniform,
            &model_cfg.device,
            model_cfg.dtype,
        )?);

        let mut out_config = LinearConfig::new(model_cfg.hidden_dim, model_cfg.hidden_dim);
        out_config.bias = true;
        let out_proj = Arc::new(Linear::with_init(
            out_config,
            &LinearInit::XavierUniform,
            &model_cfg.device,
            model_cfg.dtype,
        )?);

        let activation = ActivationKind::Silu;
        let ff_config = FeedForwardConfig::with_expansion_ratio(
            model_cfg.hidden_dim,
            model_cfg.ff_ratio,
            activation,
        );
        let mlp = Arc::new(FeedForward::with_init(
            ff_config,
            activation,
            &LinearInit::XavierUniform,
            &LinearInit::XavierUniform,
            &model_cfg.device,
            model_cfg.dtype,
        )?);

        let mut residual_cfg = ResidualConfig::new(true);
        residual_cfg.dropout_p = model_cfg.residual_dropout_p;
        let base_seed = (index as u64).saturating_mul(2);
        let residual_attn = Residual::new(residual_cfg.clone(), base_seed);
        let residual_mlp = Residual::new(residual_cfg, base_seed + 1);

        attn_cfg.dropout_p = model_cfg.attn_dropout_p;
        attn_cfg.rope_mode = model_cfg.rope_mode.clone();
        attn_cfg.use_padding_mask = true;

        let rope_mode = model_cfg.rope_mode.clone();
        let rope_config = match rope_mode {
            RopeMode::Off => None,
            _ => Some(default_rope_config(model_cfg.head_dim)),
        };

        let (attention, rope_preapply) = match rope_mode {
            RopeMode::Off => (FusedAttention::new(), None),
            RopeMode::OnTheFly => {
                let cfg = rope_config
                    .clone()
                    .ok_or_else(|| Error::Msg("missing rope config".into()))?;
                (FusedAttention::with_rope(cfg), None)
            }
            RopeMode::Preapply => {
                let cfg = rope_config
                    .clone()
                    .ok_or_else(|| Error::Msg("missing rope config".into()))?;
                let rope = Rope::new(cfg)?;
                (FusedAttention::new(), Some(rope))
            }
        };

        Ok(Self {
            hidden_dim: model_cfg.hidden_dim,
            heads: model_cfg.n_heads,
            head_dim: model_cfg.head_dim,
            policy,
            norm_attn,
            norm_mlp,
            qkv_proj,
            out_proj,
            mlp,
            attention,
            attention_config: attn_cfg,
            residual_attn,
            residual_mlp,
            rope_preapply,
            rope_mode,
            training: AtomicBool::new(true),
        })
    }

    fn expand_to_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        checks::expect_batch_seq_hidden("attention.input", tensor, self.hidden_dim)?;
        let [batch, seq, _] = match tensor.dims() {
            [b, s, h] => [*b, *s, *h],
            dims => bail!(
                "attention.input expected [batch, seq, hidden] got {:?}",
                dims
            ),
        };
        let reshaped = tensor.reshape((batch, seq, self.heads, self.head_dim))?;
        let permuted = reshaped.permute((0, 2, 1, 3))?;
        permuted.contiguous()
    }

    fn merge_from_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.dims();
        if dims.len() != 4 {
            bail!(
                "attention output expected [batch, heads, seq, head_dim] got {:?}",
                dims
            );
        }
        let batch = dims[0];
        let seq = dims[2];
        let permuted = tensor.permute((0, 2, 1, 3))?;
        let contiguous = permuted.contiguous()?;
        contiguous.reshape((batch, seq, self.hidden_dim))
    }

    /// Returns the trainable parameters for this block with an optional scope prefix.
    pub fn parameters(&self, prefix: &str) -> Vec<(String, Var)> {
        fn join(prefix: &str, name: &str) -> String {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}.{}", prefix, name)
            }
        }

        fn sanitize_linear_scope(name: &str) -> String {
            let mut segments: Vec<&str> = name.split('.').collect();
            if segments.len() < 2 {
                return name.to_string();
            }
            let suffix = segments.pop().unwrap();
            if let Some(last) = segments.last() {
                if last.starts_with("linear_") {
                    segments.pop();
                }
            }
            segments.push(suffix);
            segments.join(".")
        }

        let mut params = Vec::new();
        params.extend(self.norm_attn.named_parameters(&join(prefix, "norm_attn")));
        params.extend(self.norm_mlp.named_parameters(&join(prefix, "norm_mlp")));
        params.extend(
            self.qkv_proj
                .named_parameters_with_scope(&join(prefix, "qkv")),
        );
        params.extend(
            self.out_proj
                .named_parameters_with_scope(&join(prefix, "out_proj")),
        );
        params.extend(self.mlp.named_parameters(&join(prefix, "mlp")));

        params
            .into_iter()
            .map(|(name, var)| (sanitize_linear_scope(&name), var))
            .collect()
    }

    /// Legacy shim preserved for external callers.
    pub fn named_parameters(&self, scope: &str) -> Vec<(String, Var)> {
        self.parameters(scope)
    }

    /// Toggle training behaviour (dropout) for this block.
    pub fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
        self.residual_attn.set_training(training);
        self.residual_mlp.set_training(training);
    }

    /// Forward pass through the decoder block.
    pub fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        rope_positions: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.norm_attn.forward(hidden, &self.policy)?;
        let qkv = self.qkv_proj.forward(&normed, &self.policy)?;

        let q = qkv.narrow(2, 0, self.hidden_dim)?;
        let k = qkv.narrow(2, self.hidden_dim, self.hidden_dim)?;
        let v = qkv.narrow(2, 2 * self.hidden_dim, self.hidden_dim)?;

        let mut q_heads = self.expand_to_heads(&q)?;
        let mut k_heads = self.expand_to_heads(&k)?;
        let v_heads = self.expand_to_heads(&v)?;

        if self.rope_mode == RopeMode::Preapply {
            let rope = self
                .rope_preapply
                .as_ref()
                .ok_or_else(|| Error::Msg("preapply rope requires rope instance".into()))?;
            let positions = rope_positions
                .ok_or_else(|| Error::Msg("preapply rope requires position tensor".into()))?;
            let (q_rot, k_rot) = rope.apply_rotary_embeddings(&q_heads, &k_heads, positions)?;
            q_heads = q_rot;
            k_heads = k_rot;
        }

        let mut attn_cfg = self.attention_config.clone();
        if !self.training.load(Ordering::Relaxed) {
            attn_cfg.dropout_p = None;
        }

        let attn = self
            .attention
            .attend(&q_heads, &k_heads, &v_heads, mask, &attn_cfg)
            .map_err(|e| Error::Msg(e.to_string()))?;
        let attn_merged = self.merge_from_heads(&attn)?;
        let projected = self.out_proj.forward(&attn_merged, &self.policy)?;
        let after_attn = self
            .residual_attn
            .prenorm_step(&projected, hidden, &self.policy)?;

        let normed_mlp = self.norm_mlp.forward(&after_attn, &self.policy)?;
        let mlp_out = self.mlp.forward(&normed_mlp, &self.policy)?;
        self.residual_mlp
            .prenorm_step(&mlp_out, &after_attn, &self.policy)
    }
}
