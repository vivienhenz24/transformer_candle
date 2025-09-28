//! Optimised fused attention kernels.
//!
//! These implementations are built for hardware-specific acceleration and are
//! only compiled when the `fused` feature is enabled. The implementation here
//! provides a Flash-attention style execution path that keeps semantics aligned
//! with the portable reference backend while chunking work to minimise memory
//! traffic.

use std::cmp::max;
use std::sync::Mutex;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use embedding::positional::rope::RopeConfig;

use crate::core::{Attention, AttentionError, BackendSelection, Config};
use crate::interop::RopeAdapter;
use crate::masks::MASK_DTYPE;
use crate::reference::ExactAttention;

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeviceKind {
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl DeviceKind {
    fn from_device(device: &Device) -> Self {
        match device.location() {
            candle_core::DeviceLocation::Cpu => Self::Cpu,
            candle_core::DeviceLocation::Cuda { gpu_id } => Self::Cuda(gpu_id),
            candle_core::DeviceLocation::Metal { gpu_id } => Self::Metal(gpu_id),
        }
    }
}

#[derive(Debug)]
struct FusedKernel {
    rope_config: Option<RopeConfig>,
    rope_state: Mutex<Option<(DeviceKind, RopeAdapter)>>,
}

impl FusedKernel {
    fn new() -> Self {
        Self {
            rope_config: None,
            rope_state: Mutex::new(None),
        }
    }

    fn with_rope(rope_config: RopeConfig) -> Self {
        Self {
            rope_config: Some(rope_config),
            rope_state: Mutex::new(None),
        }
    }

    fn apply_rope(
        &self,
        device: &Device,
        q: &Tensor,
        k: &Tensor,
    ) -> Result<(Tensor, Tensor), AttentionError> {
        if let Some(cfg) = &self.rope_config {
            let mut guard = self
                .rope_state
                .lock()
                .map_err(|_| AttentionError::Backend {
                    message: "rope adapter mutex poisoned".to_string(),
                })?;
            let kind = DeviceKind::from_device(device);
            if guard
                .as_ref()
                .map(|(existing_kind, _)| existing_kind != &kind)
                .unwrap_or(true)
            {
                *guard = Some((kind.clone(), RopeAdapter::new(cfg.clone(), device.clone())));
            }
            if let Some((_, adapter)) = guard.as_mut() {
                let (q_rot, k_rot) =
                    adapter
                        .apply(q, k, 0)
                        .map_err(|err| AttentionError::Backend {
                            message: err.to_string(),
                        })?;
                return Ok((q_rot, k_rot));
            }
        }

        Ok((q.clone(), k.clone()))
    }

    fn fused_forward(
        &self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, usize), AttentionError> {
        let device = q.device();
        let dtype = q.dtype();

        if !device.same_device(k.device()) || !device.same_device(v.device()) {
            return Err(AttentionError::InvalidShape {
                context: "q, k, v must reside on the same device".to_string(),
            });
        }

        if dtype != k.dtype() || dtype != v.dtype() {
            return Err(AttentionError::InvalidShape {
                context: "q, k, v must share the same dtype".to_string(),
            });
        }

        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(AttentionError::UnsupportedDType {
                requested: format!("{dtype:?}"),
            });
        }

        if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
            return Err(AttentionError::InvalidShape {
                context: "q, k, v must be contiguous in memory".to_string(),
            });
        }

        let (batch, heads, q_len, head_dim) =
            q.dims4().map_err(|_| AttentionError::InvalidShape {
                context: "q must have shape [batch, heads, seq_len, head_dim]".to_string(),
            })?;
        let (kb, kh, k_len, kd) = k.dims4().map_err(|_| AttentionError::InvalidShape {
            context: "k must have shape [batch, heads, seq_len, head_dim]".to_string(),
        })?;
        let (vb, vh, vk, vd) = v.dims4().map_err(|_| AttentionError::InvalidShape {
            context: "v must have shape [batch, heads, seq_len, head_dim]".to_string(),
        })?;

        if kb != batch || kh != heads || kd != head_dim {
            return Err(AttentionError::InvalidShape {
                context: format!(
                    "k shape mismatch: expected [{batch}, {heads}, ?, {head_dim}] got [{kb}, {kh}, {k_len}, {kd}]"
                ),
            });
        }
        if vb != batch || vh != heads || vk != k_len || vd != head_dim {
            return Err(AttentionError::InvalidShape {
                context: format!(
                    "v shape mismatch: expected [{batch}, {heads}, {k_len}, {head_dim}] got [{vb}, {vh}, {vk}, {vd}]"
                ),
            });
        }

        let (q_processed, k_processed) = self.apply_rope(device, q, &k)?;

        let q_f32 = q_processed.to_dtype(DType::F32).map_err(to_backend_err)?;
        let k_cast = k_processed.to_dtype(DType::F32).map_err(to_backend_err)?;
        let v_cast = v.to_dtype(DType::F32).map_err(to_backend_err)?;

        let mut mask_broadcast = None;
        if let Some(mask) = mask {
            if !device.same_device(mask.device()) {
                return Err(AttentionError::InvalidShape {
                    context: "mask must reside on the same device as q".to_string(),
                });
            }
            if mask.dtype() != MASK_DTYPE {
                return Err(AttentionError::UnsupportedDType {
                    requested: format!("mask expects dtype {MASK_DTYPE:?}, got {:?}", mask.dtype()),
                });
            }
            let (mb, mh, mq, mk) = mask.dims4().map_err(|_| AttentionError::InvalidShape {
                context: "mask must have shape [batch, heads|1, q_len, k_len]".to_string(),
            })?;
            if mb != batch || mq != q_len || mk != k_len {
                return Err(AttentionError::InvalidShape {
                    context: format!(
                        "mask shape mismatch: expected [{batch}, 1|{heads}, {q_len}, {k_len}] got [{mb}, {mh}, {mq}, {mk}]"
                    ),
                });
            }
            if mh != 1 && mh != heads {
                return Err(AttentionError::InvalidShape {
                    context: format!("mask head dimension must be 1 or {heads}, got {mh}"),
                });
            }
            let broadcasted = if mh == heads {
                mask.clone()
            } else {
                mask.broadcast_as((batch, heads, q_len, k_len))
                    .map_err(to_backend_err)?
            };
            mask_broadcast = Some(broadcasted);
        }

        let tile_k = select_tile_size(k_len, head_dim);
        log::info!(
            "attention backend=fused tile={} batch={} heads={} q_len={} k_len={} head_dim={} dtype={:?}",
            tile_k,
            batch,
            heads,
            q_len,
            k_len,
            head_dim,
            dtype
        );

        let merged = batch * heads;
        let q_view = q_f32
            .reshape((merged, q_len, head_dim))
            .map_err(to_backend_err)?;
        let k_f32 = k_cast
            .reshape((merged, k_len, head_dim))
            .map_err(to_backend_err)?;
        let v_f32 = v_cast
            .reshape((merged, k_len, head_dim))
            .map_err(to_backend_err)?;

        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut max_scores = Tensor::full(f32::NEG_INFINITY, (batch, heads, q_len, 1usize), device)
            .map_err(to_backend_err)?;
        let mut sum_exp = Tensor::zeros((batch, heads, q_len, 1usize), DType::F32, device)
            .map_err(to_backend_err)?;
        let mut weighted = Tensor::zeros((batch, heads, q_len, head_dim), DType::F32, device)
            .map_err(to_backend_err)?;

        let mut start = 0;
        while start < k_len {
            let current = max(tile_k, 1).min(k_len - start);
            let k_chunk = k_f32.narrow(1, start, current).map_err(to_backend_err)?;
            let v_chunk = v_f32.narrow(1, start, current).map_err(to_backend_err)?;

            let k_chunk_t = k_chunk.transpose(1, 2).map_err(to_backend_err)?;
            let scores = q_view
                .matmul(&k_chunk_t)
                .map_err(to_backend_err)?
                .reshape((batch, heads, q_len, current))
                .map_err(to_backend_err)?
                .mul_scalar(scale)
                .map_err(to_backend_err)?;

            let mut scores = scores;
            if let Some(mask) = &mask_broadcast {
                let mask_chunk = mask.narrow(3, start, current).map_err(to_backend_err)?;
                scores = scores.add(&mask_chunk).map_err(to_backend_err)?;
            }

            let tile_max = scores.max_keepdim(3).map_err(to_backend_err)?;
            let candidates =
                Tensor::cat(&[max_scores.clone(), tile_max.clone()], 3).map_err(to_backend_err)?;
            let new_max = candidates.max_keepdim(3).map_err(to_backend_err)?;

            let zeros_prev = sum_exp.zeros_like().map_err(to_backend_err)?;
            let initial_mask = sum_exp.eq(&zeros_prev).map_err(to_backend_err)?;
            let prev_scale = max_scores
                .sub(&new_max)
                .map_err(to_backend_err)?
                .exp()
                .map_err(to_backend_err)?;
            let prev_scale = initial_mask
                .where_cond(&zeros_prev, &prev_scale)
                .map_err(to_backend_err)?;

            let scores_shifted = scores
                .sub(
                    &new_max
                        .broadcast_as((batch, heads, q_len, current))
                        .map_err(to_backend_err)?,
                )
                .map_err(to_backend_err)?;
            let scores_exp = scores_shifted.exp().map_err(to_backend_err)?;
            let sum_tile = scores_exp.sum_keepdim(3).map_err(to_backend_err)?;

            let prev_scale_expanded = prev_scale
                .broadcast_as((batch, heads, q_len, head_dim))
                .map_err(to_backend_err)?;

            let sum_exp_scaled = sum_exp.mul(&prev_scale).map_err(to_backend_err)?;

            let scores_mat = scores_exp
                .reshape((merged, q_len, current))
                .map_err(to_backend_err)?;
            let v_mat = v_chunk
                .reshape((merged, current, head_dim))
                .map_err(to_backend_err)?;
            let chunk_weighted = scores_mat
                .matmul(&v_mat)
                .map_err(to_backend_err)?
                .reshape((batch, heads, q_len, head_dim))
                .map_err(to_backend_err)?;

            weighted = weighted
                .mul(&prev_scale_expanded)
                .map_err(to_backend_err)?
                .add(&chunk_weighted)
                .map_err(to_backend_err)?;

            sum_exp = sum_exp_scaled.add(&sum_tile).map_err(to_backend_err)?;
            max_scores = new_max;
            start += current;
        }

        let zeros = sum_exp.zeros_like().map_err(to_backend_err)?;
        let ones = sum_exp.ones_like().map_err(to_backend_err)?;
        let zero_mask = sum_exp.eq(&zeros).map_err(to_backend_err)?;
        let safe_sum = zero_mask
            .where_cond(&ones, &sum_exp)
            .map_err(to_backend_err)?;

        let denom = safe_sum
            .broadcast_as((batch, heads, q_len, head_dim))
            .map_err(to_backend_err)?;
        let mut output = weighted.div(&denom).map_err(to_backend_err)?;
        let zeros_out = output.zeros_like().map_err(to_backend_err)?;
        let zero_mask_expanded = zero_mask
            .broadcast_as((batch, heads, q_len, head_dim))
            .map_err(to_backend_err)?;
        output = zero_mask_expanded
            .where_cond(&zeros_out, &output)
            .map_err(to_backend_err)?;

        Ok((output, max_scores, safe_sum, tile_k))
    }
}

/// Hybrid attention that forwards to the fused kernel when heuristics (or the
/// configuration override) allow, falling back to the reference path otherwise.
#[derive(Debug)]
pub struct FusedAttention {
    reference: ExactAttention,
    kernel: FusedKernel,
}

impl FusedAttention {
    /// Construct the dispatcher without RoPE support.
    pub fn new() -> Self {
        Self {
            reference: ExactAttention::new(),
            kernel: FusedKernel::new(),
        }
    }

    /// Construct the dispatcher with a shared RoPE configuration.
    pub fn with_rope(rope_config: RopeConfig) -> Self {
        Self {
            reference: ExactAttention::with_rope(rope_config.clone()),
            kernel: FusedKernel::with_rope(rope_config),
        }
    }

    fn should_use_fused(
        &self,
        config: &Config,
        batch: usize,
        heads: usize,
        q_len: usize,
        head_dim: usize,
        dtype: DType,
    ) -> bool {
        match config.backend {
            BackendSelection::ReferenceOnly => false,
            BackendSelection::FusedOnly => true,
            BackendSelection::Auto => {
                if config.dropout_p.unwrap_or(0.0) > 0.0 {
                    return false;
                }
                if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
                    return false;
                }
                let tokens = batch * heads * q_len;
                q_len >= 128 && head_dim <= 160 && tokens >= 2048
            }
        }
    }
}

impl Default for FusedAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl Attention for FusedAttention {
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        config: &Config,
    ) -> Result<Tensor, AttentionError> {
        let dtype = q.dtype();
        let (batch, heads, q_len, head_dim) =
            q.dims4().map_err(|_| AttentionError::InvalidShape {
                context: "q must have shape [batch, heads, seq_len, head_dim]".to_string(),
            })?;

        if config.dropout_p.unwrap_or(0.0) > 0.0 {
            log::warn!(
                "fused backend requested but dropout > 0 detected; falling back to reference"
            );
            return self.reference.attend(q, k, v, mask, config);
        }

        if self.should_use_fused(config, batch, heads, q_len, head_dim, dtype) {
            match self.kernel.fused_forward(q, k.clone(), v.clone(), mask) {
                Ok((output, _, _, _)) => {
                    return output.to_dtype(dtype).map_err(|e| AttentionError::Backend {
                        message: e.to_string(),
                    });
                }
                Err(err) => {
                    log::warn!("fused backend failed, falling back to reference: {}", err);
                }
            }
        } else {
            log::debug!(
                "attention backend=reference batch={} heads={} q_len={} head_dim={} dtype={:?}",
                batch,
                heads,
                q_len,
                head_dim,
                dtype
            );
        }

        self.reference.attend(q, k, v, mask, config)
    }
}

fn select_tile_size(k_len: usize, head_dim: usize) -> usize {
    let base = if head_dim <= 64 {
        128
    } else if head_dim <= 128 {
        96
    } else {
        64
    };
    base.min(k_len.max(1))
}

fn to_backend_err(err: candle_core::Error) -> AttentionError {
    AttentionError::Backend {
        message: err.to_string(),
    }
}

trait MulScalarExt {
    fn mul_scalar(&self, value: f32) -> CandleResult<Tensor>;
}

impl MulScalarExt for Tensor {
    fn mul_scalar(&self, value: f32) -> CandleResult<Tensor> {
        let scalar = Tensor::new(value, self.device())?;
        let scalar = scalar.broadcast_as(self.shape())?;
        self.mul(&scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::ops::softmax_last_dim;
    use crate::core::{BackendSelection, Config};
    use crate::masks::build_causal_mask;
    use crate::reference::ExactAttention;
    use candle_core::{DType, Device};

    fn compute_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let dtype = q.dtype();
        let (batch, heads, q_len, head_dim) = q.dims4().unwrap();
        let k_len = k.dims()[2];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let merged = batch * heads;
        let q_f32 = q.to_dtype(DType::F32).unwrap();
        let k_f32 = k.to_dtype(DType::F32).unwrap();
        let v_f32 = v.to_dtype(DType::F32).unwrap();
        let mask = mask.clone();

        let q_view = q_f32.reshape((merged, q_len, head_dim)).unwrap();
        let k_view = k_f32.reshape((merged, k_len, head_dim)).unwrap();
        let k_t = k_view.transpose(1, 2).unwrap();
        let scores = q_view
            .matmul(&k_t)
            .unwrap()
            .reshape((batch, heads, q_len, k_len))
            .unwrap()
            .mul_scalar(scale)
            .unwrap();
        let scores = scores.add(&mask).unwrap();
        let probs = softmax_last_dim(&scores).unwrap();
        let probs_f = probs.to_dtype(DType::F32).unwrap();
        let probs_2d = probs_f.reshape((merged, q_len, k_len)).unwrap();
        let v_2d = v_f32.reshape((merged, k_len, head_dim)).unwrap();
        let output = probs_2d
            .matmul(&v_2d)
            .unwrap()
            .reshape((batch, heads, q_len, head_dim))
            .unwrap();
        let log_norm = scores.max_keepdim(3).unwrap();
        let shifted = scores
            .sub(&log_norm.broadcast_as((batch, heads, q_len, k_len)).unwrap())
            .unwrap();
        let exp = shifted.exp().unwrap();
        let sum_exp = exp.sum_keepdim(3).unwrap();

        (output.to_dtype(dtype).unwrap(), log_norm, sum_exp)
    }

    fn compute_fused(
        kernel: &FusedKernel,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let (output, max_scores, sum_exp, _) = kernel
            .fused_forward(q, k.clone(), v.clone(), Some(mask))
            .unwrap();
        (output, max_scores, sum_exp)
    }

    #[test]
    #[ignore = "expensive parity sweep"]
    fn parity_against_reference() {
        let device = Device::Cpu;
        let fused = FusedAttention::new();
        let reference = ExactAttention::new();
        let config = Config::default();

        let batch = 1usize;
        let heads = 4usize;
        let seq_lengths = [32usize, 128, 512, 2048, 8192];
        let head_dims = [64usize, 80, 128];
        let dtypes = [DType::F32, DType::BF16, DType::F16];

        for &seq_len in &seq_lengths {
            for &head_dim in &head_dims {
                for &dtype in &dtypes {
                    let q = Tensor::rand(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                    let k = Tensor::rand(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                    let v = Tensor::rand(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                    let mask = build_causal_mask(&device, batch, heads, seq_len, seq_len).unwrap();

                    let fused_out = fused.attend(&q, &k, &v, Some(&mask), &config).unwrap();
                    let reference_out = reference.attend(&q, &k, &v, Some(&mask), &config).unwrap();

                    let (_fused_num, fused_max, fused_sum) =
                        compute_fused(&fused.kernel, &q, &k, &v, &mask);
                    let (_reference_num, reference_max, reference_sum) =
                        compute_reference(&q, &k, &v, &mask);

                    let diff = fused_out
                        .to_dtype(DType::F32)
                        .unwrap()
                        .sub(&reference_out.to_dtype(DType::F32).unwrap())
                        .unwrap()
                        .abs()
                        .unwrap();
                    let max_diff = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
                    assert!(max_diff < 5e-2, "max diff {} too high", max_diff);

                    let fused_log = fused_max.add(&fused_sum.log().unwrap()).unwrap();
                    let reference_log = reference_max.add(&reference_sum.log().unwrap()).unwrap();
                    let log_diff = fused_log
                        .sub(&reference_log)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max_all()
                        .unwrap()
                        .to_vec0::<f32>()
                        .unwrap();
                    assert!(log_diff < 5e-2, "log diff {} too high", log_diff);
                }
            }
        }
    }

    #[test]
    fn backend_override_respected() {
        let device = Device::Cpu;
        let fused = FusedAttention::new();
        let q = Tensor::rand(0.0f32, 1.0, (1, 2, 128, 64), &device).unwrap();
        let k = q.clone();
        let v = q.clone();
        let mask = build_causal_mask(&device, 1, 2, 128, 128).unwrap();

        let mut config = Config::default();
        config.backend = BackendSelection::ReferenceOnly;
        let out_ref = fused.attend(&q, &k, &v, Some(&mask), &config).unwrap();

        config.backend = BackendSelection::FusedOnly;
        let out_fused = fused.attend(&q, &k, &v, Some(&mask), &config).unwrap();

        let diff = out_ref
            .to_dtype(DType::F32)
            .unwrap()
            .sub(&out_fused.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(diff < 5e-2);
    }
}
