//! Reference CPU/PTX-friendly attention kernels.
//!
//! The exact path prioritises numerical fidelity and mirrors the semantics
//! described by the [`Attention`](crate::core::Attention) trait.

use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::ops::{dropout, softmax_last_dim};
use embedding::positional::rope::{scaling_fingerprint, RopeConfig};

use crate::core::{Attention, AttentionError, Config, PrecisionPolicy, RopeMode};
use crate::interop::RopeAdapter;
use crate::masks::MASK_DTYPE;

/// Numerically stable, portable attention kernel with optional RoPE support.
#[derive(Debug)]
pub struct ExactAttention {
    rope_config: Option<RopeConfig>,
    rope_state: Mutex<Option<(DeviceKind, RopeAdapter)>>,
    first_call: OnceLock<()>,
}

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

impl ExactAttention {
    /// Construct a reference attention kernel without RoPE support.
    pub fn new() -> Self {
        Self {
            rope_config: None,
            rope_state: Mutex::new(None),
            first_call: OnceLock::new(),
        }
    }

    /// Construct a reference attention kernel configured with rotary embeddings.
    pub fn with_rope(rope_config: RopeConfig) -> Self {
        Self {
            rope_config: Some(rope_config),
            rope_state: Mutex::new(None),
            first_call: OnceLock::new(),
        }
    }
}

impl Default for ExactAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl Attention for ExactAttention {
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        config: &Config,
    ) -> Result<Tensor, AttentionError> {
        let mut cfg = config.clone();
        cfg.apply_env_overrides();

        if self.first_call.set(()).is_ok() {
            let rope_summary = self
                .rope_config
                .as_ref()
                .map(|rc| scaling_fingerprint(rc, rc.head_dim))
                .unwrap_or_else(|| "none".to_string());
            log::info!(
                "attention::reference init backend={:?} precision={:?} padding_mask={} rope_mode={:?} kv_enabled={} kv_page_size={:?} rope={}",
                cfg.backend,
                cfg.precision,
                cfg.use_padding_mask,
                cfg.rope_mode,
                cfg.kv.enabled,
                cfg.kv.page_size,
                rope_summary
            );
        }

        let device = q.device();
        if !device.same_device(k.device()) || !device.same_device(v.device()) {
            return Err(AttentionError::InvalidShape {
                context: "q, k, v must reside on the same device".to_string(),
            });
        }

        let dtype = q.dtype();
        if dtype != k.dtype() || dtype != v.dtype() {
            return Err(AttentionError::InvalidShape {
                context: "q, k, v must share the same dtype".to_string(),
            });
        }

        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
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

        let mut q_tensor = q.clone();
        let mut k_tensor = k.clone();
        if cfg.rope_mode == RopeMode::OnTheFly {
            if let Some(rcfg) = &self.rope_config {
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
                    *guard = Some((kind.clone(), RopeAdapter::new(rcfg.clone(), device.clone())));
                }
                if let Some((_, adapter)) = guard.as_mut() {
                    let (q_rot, k_rot) =
                        adapter
                            .apply(q, k, 0)
                            .map_err(|err| AttentionError::Backend {
                                message: err.to_string(),
                            })?;
                    q_tensor = q_rot;
                    k_tensor = k_rot;
                }
            }
        }

        let (q_work, k_work, v_work) = match cfg.precision {
            PrecisionPolicy::ForceF32 => (
                q_tensor
                    .to_dtype(DType::F32)
                    .map_err(|e| AttentionError::Backend {
                        message: e.to_string(),
                    })?,
                k_tensor
                    .to_dtype(DType::F32)
                    .map_err(|e| AttentionError::Backend {
                        message: e.to_string(),
                    })?,
                v.to_dtype(DType::F32)
                    .map_err(|e| AttentionError::Backend {
                        message: e.to_string(),
                    })?,
            ),
            PrecisionPolicy::Inherit => (q_tensor.clone(), k_tensor.clone(), v.clone()),
        };

        let merged = batch * heads;
        let q_view =
            q_work
                .reshape((merged, q_len, head_dim))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;
        let k_view =
            k_work
                .reshape((merged, k_len, head_dim))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;
        let k_t = k_view
            .transpose(1, 2)
            .map_err(|e| AttentionError::Backend {
                message: e.to_string(),
            })?;
        let mut scores = q_view.matmul(&k_t).map_err(|e| AttentionError::Backend {
            message: e.to_string(),
        })?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        scores = scores
            .mul_scalar(scale)
            .map_err(|e| AttentionError::Backend {
                message: e.to_string(),
            })?;
        let mut scores =
            scores
                .reshape((batch, heads, q_len, k_len))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;

        if cfg.use_padding_mask {
            if let Some(mask) = mask {
                if !device.same_device(mask.device()) {
                    return Err(AttentionError::InvalidShape {
                        context: "mask must reside on the same device as q".to_string(),
                    });
                }
                if mask.dtype() != MASK_DTYPE {
                    return Err(AttentionError::UnsupportedDType {
                        requested: format!(
                            "mask expects dtype {MASK_DTYPE:?}, got {:?}",
                            mask.dtype()
                        ),
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
                let mask_applied = if mh == heads {
                    mask.clone()
                } else {
                    mask.broadcast_as((batch, heads, q_len, k_len))
                        .map_err(|e| AttentionError::Backend {
                            message: e.to_string(),
                        })?
                };
                scores = scores
                    .add(&mask_applied)
                    .map_err(|e| AttentionError::Backend {
                        message: e.to_string(),
                    })?;
            }
        }

        let scores_2d =
            scores
                .reshape((merged, q_len, k_len))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;
        let probs = softmax_last_dim(&scores_2d).map_err(|e| AttentionError::Backend {
            message: e.to_string(),
        })?;
        let probs =
            probs
                .reshape((batch, heads, q_len, k_len))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;

        let probs = if let Some(dropout_p) = cfg.dropout_p {
            if dropout_p < 0.0 || dropout_p >= 1.0 {
                return Err(AttentionError::InvalidShape {
                    context: format!("dropout probability must be in [0, 1), got {dropout_p}"),
                });
            }
            if dropout_p > 0.0 {
                dropout(&probs, dropout_p).map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?
            } else {
                probs
            }
        } else {
            probs
        };

        let probs_2d =
            probs
                .reshape((merged, q_len, k_len))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;
        let v_view =
            v_work
                .reshape((merged, k_len, head_dim))
                .map_err(|e| AttentionError::Backend {
                    message: e.to_string(),
                })?;
        let output = probs_2d
            .matmul(&v_view)
            .map_err(|e| AttentionError::Backend {
                message: e.to_string(),
            })?;
        let output = output
            .reshape((batch, heads, q_len, head_dim))
            .map_err(|e| AttentionError::Backend {
                message: e.to_string(),
            })?;

        output.to_dtype(dtype).map_err(|e| AttentionError::Backend {
            message: e.to_string(),
        })
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
    use crate::core::{BackendSelection, Config};
    use crate::masks::build_causal_mask;
    use candle_core::{DType as CandleDType, Device};

    fn build_inputs(device: &Device) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let q = Tensor::from_vec(data.clone(), (1, 2, 4, 8), device)?;
        let k = Tensor::from_vec(data.clone(), (1, 2, 4, 8), device)?;
        let v = Tensor::from_vec(data, (1, 2, 4, 8), device)?;
        Ok((q, k, v))
    }

    fn naive_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (batch, heads, q_len, head_dim) = q.dims4()?;
        let (_, _, k_len, _) = k.dims4()?;
        let mut output = vec![0f32; batch * heads * q_len * head_dim];

        let q_vec = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let k_vec = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_vec = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mask_vec = if let Some(m) = mask {
            Some(m.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        let scale = 1.0 / (head_dim as f32).sqrt();

        for b in 0..batch {
            for h in 0..heads {
                for q_idx in 0..q_len {
                    let mut row = vec![0f32; k_len];
                    let mut max_val = f32::NEG_INFINITY;
                    for k_idx in 0..k_len {
                        let mut dot = 0f32;
                        for d in 0..head_dim {
                            let qi = (((b * heads + h) * q_len + q_idx) * head_dim + d) as usize;
                            let ki = (((b * heads + h) * k_len + k_idx) * head_dim + d) as usize;
                            dot += q_vec[qi] * k_vec[ki];
                        }
                        dot *= scale;
                        if let Some(mask_vec) = &mask_vec {
                            let mi = (((b * heads + h) * q_len + q_idx) * k_len + k_idx) as usize;
                            dot += mask_vec[mi];
                        }
                        row[k_idx] = dot;
                        if dot.is_finite() && dot > max_val {
                            max_val = dot;
                        }
                    }
                    let mut denom = 0f32;
                    for val in row.iter_mut() {
                        if *val == f32::NEG_INFINITY {
                            *val = 0.0;
                        } else {
                            *val = (*val - max_val).exp();
                            denom += *val;
                        }
                    }
                    if denom == 0.0 {
                        continue;
                    }
                    for d in 0..head_dim {
                        let mut acc = 0f32;
                        for k_idx in 0..k_len {
                            let weight = row[k_idx] / denom;
                            let vi = (((b * heads + h) * k_len + k_idx) * head_dim + d) as usize;
                            acc += weight * v_vec[vi];
                        }
                        let oi = (((b * heads + h) * q_len + q_idx) * head_dim + d) as usize;
                        output[oi] = acc;
                    }
                }
            }
        }

        Tensor::from_vec(output, (batch, heads, q_len, head_dim), q.device())
    }

    #[test]
    fn exact_attention_matches_naive() -> CandleResult<()> {
        let device = Device::Cpu;
        let (q, k, v) = build_inputs(&device)?;
        let mask = build_causal_mask(&device, 1, 2, 4, 4)?;
        let attention = ExactAttention::default();
        let config = Config::default();
        let output = attention.attend(&q, &k, &v, Some(&mask), &config).unwrap();
        let expected = naive_attention(&q, &k, &v, Some(&mask))?;
        let diff = output
            .to_dtype(DType::F32)?
            .sub(&expected.to_dtype(DType::F32)?)?
            .abs()?;
        let max = diff.max_all()?.to_vec0::<f32>()?;
        assert!(max < 1e-4);
        Ok(())
    }

    #[test]
    fn mismatched_shapes_error() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 5, 8), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
        let attention = ExactAttention::default();
        let err = attention
            .attend(&q, &k, &v, None, &Config::default())
            .unwrap_err();
        assert!(matches!(err, AttentionError::InvalidShape { .. }));
    }

    #[test]
    fn mask_shape_validation() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((1, 3, 4, 4), DType::F32, &device).unwrap();
        let attention = ExactAttention::default();
        let err = attention
            .attend(&q, &k, &v, Some(&mask), &Config::default())
            .unwrap_err();
        assert!(matches!(err, AttentionError::InvalidShape { .. }));
    }

    #[test]
    fn dtype_matrix() -> CandleResult<()> {
        let device = Device::Cpu;
        let (q, k, v) = build_inputs(&device)?;
        let mask = build_causal_mask(&device, 1, 2, 4, 4)?;
        let reference = ExactAttention::default()
            .attend(&q, &k, &v, Some(&mask), &Config::default())
            .unwrap()
            .to_dtype(DType::F32)?;
        for dtype in [CandleDType::F32, CandleDType::BF16, CandleDType::F16] {
            let q_cast = q.to_dtype(dtype)?;
            let k_cast = k.to_dtype(dtype)?;
            let v_cast = v.to_dtype(dtype)?;
            let out = ExactAttention::default()
                .attend(&q_cast, &k_cast, &v_cast, Some(&mask), &Config::default())
                .unwrap()
                .to_dtype(DType::F32)?;
            let diff = out.sub(&reference)?.abs()?;
            let max = diff.max_all()?.to_vec0::<f32>()?;
            assert!(max < 5e-2, "dtype {:?} diverged by {max}", dtype);
        }
        Ok(())
    }

    #[test]
    fn numerical_stability() {
        let device = Device::Cpu;
        let q = Tensor::full(10_000.0f32, (1, 1, 4, 4), &device).unwrap();
        let k = Tensor::full(-10_000.0f32, (1, 1, 4, 4), &device).unwrap();
        let v = Tensor::ones((1, 1, 4, 4), DType::F32, &device).unwrap();
        let attention = ExactAttention::default();
        let out = attention
            .attend(&q, &k, &v, None, &Config::default())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(out.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn dropout_zero_probability_is_noop() {
        let device = Device::Cpu;
        let (q, k, v) = build_inputs(&device).unwrap();
        let mask = build_causal_mask(&device, 1, 2, 4, 4).unwrap();
        let config = Config {
            dropout_p: Some(0.0),
            ..Config::default()
        };
        let out = ExactAttention::default()
            .attend(&q, &k, &v, Some(&mask), &config)
            .unwrap();
        let reference = ExactAttention::default()
            .attend(&q, &k, &v, Some(&mask), &Config::default())
            .unwrap();
        let diff = out
            .to_dtype(DType::F32)
            .unwrap()
            .sub(&reference.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap();
        let max = diff.max_all().unwrap().to_vec0::<f32>().unwrap();
        assert!(max < 1e-5);
    }
}
