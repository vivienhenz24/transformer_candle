//! Position-wise feed-forward blocks built on top of projections and activations.
//!
//! MLPs operate on hidden states shaped `(batch, seq, hidden)` and return the
//! same layout. Intermediate projections expand the hidden dimension to
//! `config.intermediate_size`, apply an activation, then contract back to the
//! model hidden size. Gated variants expect fused projections in
//! `(batch, seq, 2 * intermediate_size)` before splitting along the last axis.
//!
//! The block honours the crate-wide [`PrecisionPolicy`]: inputs are promoted to
//! the compute dtype for matmuls and activation evaluation, while outputs are
//! cast back to the storage dtype to preserve determinism across mixed precision
//! runs.

use std::fmt;
use std::sync::Arc;

use candle_core::{DType, Device, Error, Result, Tensor};

use crate::{
    activations::{self, Activation, ActivationKind},
    checks,
    dtypes::PrecisionPolicy,
    linear::{Linear, LinearConfig, LinearInit, LinearLayer},
};

/// Configuration shared by transformer feed-forward networks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedForwardConfig {
    /// Model hidden size.
    pub hidden_size: usize,
    /// Width of the activation space.
    pub intermediate_size: usize,
    /// Activation applied between projections.
    pub activation: ActivationKind,
    /// Whether the MLP uses a gated projection (e.g. GEGLU/SwiGLU).
    pub gated: bool,
}

impl FeedForwardConfig {
    /// Creates a standard two-projection MLP configuration.
    pub fn new(hidden_size: usize, intermediate_size: usize, activation: ActivationKind) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation,
            gated: false,
        }
    }

    /// Creates a configuration given an expansion ratio (e.g. `ratio = 4.0`).
    pub fn with_expansion_ratio(
        hidden_size: usize,
        ratio: f32,
        activation: ActivationKind,
    ) -> Self {
        let width = ((hidden_size as f32) * ratio).round() as usize;
        Self::new(hidden_size, width.max(1), activation)
    }

    /// Returns the implied expansion ratio (`intermediate / hidden`).
    pub fn expansion_ratio(&self) -> f32 {
        self.intermediate_size as f32 / self.hidden_size as f32
    }
}

/// Shared interface for feed-forward stacks.
pub trait FeedForwardLayer: Send + Sync {
    /// Configuration metadata used during block assembly.
    fn config(&self) -> &FeedForwardConfig;

    /// Performs the forward pass through the MLP.
    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor>;
}

/// Transformer feed-forward block with optional gating (GEGLU/SiGLU).
#[derive(Clone)]
pub struct FeedForward {
    config: FeedForwardConfig,
    up_proj: Linear,
    down_proj: Linear,
    activation: Arc<dyn Activation>,
}

impl fmt::Debug for FeedForward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FeedForward")
            .field("config", &self.config)
            .field("activation", &self.activation.kind())
            .finish()
    }
}

impl FeedForward {
    /// Builds a feed-forward layer from explicit projections and activation.
    pub fn new(
        config: FeedForwardConfig,
        up_proj: Linear,
        down_proj: Linear,
        activation: Arc<dyn Activation>,
    ) -> Result<Self> {
        Self::validate(&config, &up_proj, &down_proj)?;
        Ok(Self {
            config,
            up_proj,
            down_proj,
            activation,
        })
    }

    /// Constructs a feed-forward block with freshly initialised projections.
    pub fn with_init(
        config: FeedForwardConfig,
        activation: ActivationKind,
        up_init: &LinearInit,
        down_init: &LinearInit,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let multiplier = if config.gated { 2 } else { 1 };
        let mut up_cfg =
            LinearConfig::new(config.hidden_size, config.intermediate_size * multiplier);
        up_cfg.bias = true;
        let mut down_cfg = LinearConfig::new(config.intermediate_size, config.hidden_size);
        down_cfg.bias = true;

        let up_proj = Linear::with_init(up_cfg, up_init, device, dtype)?;
        let down_proj = Linear::with_init(down_cfg, down_init, device, dtype)?;
        Self::new(config, up_proj, down_proj, activations::builtin(activation))
    }

    fn validate(config: &FeedForwardConfig, up: &Linear, down: &Linear) -> Result<()> {
        let expected_up = config.intermediate_size * if config.gated { 2 } else { 1 };
        if up.config().input_dim != config.hidden_size
            || up.config().total_output_dim() != expected_up
        {
            return Err(Error::Msg(format!(
                "up projection shape mismatch: expected ({}, {}), got ({}, {})",
                config.hidden_size,
                expected_up,
                up.config().input_dim,
                up.config().total_output_dim()
            )));
        }
        if down.config().input_dim != config.intermediate_size
            || down.config().total_output_dim() != config.hidden_size
        {
            return Err(Error::Msg(format!(
                "down projection shape mismatch: expected ({}, {}), got ({}, {})",
                config.intermediate_size,
                config.hidden_size,
                down.config().input_dim,
                down.config().total_output_dim()
            )));
        }
        Ok(())
    }

    fn apply_activation(&self, tensor: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        self.activation.forward(tensor, policy)
    }
}

impl FeedForwardLayer for FeedForward {
    fn config(&self) -> &FeedForwardConfig {
        &self.config
    }

    fn forward(&self, hidden: &Tensor, policy: &PrecisionPolicy) -> Result<Tensor> {
        checks::expect_batch_seq_hidden("mlp.input", hidden, self.config.hidden_size)?;
        let up_out = self.up_proj.forward(hidden, policy)?;

        let activated = if self.config.gated {
            let compute = policy.cast_for_matmul(&up_out)?;
            checks::expect_rank("mlp.up", &compute, 3)?;
            let last = compute.rank() - 1;
            let split = self.config.intermediate_size;
            let value = compute.narrow(last, 0, split)?;
            let gate = compute.narrow(last, split, split)?;
            let gate_act = self.apply_activation(&policy.cast_to_storage(&gate)?, policy)?;
            let gate_compute = policy.cast_for_matmul(&gate_act)?;
            let fused = value.mul(&gate_compute)?;
            policy.cast_to_storage(&fused)?
        } else {
            self.apply_activation(&up_out, policy)?
        };

        self.down_proj.forward(&activated, policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn build_linear(
        input_dim: usize,
        output_dim: usize,
        dtype: DType,
        device: &Device,
        data: Vec<f32>,
    ) -> Result<Linear> {
        let mut cfg = LinearConfig::new(input_dim, output_dim);
        cfg.bias = false;
        let weight = Tensor::from_vec(data, (output_dim, input_dim), device)?.to_dtype(dtype)?;
        Linear::new(cfg, weight, None)
    }

    #[test]
    fn feed_forward_matches_manual_reference() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let config = FeedForwardConfig::new(4, 6, ActivationKind::Gelu);
        let up = build_linear(
            config.hidden_size,
            config.intermediate_size,
            dtype,
            &device,
            (0..24).map(|i| (i as f32) * 0.01 - 0.05).collect(),
        )?;
        let down = build_linear(
            config.intermediate_size,
            config.hidden_size,
            dtype,
            &device,
            (0..24).map(|i| (i as f32) * -0.02 + 0.03).collect(),
        )?;
        let ff = FeedForward::new(
            config.clone(),
            up.clone(),
            down.clone(),
            activations::builtin(ActivationKind::Gelu),
        )?;

        let input = Tensor::from_vec(
            vec![
                0.1f32, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, 1.3, -1.4, 1.5,
                -1.6,
            ],
            (2, 2, 4),
            &device,
        )?;
        let policy = PrecisionPolicy::from_parameter_dtype(dtype);
        let output = ff.forward(&input, &policy)?;

        let up_manual = up.forward(&input, &policy)?;
        let activated = activations::builtin(ActivationKind::Gelu).forward(&up_manual, &policy)?;
        let reference = down.forward(&activated, &policy)?;

        let diff = output
            .to_dtype(DType::F32)?
            .sub(&reference.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn gated_feed_forward_applies_swiglu() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut config = FeedForwardConfig::new(3, 5, ActivationKind::SwiGlu);
        config.gated = true;

        let up = build_linear(
            config.hidden_size,
            config.intermediate_size * 2,
            dtype,
            &device,
            vec![0.1; 30],
        )?;
        let down = build_linear(
            config.intermediate_size,
            config.hidden_size,
            dtype,
            &device,
            vec![0.05; 15],
        )?;
        let ff = FeedForward::new(
            config.clone(),
            up.clone(),
            down.clone(),
            activations::builtin(ActivationKind::SwiGlu),
        )?;

        let input = Tensor::from_vec(vec![1.0f32, -2.0, 3.0], (1, 1, 3), &device)?;
        let policy = PrecisionPolicy::from_parameter_dtype(dtype);
        let output = ff.forward(&input, &policy)?;

        let up_proj = up.forward(&input, &policy)?;
        let compute = policy.cast_for_matmul(&up_proj)?;
        let value = compute.narrow(2, 0, config.intermediate_size)?;
        let gate = compute.narrow(2, config.intermediate_size, config.intermediate_size)?;
        let gate_act = activations::builtin(ActivationKind::SwiGlu)
            .forward(&policy.cast_to_storage(&gate)?, &policy)?;
        let fused = value.mul(&policy.cast_for_matmul(&gate_act)?)?;
        let reference = down.forward(&policy.cast_to_storage(&fused)?, &policy)?;

        let diff = output
            .to_dtype(DType::F32)?
            .sub(&reference.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn feed_forward_respects_dtype_and_shape() -> Result<()> {
        let device = Device::Cpu;
        let weights = (0..64).map(|i| (i as f32) * 0.01).collect::<Vec<_>>();
        for &dtype in &[DType::F32, DType::F16, DType::BF16] {
            let config = FeedForwardConfig::new(4, 8, ActivationKind::Silu);
            let up = build_linear(
                config.hidden_size,
                config.intermediate_size,
                dtype,
                &device,
                weights[..32].to_vec(),
            )?;
            let down = build_linear(
                config.intermediate_size,
                config.hidden_size,
                dtype,
                &device,
                weights[32..64].to_vec(),
            )?;
            let ff = FeedForward::new(
                config.clone(),
                up,
                down,
                activations::builtin(ActivationKind::Silu),
            )?;
            let input =
                Tensor::randn(0f32, 1.0, (2, 3, config.hidden_size), &device)?.to_dtype(dtype)?;
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);
            let output = ff.forward(&input, &policy)?;
            assert_eq!(output.dims(), &[2, 3, config.hidden_size]);
            assert_eq!(output.dtype(), dtype);
        }
        Ok(())
    }

    #[test]
    fn forward_is_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let config = FeedForwardConfig::new(3, 6, ActivationKind::Gelu);
        let up = build_linear(
            config.hidden_size,
            config.intermediate_size,
            DType::F32,
            &device,
            vec![0.2; 18],
        )?;
        let down = build_linear(
            config.intermediate_size,
            config.hidden_size,
            DType::F32,
            &device,
            vec![0.1; 18],
        )?;
        let ff = FeedForward::new(config, up, down, activations::builtin(ActivationKind::Gelu))?;

        let input = Tensor::from_vec(vec![0.4f32, -0.1, 0.25], (1, 1, 3), &device)?;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let out1 = ff.forward(&input, &policy)?;
        let out2 = ff.forward(&input, &policy)?;
        let diff = out1.sub(&out2)?.abs()?.max_all()?.to_vec0::<f32>()?;
        assert!(diff <= 1e-7);
        Ok(())
    }
}
