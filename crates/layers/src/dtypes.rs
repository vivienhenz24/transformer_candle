//! Precision and dtype policy utilities used throughout the layers crate.
//!
//! Parameters typically reside in `f16`/`bf16` for memory efficiency while
//! compute-intensive paths promote tensors to `f32`. Reductions and numerical
//! stability checks also favour `f32` to mirror the behaviour of the attention
//! crate. This module exposes [`PrecisionPolicy`] so callers can consistently
//! cast tensors before matmuls, reductions, or final outputs.

use candle_core::{DType, Result, Tensor};

/// Epsilon values used for comparisons at different stages of a computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionEpsilons {
    /// Tolerance for tensors stored on disk or in module parameters.
    pub storage: f32,
    /// Tolerance for intermediate matmul/activation results.
    pub compute: f32,
    /// Tolerance for statistics computed during reductions.
    pub reduction: f32,
}

/// Describes how tensors should be cast during different phases of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionPolicy {
    storage: DType,
    compute: DType,
    reduction: DType,
}

impl PrecisionPolicy {
    /// Constructs a new policy from explicit dtype selections.
    pub fn new(storage: DType, compute: DType, reduction: DType) -> Self {
        Self {
            storage,
            compute,
            reduction,
        }
    }

    /// Builds a policy from the parameter storage dtype.
    pub fn from_parameter_dtype(storage: DType) -> Self {
        let compute = match storage {
            DType::F16 | DType::BF16 => DType::F32,
            other => other,
        };
        let reduction = DType::F32;
        Self::new(storage, compute, reduction)
    }

    /// Returns the dtype used to store parameters and outputs.
    pub fn storage(&self) -> DType {
        self.storage
    }

    /// Returns the dtype used for matmuls and activation evaluation.
    pub fn compute(&self) -> DType {
        self.compute
    }

    /// Returns the dtype used for reductions such as layer norm statistics.
    pub fn reduction(&self) -> DType {
        self.reduction
    }

    /// Indicates whether the policy performs mixed precision work.
    pub fn is_mixed_precision(&self) -> bool {
        self.storage != self.compute || self.compute != self.reduction
    }

    /// Updates the compute dtype, keeping reduction at least as wide.
    pub fn with_compute(mut self, compute: DType) -> Self {
        self.compute = compute;
        if matches!(compute, DType::F64 | DType::F32) {
            self.reduction = compute;
        }
        self
    }

    /// Tolerance values derived from the configured dtypes.
    pub fn epsilons(&self) -> PrecisionEpsilons {
        PrecisionEpsilons {
            storage: epsilon_for(self.storage),
            compute: epsilon_for(self.compute),
            reduction: epsilon_for(self.reduction),
        }
    }

    /// Casts a tensor to the compute dtype for matmul readiness.
    pub fn cast_for_matmul(&self, tensor: &Tensor) -> Result<Tensor> {
        cast_tensor(tensor, self.compute)
    }

    /// Casts a tensor to the reduction dtype for statistics.
    pub fn cast_for_reduction(&self, tensor: &Tensor) -> Result<Tensor> {
        cast_tensor(tensor, self.reduction)
    }

    /// Casts a tensor back to the storage dtype (or leaves it unchanged).
    pub fn cast_to_storage(&self, tensor: &Tensor) -> Result<Tensor> {
        cast_tensor(tensor, self.storage)
    }

    /// Sums all elements after promoting to the reduction dtype.
    pub fn reduce_sum(&self, tensor: &Tensor, output: Option<DType>) -> Result<Tensor> {
        let promoted = self.cast_for_reduction(tensor)?;
        let summed = promoted.sum_all()?;
        let target = output.unwrap_or(self.storage);
        cast_tensor(&summed, target)
    }
}

fn cast_tensor(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor.clone())
    } else {
        tensor.to_dtype(dtype)
    }
}

fn epsilon_for(dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => 2e-2,
        DType::F16 => 5e-3,
        DType::F32 => 1e-5,
        DType::F64 => 1e-7,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn policy_promotes_reduced_precision_parameters() {
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F16);
        assert_eq!(policy.storage(), DType::F16);
        assert_eq!(policy.compute(), DType::F32);
        assert_eq!(policy.reduction(), DType::F32);
        assert!(policy.is_mixed_precision());
    }

    #[test]
    fn cast_round_trip_preserves_values_within_tolerance() -> Result<()> {
        let device = Device::Cpu;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::BF16);
        let base = Tensor::from_vec(vec![0.125f32, -0.75, 3.5], (3,), &device)?;
        let storage = base.to_dtype(policy.storage())?;

        let compute = policy.cast_for_matmul(&storage)?;
        assert_eq!(compute.dtype(), policy.compute());

        let round_trip = policy.cast_to_storage(&compute)?;
        let original = base.to_vec1::<f32>()?;
        let restored = round_trip.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let eps = policy.epsilons().storage;
        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() <= eps);
        }
        Ok(())
    }

    #[test]
    fn reductions_promote_to_f32_and_cast_back() -> Result<()> {
        let device = Device::Cpu;
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F16);
        let data = vec![1.0f32; 1024];
        let tensor = Tensor::from_vec(data, (1024,), &device)?.to_dtype(policy.storage())?;

        let reduced = policy.reduce_sum(&tensor, Some(policy.storage()))?;
        assert_eq!(reduced.dtype(), policy.storage());
        let value = reduced.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        assert!((value - 1024.0).abs() < 1.0);

        let promoted = policy.cast_for_reduction(&tensor)?;
        assert_eq!(promoted.dtype(), policy.reduction());
        Ok(())
    }

    #[test]
    fn epsilons_track_dtype_expectations() {
        let policy = PrecisionPolicy::from_parameter_dtype(DType::F32);
        let eps = policy.epsilons();
        assert!(eps.compute < 1e-4);
        assert_eq!(eps.compute, eps.reduction);
    }
}
