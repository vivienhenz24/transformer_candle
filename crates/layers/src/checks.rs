//! Shared validation utilities for layer components.
//!
//! All helpers return `candle_core::Result<()>` with actionable error messages so
//! callers can fail fast before entering compute intensive paths.

use candle_core::{DType, Error, Result, Tensor};

fn format_dims(dims: &[usize]) -> String {
    let mut parts = Vec::with_capacity(dims.len());
    for dim in dims {
        parts.push(dim.to_string());
    }
    format!("[{}]", parts.join(", "))
}

/// Ensures `tensor` matches `expected` exactly.
pub fn expect_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    let actual = tensor.dims();
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensor `{}` expected shape {} but received {}",
            name,
            format_dims(expected),
            format_dims(actual)
        )))
    }
}

/// Validates the `(batch, seq, hidden)` layout with a concrete hidden size.
pub fn expect_batch_seq_hidden(name: &str, tensor: &Tensor, hidden: usize) -> Result<()> {
    let dims = tensor.dims();
    match dims {
        [_, _, current] if *current == hidden => Ok(()),
        _ => Err(Error::Msg(format!(
            "tensor `{}` expected shape [batch, seq, hidden={}] but received {}",
            name,
            hidden,
            format_dims(dims)
        ))),
    }
}

/// Checks `tensor` rank against `expected`.
pub fn expect_rank(name: &str, tensor: &Tensor, expected: usize) -> Result<()> {
    let actual = tensor.rank();
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensor `{}` expected rank {} but received {}",
            name, expected, actual
        )))
    }
}

/// Ensures `tensor` is contiguous in memory.
pub fn expect_contiguous(name: &str, tensor: &Tensor) -> Result<()> {
    if tensor.is_contiguous() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensor `{}` must be contiguous but has non-contiguous strides",
            name
        )))
    }
}

/// Ensures the dtype matches `expected` exactly.
pub fn expect_dtype(name: &str, tensor: &Tensor, expected: DType) -> Result<()> {
    let actual = tensor.dtype();
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensor `{}` expected dtype {:?} but received {:?}",
            name, expected, actual
        )))
    }
}

/// Validates `tensor` dtype is within the allowed list.
pub fn expect_dtype_in(name: &str, tensor: &Tensor, allowed: &[DType]) -> Result<()> {
    let actual = tensor.dtype();
    if allowed.iter().copied().any(|dt| dt == actual) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensor `{}` expected dtype in {:?} but received {:?}",
            name, allowed, actual
        )))
    }
}

/// Ensures two tensors share the same dtype.
pub fn expect_same_dtype(a_name: &str, a: &Tensor, b_name: &str, b: &Tensor) -> Result<()> {
    if a.dtype() == b.dtype() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensors `{}` and `{}` must share dtype (left: {:?}, right: {:?})",
            a_name,
            b_name,
            a.dtype(),
            b.dtype()
        )))
    }
}

/// Ensures casts are restricted to supported float dtypes used in the project.
pub fn ensure_cast_supported(name: &str, from: DType, to: DType) -> Result<()> {
    const SUPPORTED: &[DType] = &[DType::F16, DType::BF16, DType::F32, DType::F64];
    let supported = |dt: DType| SUPPORTED.iter().copied().any(|d| d == dt);
    if from == to {
        return Ok(());
    }
    if supported(from) && supported(to) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "tensor `{}` cannot be cast from {:?} to {:?} (unsupported dtype)",
            name, from, to
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn shape_error_includes_name_and_dims() -> Result<()> {
        let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let err = expect_shape("weights", &tensor, &[4, 3]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("weights"));
        assert!(msg.contains("[4, 3]"));
        assert!(msg.contains("[2, 3]"));
        Ok(())
    }

    #[test]
    fn batch_seq_hidden_error_is_descriptive() -> Result<()> {
        let tensor = Tensor::zeros((1, 5, 7), DType::F32, &Device::Cpu)?;
        let err = expect_batch_seq_hidden("residual", &tensor, 8).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("residual"));
        assert!(msg.contains("hidden=8"));
        assert!(msg.contains("[1, 5, 7]"));
        Ok(())
    }

    #[test]
    fn dtype_mismatch_is_reported() -> Result<()> {
        let tensor = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
        let err = expect_dtype("bias", &tensor, DType::F16).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bias"));
        assert!(msg.contains("F16"));
        assert!(msg.contains("F32"));
        Ok(())
    }

    #[test]
    fn contiguity_check_detects_non_contiguous() -> Result<()> {
        let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
        let non_contiguous = tensor.transpose(0, 1)?;
        let err = expect_contiguous("activation", &non_contiguous).unwrap_err();
        assert!(err.to_string().contains("activation"));
        Ok(())
    }

    #[test]
    fn ensure_cast_supported_blocks_unsupported() {
        let err = ensure_cast_supported("state", DType::U8, DType::F16).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("state"));
        assert!(msg.contains("U8"));
        assert!(msg.contains("F16"));
    }
}
