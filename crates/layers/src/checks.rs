//! Lightweight validation helpers shared across layer components.
//!
//! These routines provide concise shape and dtype assertions that can be wired
//! into constructors or forward paths. They return `candle_core::Result<()>`
//! so call sites can propagate errors without panicking.

use candle_core::{DType, Error, Result, Tensor};

/// Ensures a tensor matches the expected dimensions exactly.
pub fn expect_shape(tensor: &Tensor, expected: &[usize]) -> Result<()> {
    let actual = tensor.dims().to_vec();
    if actual.as_slice() == expected {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "expected shape {:?}, got {:?}",
            expected, actual
        )))
    }
}

/// Validates the `(batch, seq, hidden)` convention with a known hidden size.
pub fn expect_batch_seq_hidden(tensor: &Tensor, hidden: usize) -> Result<()> {
    let dims = tensor.dims().to_vec();
    match dims.as_slice() {
        [_, _, actual_hidden] if *actual_hidden == hidden => Ok(()),
        _ => Err(Error::Msg(format!(
            "expected (batch, seq, {}) layout, got {:?}",
            hidden, dims
        ))),
    }
}

/// Checks the tensor dtype is one of the allowed values.
pub fn expect_dtype_in(tensor: &Tensor, allowed: &[DType]) -> Result<()> {
    let dtype = tensor.dtype();
    if allowed.iter().copied().any(|allowed| allowed == dtype) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "expected dtype in {:?}, got {:?}",
            allowed, dtype
        )))
    }
}
