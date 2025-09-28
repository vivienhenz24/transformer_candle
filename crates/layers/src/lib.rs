//! Building blocks for transformer layers.
//!
//! The `layers` crate defines the shared components used to assemble end to end
//! decoder blocks. Each submodule focuses on one aspect of the transformer
//! stack, exposing well documented primitives so higher level crates can compose
//! them freely.
//!
//! # Tensor shape conventions
//!
//! Unless otherwise stated, tensors follow the `(batch, seq, hidden)` layout.
//! Multi-way projections (e.g. query, key, value) use the fused
//! `[batch, seq, projection * hidden]` convention to minimize reshapes before a
//! split. Head-shaped tensors adopt `(batch, num_heads, seq, head_dim)` for
//! attention-style operations, while feed-forward activations keep
//! `(batch, seq, ff_hidden)`.
//!
//! Shapes are documented at the module entry points so downstream callers can
//! spot the expected layout without spelunking implementations.
//!
//! # Precision and dtype policy
//!
//! Parameters typically live in reduced precision (`f16`/`bf16`) for storage
//! efficiency, while compute-critical paths are promoted to `f32`. Reductions
//! (layer norms, residual aggregations) also happen in `f32` before casting back
//! to the requested output dtype to avoid catastrophic cancellation. These rules
//! are centralised in [`dtypes::PrecisionPolicy`], mirroring the policy used by
//! the attention crate, so every layer component obeys the same casting
//! semantics and epsilon tuning.
//!
//! # Module overview
//!
//! * [`norm`] — layer norm, RMSNorm, and related normalisation utilities.
//! * [`linear`] — affine projections and packed multi-projection helpers.
//! * [`mlp`] — position-wise feed-forward blocks and activation glue.
//! * [`residual`] — residual connections, dropout wrappers, and block wiring.
//! * [`activations`] — activation catalogue with tensor-friendly interfaces.
//! * [`dtypes`] — precision policies, casting helpers, and numeric stability.
//! * [`checks`] — shape and dtype assertions shared across layer code.

pub mod activations;
pub mod checks;
pub mod dtypes;
pub mod linear;
pub mod mlp;
pub mod norm;
pub mod residual;

pub use crate::dtypes::{PrecisionEpsilons, PrecisionPolicy};
