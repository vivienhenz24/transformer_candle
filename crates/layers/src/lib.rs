//! Building blocks for transformer layers.
//!
//! The `layers` crate defines the shared components used to assemble end-to-end
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
//! * [`norm`] — LayerNorm and RMSNorm implementations with epsilon and affine
//!   controls. Inputs are `(batch, seq, hidden)` and statistics are computed in
//!   `policy.reduction()`.
//! * [`linear`] — affine projections with optional bias, fused multi-projection
//!   support, weight tying, and initialisation policies (Xavier, Kaiming,
//!   scaled variants).
//! * [`mlp`] — configurable feed-forward blocks with expansion ratios,
//!   activation plug-ins (GELU, SiLU, SwiGLU), and optional gating.
//! * [`residual`] — residual addition helpers, deterministic dropout masks, and
//!   pre/post-norm orchestration utilities.
//! * [`activations`] — activation catalogue operating in the requested compute
//!   dtype before casting back to storage precision.
//! * [`dtypes`] — precision policies, casting helpers, and epsilon tunings
//!   shared across modules.
//! * [`checks`] — centralised shape/dtype/contiguity assertions with actionable
//!   error messages.
//!
//! # Initialisation guidance
//!
//! * **Feed-forward / projection layers** — use [`LinearInit::XavierUniform`]
//!   for activations centred around zero (GELU, SiLU) and
//!   [`LinearInit::KaimingUniform`] when following ReLU-like activations. Deep
//!   decoder stacks can stabilise early training by scaling the initialisation
//!   via [`LinearInit::scaled`], e.g. `scaled(kaiming, 0.5)`.
//! * **Norm parameters** — start with unit scale and zero bias (default in the
//!   constructors) and fine-tune epsilon per architecture; RMSNorm typically
//!   uses `1e-5` while LayerNorm in GPT-style models often sits around `1e-5`
//!   or `1e-6`.
//!
//! # Residual ordering and attention
//!
//! Pre-norm residual blocks (norm → sublayer → dropout → residual add) keep the
//! gradient path short and work well with long context attention. Post-norm
//! blocks (sublayer → dropout → residual add → norm) can be easier to debug but
//! typically require more careful init/learning-rate tuning. Combine pre-norm
//! residuals with attention modules that assume pre-normalised inputs. When
//! mixing attention implementations, ensure both the attention stack and the
//! feed-forward stack agree on the ordering; the [`residual::Residual`]
//! helper exposes `prenorm_step` and `postnorm_step` to make this explicit.
//!
//! # Example
//!
//! ```no_run
//! use candle_core::{Device, DType, Result, Tensor};
//! use layers::{
//!     dtypes::PrecisionPolicy,
//!     linear::LinearInit,
//!     mlp::{FeedForward, FeedForwardConfig},
//!     activations::ActivationKind,
//! };
//!
//! fn build_mlp() -> Result<FeedForward> {
//!     let device = Device::Cpu;
//!     let dtype = DType::F16;
//!     let config = FeedForwardConfig::with_expansion_ratio(4096, 4.0, ActivationKind::Gelu);
//!     FeedForward::with_init(
//!         config,
//!         ActivationKind::Gelu,
//!         &LinearInit::XavierNormal,
//!         &LinearInit::scaled(LinearInit::KaimingUniform { negative_slope: 0.0 }, 0.5),
//!         &device,
//!         dtype,
//!     )
//! }
//!
//! # fn main() -> Result<()> {
//! let mlp = build_mlp()?;
//! let device = Device::Cpu;
//! let hidden = Tensor::randn(0f32, 1.0, (2, 16, 4096), &device)?.to_dtype(DType::F16)?;
//! let policy = PrecisionPolicy::from_parameter_dtype(DType::F16);
//! let output = mlp.forward(&hidden, &policy)?;
//! assert_eq!(output.shape().dims(), &[2, 16, 4096]);
//! Ok(())
//! # }
//! ```

pub mod activations;
pub mod checks;
pub mod dtypes;
pub mod linear;
pub mod mlp;
pub mod norm;
pub mod residual;

pub use crate::dtypes::{PrecisionEpsilons, PrecisionPolicy};
