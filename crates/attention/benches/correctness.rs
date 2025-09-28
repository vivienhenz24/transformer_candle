//! Correctness harness comparing fused and reference backends.
//! Run with: `cargo bench -p attention --features fused correctness`

#[path = "common/mod.rs"]
mod util;

use std::error::Error;

use attention::core::{BackendSelection, Config};
use util::{format_markdown_table, update_results};

#[cfg(feature = "fused")]
use attention::fused::FusedAttention;
#[cfg(feature = "fused")]
use attention::masks::build_causal_mask;
#[cfg(feature = "fused")]
use attention::reference::ExactAttention;
#[cfg(feature = "fused")]
use attention::Attention;
#[cfg(feature = "fused")]
use candle_core::{DType, Device, Tensor};

fn main() {
    if let Err(err) = run() {
        eprintln!("correctness harness failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "fused"))]
    {
        println!("fused feature disabled; skipping correctness harness");
        update_results("Correctness", "Fused backend not enabled; harness skipped.")?;
        return Ok(());
    }

    #[cfg(feature = "fused")]
    {
        let device = Device::Cpu;
        let seq_lengths = [32usize, 128, 512, 2048, 8192];
        let head_dims = [64usize, 80, 128];
        let dtypes = [DType::F32, DType::BF16, DType::F16];

        let mut rows = Vec::new();
        let fused = FusedAttention::new();
        let reference = ExactAttention::new();

        for &seq_len in &seq_lengths {
            for &head_dim in &head_dims {
                for &dtype in &dtypes {
                    let case = Case {
                        batch: 1,
                        heads: 4,
                        seq_len,
                        head_dim,
                        dtype,
                    };
                    let (q, k, v, mask) = build_inputs(&device, &case)?;

                    let config_ref = Config {
                        backend: BackendSelection::ReferenceOnly,
                        ..Config::default()
                    };
                    let config_fused = Config {
                        backend: BackendSelection::FusedOnly,
                        ..Config::default()
                    };

                    let reference_out = reference.attend(&q, &k, &v, Some(&mask), &config_ref)?;
                    let fused_out = fused.attend(&q, &k, &v, Some(&mask), &config_fused)?;

                    let reference_f32 = reference_out.to_dtype(DType::F32)?;
                    let fused_f32 = fused_out.to_dtype(DType::F32)?;

                    let abs_diff = fused_f32
                        .sub(&reference_f32)
                        .map_err(to_backend_err)?
                        .abs()
                        .map_err(to_backend_err)?
                        .flatten_all()
                        .map_err(to_backend_err)?
                        .to_vec1::<f32>()?;
                    let reference_vals = reference_f32
                        .flatten_all()
                        .map_err(to_backend_err)?
                        .to_vec1::<f32>()?;

                    let mut max_abs = 0.0f32;
                    let mut max_rel = 0.0f32;
                    for (abs, ref_val) in abs_diff.iter().zip(reference_vals.iter()) {
                        max_abs = max_abs.max(*abs);
                        let denom = ref_val.abs().max(1e-5);
                        max_rel = max_rel.max(*abs / denom);
                    }

                    let fused_scores = fused_out
                        .to_dtype(DType::F32)?
                        .flatten_all()
                        .map_err(to_backend_err)?
                        .to_vec1::<f32>()?;
                    let reference_scores = reference_out
                        .to_dtype(DType::F32)?
                        .flatten_all()
                        .map_err(to_backend_err)?
                        .to_vec1::<f32>()?;
                    let log_diff = fused_scores
                        .iter()
                        .zip(reference_scores.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);

                    const ABS_TOL: f32 = 5e-3;
                    const REL_TOL: f32 = 5e-2;
                    const LOG_TOL: f32 = 5e-3;

                    if max_abs > ABS_TOL || max_rel > REL_TOL || log_diff > LOG_TOL {
                        return Err(format!(
                            "tolerance breach for seq_len={} head_dim={} dtype={:?}: abs={:.3e} rel={:.3e} log={:.3e}",
                            seq_len, head_dim, dtype, max_abs, max_rel, log_diff
                        )
                        .into());
                    }

                    rows.push(vec![
                        format!("({},{},{},{})", case.batch, case.heads, seq_len, head_dim),
                        format!("{:?}", dtype),
                        format!("{:.2e}", max_abs),
                        format!("{:.2e}", max_rel),
                        format!("{:.2e}", log_diff),
                    ]);
                }
            }
        }

        let table = format_markdown_table(
            &["shape", "dtype", "max abs", "max rel", "max |Δlog|"],
            &rows,
        );

        println!("\nCorrectness summary:\n{table}");
        update_results("Correctness", &table)?;
    }

    Ok(())
}

#[cfg(feature = "fused")]
#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    dtype: DType,
}

#[cfg(feature = "fused")]
fn build_inputs(
    device: &Device,
    case: &Case,
) -> Result<(Tensor, Tensor, Tensor, Tensor), Box<dyn Error>> {
    let shape = (case.batch, case.heads, case.seq_len, case.head_dim);
    let q = Tensor::rand(0.0f32, 1.0, shape, device)?.to_dtype(case.dtype)?;
    let k = Tensor::rand(0.0f32, 1.0, shape, device)?.to_dtype(case.dtype)?;
    let v = Tensor::rand(0.0f32, 1.0, shape, device)?.to_dtype(case.dtype)?;
    let mask = build_causal_mask(device, case.batch, case.heads, case.seq_len, case.seq_len)?;
    Ok((q, k, v, mask))
}

#[cfg(feature = "fused")]
fn to_backend_err(err: candle_core::Error) -> attention::core::AttentionError {
    attention::core::AttentionError::Backend {
        message: err.to_string(),
    }
}
