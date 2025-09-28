//! Throughput benchmark for the attention backends.
//! Run with: `cargo bench -p attention [--features fused] throughput`

#[path = "common/mod.rs"]
mod util;

use std::error::Error;
use std::time::Instant;

use attention::core::{BackendSelection, Config};
use attention::masks::build_causal_mask;
use attention::reference::ExactAttention;
use attention::Attention;
use candle_core::{DType, Device, Tensor};
use util::{format_markdown_table, update_results};

#[cfg(feature = "fused")]
use attention::fused::FusedAttention;

#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    dtype: DType,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("throughput bench failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;
    let batches = [1usize, 2];
    let heads = [4usize];
    let seq_lens = [128usize, 512, 2048];
    let head_dims = [64usize, 128];
    let dtypes = [DType::F32, DType::BF16];

    let mut cases = Vec::new();
    for &batch in &batches {
        for &head in &heads {
            for &seq_len in &seq_lens {
                for &head_dim in &head_dims {
                    for &dtype in &dtypes {
                        cases.push(Case {
                            batch,
                            heads: head,
                            seq_len,
                            head_dim,
                            dtype,
                        });
                    }
                }
            }
        }
    }

    let mut rows = Vec::new();

    for case in cases {
        let tokens_per_iter = (case.batch * case.heads * case.seq_len) as f64;
        let iterations = match case.seq_len {
            0..=128 => 200,
            129..=512 => 80,
            513..=2048 => 20,
            _ => 10,
        };

        let (q, k, v, mask) = build_inputs(&device, &case)?;
        let mut reference = ExactAttention::new();
        let config_ref = Config {
            backend: BackendSelection::ReferenceOnly,
            ..Config::default()
        };

        let tokens_sec = measure(&mut reference, &config_ref, &q, &k, &v, &mask, iterations)?;
        rows.push(vec![
            "reference".to_string(),
            describe_case(&case),
            format!("{:?}", case.dtype),
            format_tokens_per_sec(tokens_sec, tokens_per_iter),
        ]);

        #[cfg(feature = "fused")]
        {
            let mut fused = FusedAttention::new();
            let config_fused = Config {
                backend: BackendSelection::FusedOnly,
                ..Config::default()
            };
            let tokens_sec = measure(&mut fused, &config_fused, &q, &k, &v, &mask, iterations)?;
            rows.push(vec![
                "fused".to_string(),
                describe_case(&case),
                format!("{:?}", case.dtype),
                format_tokens_per_sec(tokens_sec, tokens_per_iter),
            ]);
        }
    }

    let table = format_markdown_table(
        &["backend", "shape (b,h,s,d)", "dtype", "tokens/sec"],
        &rows,
    );

    println!("\nThroughput summary:\n{table}");
    update_results("Throughput", &table)?;

    Ok(())
}

fn measure<A: Attention>(
    backend: &mut A,
    config: &Config,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    iterations: usize,
) -> Result<f64, Box<dyn Error>> {
    // Warm-up
    for _ in 0..3 {
        let _ = backend.attend(q, k, v, Some(mask), config)?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.attend(q, k, v, Some(mask), config)?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    Ok(iterations as f64 / elapsed)
}

fn format_tokens_per_sec(iters_per_sec: f64, tokens_per_iter: f64) -> String {
    let tokens_sec = iters_per_sec * tokens_per_iter;
    if tokens_sec >= 1e6 {
        format!("{:.2} M", tokens_sec / 1e6)
    } else if tokens_sec >= 1e3 {
        format!("{:.2} K", tokens_sec / 1e3)
    } else {
        format!("{:.2}", tokens_sec)
    }
}

fn describe_case(case: &Case) -> String {
    format!(
        "({batch},{heads},{seq},{dim})",
        batch = case.batch,
        heads = case.heads,
        seq = case.seq_len,
        dim = case.head_dim
    )
}

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
