//! Memory benchmark for the attention backends.
//! Run with: `cargo bench -p attention [--features fused] memory`

#[path = "common/mod.rs"]
mod util;

use std::error::Error;
use std::time::Instant;

use attention::core::{BackendSelection, Config};
use attention::masks::build_causal_mask;
use attention::reference::ExactAttention;
use attention::Attention;
use candle_core::{Device, DType, Tensor};
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
        eprintln!("memory bench failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;
    let cases = [
        Case {
            batch: 1,
            heads: 4,
            seq_len: 2048,
            head_dim: 64,
            dtype: DType::F32,
        },
        Case {
            batch: 2,
            heads: 4,
            seq_len: 4096,
            head_dim: 128,
            dtype: DType::BF16,
        },
    ];

    let mut rows = Vec::new();

    for case in cases {
        let (q, k, v, mask) = build_inputs(&device, &case)?;
        let tokens_per_iter = (case.batch * case.heads * case.seq_len) as f64;

        let mut reference = ExactAttention::new();
        let config_ref = Config {
            backend: BackendSelection::ReferenceOnly,
            ..Config::default()
        };
        let metrics = memory_profile(
            "reference",
            &mut reference,
            &config_ref,
            &q,
            &k,
            &v,
            &mask,
            tokens_per_iter,
        )?;
        rows.push(metrics);

        #[cfg(feature = "fused")]
        {
            let mut fused = FusedAttention::new();
            let config_fused = Config {
                backend: BackendSelection::FusedOnly,
                ..Config::default()
            };
            let metrics = memory_profile(
                "fused",
                &mut fused,
                &config_fused,
                &q,
                &k,
                &v,
                &mask,
                tokens_per_iter,
            )?;
            rows.push(metrics);
        }
    }

    let table = format_markdown_table(
        &["backend", "shape", "dtype", "tokens/sec", "peak MB", "steady MB"],
        &rows,
    );

    println!("\nMemory summary:\n{table}");
    update_results("Memory", &table)?;

    Ok(())
}

fn memory_profile<A: Attention>(
    backend: &str,
    attention: &mut A,
    config: &Config,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    tokens_per_iter: f64,
) -> Result<Vec<String>, Box<dyn Error>> {
    let baseline = current_mem_mb();

    // Warm-up
    for _ in 0..3 {
        let _ = attention.attend(q, k, v, Some(mask), config)?;
    }
    let after_warm = current_mem_mb();

    let mut peak = after_warm;
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = attention.attend(q, k, v, Some(mask), config)?;
        peak = peak.max(current_mem_mb());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tokens_sec = iterations as f64 * tokens_per_iter / elapsed;

    let steady = current_mem_mb();

    Ok(vec![
        backend.to_string(),
        describe_case(q),
        format!("{:?}", q.dtype()),
        format_tokens_per_sec(tokens_sec),
        format!("{:.2}", (peak - baseline).max(0.0)),
        format!("{:.2}", (steady - baseline).max(0.0)),
    ])
}

fn current_mem_mb() -> f64 {
    use std::process::Command;

    let pid = std::process::id().to_string();
    if let Ok(output) = Command::new("ps").args(["-o", "rss=", "-p", &pid]).output() {
        if output.status.success() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                if let Ok(kb) = text.trim().parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

fn format_tokens_per_sec(tokens_sec: f64) -> String {
    if tokens_sec >= 1e6 {
        format!("{:.2} M", tokens_sec / 1e6)
    } else if tokens_sec >= 1e3 {
        format!("{:.2} K", tokens_sec / 1e3)
    } else {
        format!("{:.2}", tokens_sec)
    }
}

fn describe_case(q: &Tensor) -> String {
    let dims = q.dims();
    format!("({},{},{},{})", dims[0], dims[1], dims[2], dims[3])
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
