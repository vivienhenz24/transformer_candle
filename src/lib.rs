pub use cascade_core as core;
pub use cascade_core::*;
pub use cascade_training as training;
pub use transformer_tokenization as tokenization;

use anyhow::Result;
#[cfg(feature = "metal")]
use candle_core::DType;
use candle_core::Device;

pub fn setup_device() -> Result<Device> {
    println!("Starting device detection...");

    if std::env::var("CANDLE_FORCE_CPU").is_ok() {
        println!("CANDLE_FORCE_CPU set, using CPU backend");
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        use std::panic::AssertUnwindSafe;

        println!("Checking Metal backend...");
        let metal_device = std::panic::catch_unwind(AssertUnwindSafe(|| Device::new_metal(0)));
        if let Ok(Ok(device)) = metal_device {
            if metal_preflight(&device).is_ok() {
                println!("Metal device selected: {:?}", device);
                return Ok(device);
            } else {
                println!("Metal device detected but preflight failed, falling back to CPU");
            }
        } else {
            println!("Metal unavailable, falling back...");
        }
    }

    println!("Checking CUDA backend...");
    match Device::cuda_if_available(0) {
        Ok(device) if device.is_cuda() => {
            println!("CUDA device selected: {:?}", device);
            Ok(device)
        }
        Ok(_) | Err(_) => {
            println!("Using CPU backend");
            Ok(Device::Cpu)
        }
    }
}

pub fn check_available_backends() {
    println!(" System Information:");
    println!("   OS: {}", std::env::consts::OS);
    println!("   Architecture: {}", std::env::consts::ARCH);
    println!("   Family: {}", std::env::consts::FAMILY);
    #[cfg(target_os = "macos")]
    println!("   Platform: macOS (Metal available)");
}

#[cfg(feature = "metal")]
fn metal_preflight(device: &Device) -> Result<()> {
    use candle_core::{DType, Tensor};

    let a = Tensor::ones((2, 4), candle_core::DType::F32, device)?;
    let b = Tensor::ones((4, 2), candle_core::DType::F32, device)?;
    let _ = a.matmul(&b)?;

    let x = Tensor::randn(0.0f32, 1.0f32, (4, 16, 32), device)?;
    let q = x.clone();
    let k = x.transpose(1, 2)?;
    let _ = q.matmul(&k)?;

    let host = (0..32u32).collect::<Vec<_>>();
    let indices = Tensor::from_vec(host, (32,), &Device::Cpu)?.to_device(device)?;
    let emb = Tensor::randn(0.0f32, 1.0f32, (32, 64), device)?;
    let _ = emb.index_select(&indices.to_dtype(DType::U32)?, 0)?;

    Ok(())
}
