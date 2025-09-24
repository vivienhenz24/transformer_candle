pub use cascade_core as core;
pub use cascade_core::*;
pub use cascade_training as training;
pub use transformer_tokenization as tokenization;

use anyhow::Result;
use candle_core::Device;

pub fn setup_device() -> Result<Device> {
    println!("Starting device detection...");
    #[cfg(feature = "metal")]
    {
        println!("Checking Metal backend...");
        if let Ok(device) = Device::new_metal(0) {
            println!("Metal device selected: {:?}", device);
            return Ok(device);
        }
        println!("Metal unavailable, falling back...");
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
