pub mod gpt;
pub mod tokenizer;
pub mod training;

pub use gpt::{Block, FeedForward, GPTConfig, GPTLanguageModel, Head, MultiHeadAttention};
pub use tokenizer::{CharTokenizer, DataSplit};
pub use training::{
    create_medium_gpt_config,
    create_small_gpt_config,
    estimate_loss,
    train_model,
    TrainingConfig,
    TrainingStats,
};

use anyhow::Result;
use candle_core::Device;

/// Setup compute device (prefer Metal on Apple, then CUDA, otherwise CPU)
pub fn setup_device() -> Result<Device> {
    println!("Starting device detection...");

    // Prefer the Metal backend when compiled with the Metal feature.
    #[cfg(feature = "metal")]
    {
        println!("Checking for Metal GPU support...");
        match Device::new_metal(0) {
            Ok(device) => {
                println!("Metal GPU detected successfully!");
                println!("   Device type: {:?}", device);
                println!("   Device info: {:?}", device);
                println!("Metal GPU will be used for training");
                return Ok(device);
            }
            Err(err) => {
                println!("Metal GPU detection failed: {err}");
                println!("Falling back to other backends...\n");
            }
        }
    }

    // When the Metal feature isn’t enabled we silently fall through to other backends.
    #[cfg(not(feature = "metal"))]
    {
        println!("Metal backend not enabled at compile time, skipping detection.\n");
    }

    // Check for CUDA support next.
    println!("🔍 Checking for CUDA GPU support...");
    match Device::cuda_if_available(0) {
        Ok(device) => {
            if device.is_cuda() {
                println!("CUDA GPU detected and will be used for training");
                return Ok(device);
            }

            println!("CUDA device handle returned but not reporting as CUDA, skipping.");
        }
        Err(err) => {
            println!("CUDA GPU detection failed: {err}");
        }
    }

    println!("💻 Using CPU for training (no GPU backend available)");
    Ok(Device::Cpu)
}

/// Check what backends are available and provide system information
pub fn check_available_backends() {
    println!(" System Information:");
    println!("   OS: {}", std::env::consts::OS);
    println!("   Architecture: {}", std::env::consts::ARCH);
    println!("   Family: {}", std::env::consts::FAMILY);

    // Check if we're on macOS (where Metal should be available)
    #[cfg(target_os = "macos")]
    {
        println!("   Platform: macOS (Metal should be available)");

        // Try to get more system info
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(["SPHardwareDataType"])
            .output()
        {
            if let Ok(hardware_info) = String::from_utf8(output.stdout) {
                for line in hardware_info.lines() {
                    if line.contains("Chip") || line.contains("Processor") || line.contains("Model")
                    {
                        println!("   Hardware: {}", line.trim());
                    }
                }
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("   Platform: Non-macOS (Metal not available)");
    }

    // Check candle-core features
    println!("\n Candle-core backend availability:");
    println!("   CPU: Always available");
    println!("   Metal: Checking at runtime (feature detection not available)");
    println!("   CUDA: Checking at runtime (feature detection not available)");
}
