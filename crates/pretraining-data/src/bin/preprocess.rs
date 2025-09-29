use pretraining_data::preprocessing::{
    preprocess_and_combine, preprocess_mixed_datasets, PreprocessConfig,
    get_corpus_stats, check_hf_datasets_available
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let use_mixed_datasets = args.contains(&"--mixed".to_string()) || args.contains(&"--hf".to_string());

    // Default configuration
    let mut config = PreprocessConfig::default();

    // Override paths based on current working directory structure
    // Assuming we're running from the crate root
    config.raw_data_dir = PathBuf::from("data/raw");
    config.processed_data_dir = PathBuf::from("data/processed");
    config.hf_data_dir = Some(PathBuf::from("data/raw_hf"));

    if use_mixed_datasets {
        config.output_filename = "mixed_corpus.txt".to_string();
    } else {
        config.output_filename = "hbs_corpus.txt".to_string();
    }

    // Check if custom output filename was provided
    if args.len() > 1 && !args[1].starts_with("--") {
        config.output_filename = args[1].clone();
    }

    println!("Preprocessing Configuration:");
    println!("  Mode: {}", if use_mixed_datasets { "Mixed Datasets (HF + Local)" } else { "Local Only" });
    println!("  Raw data directory: {}", config.raw_data_dir.display());
    println!("  Processed data directory: {}", config.processed_data_dir.display());
    if let Some(hf_dir) = &config.hf_data_dir {
        println!("  HF data directory: {}", hf_dir.display());
    }
    println!("  Output filename: {}", config.output_filename);
    println!("  Document separators: {}", config.add_document_separators);
    println!("  Text cleaning: {}", config.clean_text);
    println!("  Min line length: {}", config.min_line_length);

    // Check available datasets
    if use_mixed_datasets {
        if let Some(hf_dir) = &config.hf_data_dir {
            let file_count = check_hf_datasets_available(hf_dir);
            if file_count > 0 {
                println!("\n📊 Available HF files: {}", file_count);
            } else {
                println!("\n⚠️  No HF datasets found. Run Python download script first!");
            }
        }
    }
    println!();

    // Run preprocessing
    let result = if use_mixed_datasets {
        preprocess_mixed_datasets(&config)
    } else {
        preprocess_and_combine(&config)
    };

    match result {
        Ok(()) => {
            println!("✅ Preprocessing completed successfully!");

            // Show statistics
            let output_path = config.processed_data_dir.join(&config.output_filename);
            match get_corpus_stats(&output_path) {
                Ok(stats) => {
                    println!("\n📊 Corpus Statistics:");
                    println!("  Files processed: {}", stats.total_files);
                    println!("  Total lines: {}", stats.total_lines);
                    println!("  Total characters: {}", stats.total_chars);
                    println!("  Output file: {}", stats.output_path.display());

                    // Estimate file size
                    if let Ok(metadata) = std::fs::metadata(&output_path) {
                        let size_mb = metadata.len() as f64 / 1_048_576.0;
                        println!("  File size: {:.2} MB", size_mb);
                    }
                }
                Err(e) => {
                    eprintln!("⚠️  Could not get corpus statistics: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Preprocessing failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}