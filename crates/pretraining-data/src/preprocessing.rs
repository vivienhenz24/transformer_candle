use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

/// Configuration for data preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub raw_data_dir: PathBuf,
    pub processed_data_dir: PathBuf,
    pub output_filename: String,
    pub add_document_separators: bool,
    pub clean_text: bool,
    pub min_line_length: usize,
    pub hf_data_dir: Option<PathBuf>,
    pub dataset_weights: HashMap<String, f32>,
}

/// Configuration for mixing multiple datasets
#[derive(Debug, Clone)]
pub struct DatasetMixConfig {
    pub local_data_weight: f32,     // Weight for your local HBS data
    pub openwebtext_weight: f32,    // Weight for OpenWebText
    pub wikipedia_weight: f32,      // Weight for Wikipedia
    pub books_weight: f32,          // Weight for books
    pub code_weight: f32,           // Weight for code
}

impl Default for DatasetMixConfig {
    fn default() -> Self {
        Self {
            local_data_weight: 0.1,    // 10%
            openwebtext_weight: 0.5,   // 50%
            wikipedia_weight: 0.2,     // 20%
            books_weight: 0.1,         // 10%
            code_weight: 0.1,          // 10%
        }
    }
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        let mut dataset_weights = HashMap::new();
        dataset_weights.insert("local".to_string(), 0.1);
        dataset_weights.insert("openwebtext".to_string(), 0.5);
        dataset_weights.insert("wikipedia".to_string(), 0.2);
        dataset_weights.insert("books".to_string(), 0.1);
        dataset_weights.insert("code".to_string(), 0.1);

        Self {
            raw_data_dir: PathBuf::from("data/raw"),
            processed_data_dir: PathBuf::from("data/processed"),
            output_filename: "combined_corpus.txt".to_string(),
            add_document_separators: true,
            clean_text: true,
            min_line_length: 10,
            hf_data_dir: Some(PathBuf::from("data/raw_hf")),
            dataset_weights,
        }
    }
}

/// Preprocesses and combines multiple text files into a single corpus file
pub fn preprocess_and_combine(config: &PreprocessConfig) -> io::Result<()> {
    println!("Starting data preprocessing...");

    // Create processed directory if it doesn't exist
    fs::create_dir_all(&config.processed_data_dir)?;

    // Get all .txt files from raw directory
    let txt_files = get_txt_files(&config.raw_data_dir)?;
    println!("Found {} .txt files to process", txt_files.len());

    if txt_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("No .txt files found in {}", config.raw_data_dir.display())
        ));
    }

    // Create output file
    let output_path = config.processed_data_dir.join(&config.output_filename);
    let mut output_file = File::create(&output_path)?;

    let mut total_lines = 0;
    let mut total_files = 0;

    // Process each file
    for (index, file_path) in txt_files.iter().enumerate() {
        println!("Processing file {}/{}: {}",
                 index + 1, txt_files.len(), file_path.display());

        let lines_processed = process_single_file(
            file_path,
            &mut output_file,
            config,
            index == 0  // Don't add separator before first file
        )?;

        total_lines += lines_processed;
        total_files += 1;

        println!("  → Processed {} lines", lines_processed);
    }

    println!("\nPreprocessing complete!");
    println!("Files processed: {}", total_files);
    println!("Total lines: {}", total_lines);
    println!("Output written to: {}", output_path.display());

    Ok(())
}

/// Get all .txt files from a directory
fn get_txt_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut txt_files = Vec::new();

    if !dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Directory not found: {}", dir.display())
        ));
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "txt" {
                    txt_files.push(path);
                }
            }
        }
    }

    // Sort files for consistent ordering
    txt_files.sort();

    Ok(txt_files)
}

/// Process a single file and append to output
fn process_single_file(
    input_path: &Path,
    output_file: &mut File,
    config: &PreprocessConfig,
    is_first_file: bool,
) -> io::Result<usize> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut lines_written = 0;

    // Add document separator (except for first file)
    if config.add_document_separators && !is_first_file {
        writeln!(output_file)?;
        writeln!(output_file, "--- Document: {} ---",
                 input_path.file_name().unwrap_or_default().to_string_lossy())?;
        writeln!(output_file)?;
        lines_written += 3;
    }

    // Process each line
    for line in reader.lines() {
        let line = line?;

        if config.clean_text {
            let cleaned_line = clean_text_line(&line);

            // Skip very short lines or empty lines after cleaning
            if cleaned_line.len() >= config.min_line_length {
                writeln!(output_file, "{}", cleaned_line)?;
                lines_written += 1;
            }
        } else {
            // Just write the line as-is if it's not empty
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                writeln!(output_file, "{}", trimmed)?;
                lines_written += 1;
            }
        }
    }

    Ok(lines_written)
}

/// Clean and normalize a text line
fn clean_text_line(line: &str) -> String {
    let mut cleaned = line.trim().to_string();

    // Remove excessive whitespace
    cleaned = cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    // Remove control characters but keep basic punctuation
    cleaned = cleaned
        .chars()
        .filter(|c| !c.is_control() || *c == '\t')
        .collect();

    // Replace multiple consecutive spaces with single space
    while cleaned.contains("  ") {
        cleaned = cleaned.replace("  ", " ");
    }

    cleaned
}

/// Statistics about the processed corpus
#[derive(Debug)]
pub struct CorpusStats {
    pub total_files: usize,
    pub total_lines: usize,
    pub total_chars: usize,
    pub output_path: PathBuf,
}

/// Get statistics about a processed corpus file
pub fn get_corpus_stats(corpus_path: &Path) -> io::Result<CorpusStats> {
    let file = File::open(corpus_path)?;
    let reader = BufReader::new(file);

    let mut total_lines = 0;
    let mut total_chars = 0;
    let mut total_files = 0;

    for line in reader.lines() {
        let line = line?;
        total_lines += 1;
        total_chars += line.len();

        // Count document separators to estimate file count
        if line.starts_with("--- Document:") {
            total_files += 1;
        }
    }

    // If no separators found, assume it was a single file or combined without separators
    if total_files == 0 {
        total_files = 1;
    }

    Ok(CorpusStats {
        total_files,
        total_lines,
        total_chars,
        output_path: corpus_path.to_path_buf(),
    })
}

/// Process and combine datasets from both local and Hugging Face sources
pub fn preprocess_mixed_datasets(config: &PreprocessConfig) -> io::Result<()> {
    println!("🚀 Starting mixed dataset preprocessing...");

    // Create processed directory if it doesn't exist
    fs::create_dir_all(&config.processed_data_dir)?;

    let output_path = config.processed_data_dir.join(&config.output_filename);
    let mut output_file = File::create(&output_path)?;

    let mut total_lines = 0;
    let mut total_datasets = 0;

    // Process local data first
    let local_files = get_txt_files(&config.raw_data_dir)?;
    if !local_files.is_empty() {
        println!("📁 Processing {} local files...", local_files.len());
        let local_weight = config.dataset_weights.get("local").unwrap_or(&0.1);
        let local_lines = process_dataset_files(
            &local_files,
            &mut output_file,
            config,
            "Local Data",
            *local_weight,
            total_datasets == 0
        )?;
        total_lines += local_lines;
        total_datasets += 1;
    }

    // Process HF datasets if available
    if let Some(hf_dir) = &config.hf_data_dir {
        if hf_dir.exists() {
            total_lines += process_hf_datasets(hf_dir, &mut output_file, config, total_datasets == 0)?;
        } else {
            println!("⚠️  HF data directory not found: {}. Skipping HF datasets.", hf_dir.display());
        }
    }

    println!("\n🎉 Mixed dataset preprocessing complete!");
    println!("Total lines processed: {}", total_lines);
    println!("Output written to: {}", output_path.display());

    Ok(())
}

/// Process Hugging Face datasets from downloaded chunks
fn process_hf_datasets(
    hf_dir: &Path,
    output_file: &mut File,
    config: &PreprocessConfig,
    is_first: bool,
) -> io::Result<usize> {
    let mut total_lines = 0;
    let mut dataset_count = 0;

    // Define dataset patterns and their weights
    let dataset_patterns = vec![
        ("openwebtext", config.dataset_weights.get("openwebtext").unwrap_or(&0.5)),
        ("wikipedia", config.dataset_weights.get("wikipedia").unwrap_or(&0.2)),
        ("books", config.dataset_weights.get("books").unwrap_or(&0.1)),
        ("code", config.dataset_weights.get("code").unwrap_or(&0.1)),
    ];

    for (dataset_name, weight) in dataset_patterns {
        let pattern = format!("{}_chunk_", dataset_name);
        let dataset_files = get_files_matching_pattern(hf_dir, &pattern)?;

        if !dataset_files.is_empty() {
            println!("📊 Processing {} {} chunks (weight: {:.1}%)...",
                    dataset_files.len(), dataset_name, weight * 100.0);

            // Sample files based on weight
            let sampled_files = sample_files_by_weight(&dataset_files, *weight);

            let lines = process_dataset_files(
                &sampled_files,
                output_file,
                config,
                &format!("HF-{}", dataset_name),
                *weight,
                is_first && dataset_count == 0
            )?;

            total_lines += lines;
            dataset_count += 1;

            println!("  ✅ Processed {} lines from {}", lines, dataset_name);
        }
    }

    Ok(total_lines)
}

/// Get files matching a specific pattern in a directory
fn get_files_matching_pattern(dir: &Path, pattern: &str) -> io::Result<Vec<PathBuf>> {
    let mut matching_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if let Some(filename_str) = filename.to_str() {
                    if filename_str.contains(pattern) && filename_str.ends_with(".txt") {
                        matching_files.push(path);
                    }
                }
            }
        }
    }

    matching_files.sort();
    Ok(matching_files)
}

/// Sample files based on dataset weight to control mix proportions
fn sample_files_by_weight(files: &[PathBuf], weight: f32) -> Vec<PathBuf> {
    if weight >= 1.0 {
        return files.to_vec();
    }

    let sample_count = ((files.len() as f32) * weight).ceil() as usize;
    let sample_count = sample_count.max(1).min(files.len());

    // Take evenly distributed samples
    let step = files.len() / sample_count;
    let step = step.max(1);

    files.iter()
        .step_by(step)
        .take(sample_count)
        .cloned()
        .collect()
}

/// Process a group of dataset files with consistent formatting
fn process_dataset_files(
    files: &[PathBuf],
    output_file: &mut File,
    config: &PreprocessConfig,
    dataset_name: &str,
    _weight: f32,
    is_first_dataset: bool,
) -> io::Result<usize> {
    let mut total_lines = 0;

    for (index, file_path) in files.iter().enumerate() {
        let lines_processed = process_single_file(
            file_path,
            output_file,
            config,
            is_first_dataset && index == 0
        )?;

        total_lines += lines_processed;

        // Add dataset boundary marker
        if config.add_document_separators && (index + 1) % 10 == 0 {
            writeln!(output_file)?;
            writeln!(output_file, "=== {} Batch {} ===", dataset_name, (index + 1) / 10)?;
            writeln!(output_file)?;
            total_lines += 3;
        }
    }

    Ok(total_lines)
}

/// Check if downloaded HF datasets are available
pub fn check_hf_datasets_available(hf_dir: &Path) -> HashMap<String, usize> {
    let mut available = HashMap::new();

    if !hf_dir.exists() {
        return available;
    }

    let datasets = ["openwebtext", "wikipedia", "books", "code"];

    for dataset in datasets {
        let pattern = format!("{}_chunk_", dataset);
        if let Ok(files) = get_files_matching_pattern(hf_dir, &pattern) {
            if !files.is_empty() {
                available.insert(dataset.to_string(), files.len());
            }
        }
    }

    available
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_clean_text_line() {
        assert_eq!(clean_text_line("  Hello    world  "), "Hello world");
        assert_eq!(clean_text_line("Text\twith\ttabs"), "Text with tabs");
        assert_eq!(clean_text_line(""), "");
    }

    #[test]
    fn test_get_txt_files() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create test files
        fs::write(dir_path.join("test1.txt"), "content1").unwrap();
        fs::write(dir_path.join("test2.txt"), "content2").unwrap();
        fs::write(dir_path.join("other.doc"), "other").unwrap();

        let txt_files = get_txt_files(dir_path).unwrap();
        assert_eq!(txt_files.len(), 2);
        assert!(txt_files.iter().all(|p| p.extension().unwrap() == "txt"));
    }
}