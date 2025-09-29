use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

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
}


impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            raw_data_dir: PathBuf::from("data/raw"),
            processed_data_dir: PathBuf::from("data/processed"),
            output_filename: "combined_corpus.txt".to_string(),
            add_document_separators: true,
            clean_text: true,
            min_line_length: 10,
            hf_data_dir: Some(PathBuf::from("data/raw_hf")),
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
        let local_lines = process_simple_files(
            &local_files,
            &mut output_file,
            config,
            "Local Data",
            total_datasets == 0
        )?;
        total_lines += local_lines;
        total_datasets += 1;
    }

    // Process HF datasets if available
    if let Some(hf_dir) = &config.hf_data_dir {
        if hf_dir.exists() {
            let hf_files = get_txt_files(hf_dir)?;
            if !hf_files.is_empty() {
                println!("📁 Processing {} HF files...", hf_files.len());
                let hf_lines = process_simple_files(
                    &hf_files,
                    &mut output_file,
                    config,
                    "HF Data",
                    total_datasets == 0
                )?;
                total_lines += hf_lines;
            }
        } else {
            println!("⚠️  HF data directory not found: {}. Skipping HF datasets.", hf_dir.display());
        }
    }

    println!("\n🎉 Mixed dataset preprocessing complete!");
    println!("Total lines processed: {}", total_lines);
    println!("Output written to: {}", output_path.display());

    Ok(())
}

/// Simple file processing without weights
fn process_simple_files(
    files: &[PathBuf],
    output_file: &mut File,
    config: &PreprocessConfig,
    dataset_name: &str,
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
        println!("  ✅ Processed {} lines from {}", lines_processed, file_path.file_name().unwrap_or_default().to_string_lossy());
    }

    println!("📊 {} total: {} lines", dataset_name, total_lines);
    Ok(total_lines)
}

/// Check if downloaded HF datasets are available
pub fn check_hf_datasets_available(hf_dir: &Path) -> usize {
    if !hf_dir.exists() {
        return 0;
    }

    match get_txt_files(hf_dir) {
        Ok(files) => files.len(),
        Err(_) => 0,
    }
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