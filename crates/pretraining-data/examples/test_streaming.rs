use pretraining_data::HuggingFaceStreamingCorpus;
use std::io;

fn main() -> io::Result<()> {
    println!("Testing HuggingFace Streaming Corpus");
    println!("=====================================\n");

    // Create streaming corpus for FineWeb with a small sample limit
    let corpus = HuggingFaceStreamingCorpus::new(
        "HuggingFaceFW/fineweb".to_string(),
        "train".to_string(),
        100,  // Fetch 100 samples per batch
        Some(1000),  // Limit to 1000 samples total for testing
    )?;

    println!("Created corpus, starting stream...\n");

    // Create iterator
    let mut stream = corpus.stream()?;

    let mut count = 0;
    let mut total_chars = 0;

    // Process samples
    for result in stream {
        match result {
            Ok(text) => {
                count += 1;
                total_chars += text.len();

                if count <= 3 {
                    println!("Sample {}: {} chars", count, text.len());
                    println!("Preview: {}\n", &text.chars().take(200).collect::<String>());
                } else if count % 100 == 0 {
                    println!("Processed {} samples...", count);
                }
            }
            Err(e) => {
                eprintln!("Error reading sample: {}", e);
                return Err(e);
            }
        }
    }

    println!("\n=====================================");
    println!("Streaming completed successfully!");
    println!("Total samples: {}", count);
    println!("Total characters: {}", total_chars);
    println!("Average chars per sample: {}", total_chars / count.max(1));

    Ok(())
}
