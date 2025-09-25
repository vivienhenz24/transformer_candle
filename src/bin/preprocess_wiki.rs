use anyhow::Result;
use std::env;
use std::path::PathBuf;
use utils::wiki::preprocess_wikipedia_dump;

fn main() -> Result<()> {
    let input = env::args()
        .nth(1)
        .unwrap_or_else(|| "pt-data/enwiki-latest-pages-articles-multistream.xml.bz2".to_string());
    let output = env::args().nth(2).unwrap_or_else(|| {
        let path = PathBuf::from(&input);
        match path.file_stem().and_then(|stem| stem.to_str()) {
            Some(stem) => format!("pt-data/{}-clean.txt", stem),
            None => "pt-data/wiki-clean.txt".to_string(),
        }
    });

    preprocess_wikipedia_dump(&input, &output)?;
    println!("Cleaned corpus written to {}", output);
    Ok(())
}
