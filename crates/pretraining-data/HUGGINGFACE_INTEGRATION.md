# Hugging Face Dataset Integration Guide

This guide shows how to integrate large-scale datasets from Hugging Face with your existing tokenizer training pipeline.

## 🚀 Quick Start

### 1. Setup Python Environment

**Windows:**
```bash
cd crates/pretraining-data
scripts\setup.bat
```

**Linux/Mac:**
```bash
cd crates/pretraining-data
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Download Datasets

**Start Small (for testing):**
```bash
# Activate environment first
venv\Scripts\activate.bat  # Windows
# or
source venv/bin/activate   # Linux/Mac

# Download a small sample for testing
python scripts/download_datasets.py --datasets openwebtext --max-examples 50000
```

Note: For private databases, you should set up Hugging Face authentication. This is simple. Make an account, create an access token, and put into .env as HUGGINGFACE_HUB_TOKEN.

**Production Scale:**
```bash
# Download full datasets (this will take time and space!)
python scripts/download_datasets.py --datasets openwebtext wikipedia books
```

### 3. Process Combined Datasets

```bash
# Process only your local HBS data
cargo run --bin preprocess

# Process mixed datasets (local + HF)
cargo run --bin preprocess -- --mixed
```

## 📊 Available Datasets

| Dataset | Size | Description | Recommended Use |
|---------|------|-------------|-----------------|
| `openwebtext` | ~38GB | GPT-2 training data recreation | General text understanding |
| `wikipedia` | ~20GB | Wikipedia articles | Factual knowledge |
| `books` | ~4GB | Long-form narrative text | Language structure |
| `code` | Variable | Programming code in multiple languages | Code understanding |

## 🛠️ Configuration Options

### Dataset Mixing Weights

Edit the weights in your preprocessing config to control dataset proportions:

```rust
// In your code, these are the default weights:
local_data_weight: 0.1,      // 10% - Your HBS articles
openwebtext_weight: 0.5,     // 50% - Web text
wikipedia_weight: 0.2,       // 20% - Wikipedia
books_weight: 0.1,           // 10% - Books
code_weight: 0.1,            // 10% - Code
```

### Memory Management

For large datasets, the system automatically:
- **Chunks data** into manageable pieces (10,000 examples per chunk)
- **Samples by weight** to maintain proportions without processing everything
- **Streams processing** to avoid loading entire datasets into memory

## 📝 Usage Examples

### Example 1: Quick Test with Small Data
```bash
# Download small sample
python scripts/download_datasets.py --datasets wikipedia --max-examples 10000 --chunk-size 1000

# Process mixed data
cargo run --bin preprocess -- --mixed
```

### Example 2: Production Training Data
```bash
# Download full datasets
python scripts/download_datasets.py --datasets openwebtext wikipedia

# Process everything
cargo run --bin preprocess -- --mixed
```

### Example 3: Code-Focused Model
```bash
# Download code + some text
python scripts/download_datasets.py --datasets code openwebtext --max-examples 100000

# Adjust weights in code to prefer code data
cargo run --bin preprocess -- --mixed
```

## 🔧 Tokenizer Training Integration

Once you have processed datasets, use them with your existing tokenizer training:

```rust
use tokenizer::trainer::train_from_corpus;
use tokenizer::config::*;

let config = Config {
    model: ModelCfg {
        vocab_size: 32000,  // Increased for diverse data
        min_frequency: 2,
        dropout: None,
        special_tokens: vec![
            "<|endoftext|>".to_string(),
            "<|pad|>".to_string(),
        ],
        byte_fallback_on_decode: true,
    },
    training: Some(TrainingCfg {
        inputs: vec![PathBuf::from("data/processed/mixed_corpus.txt")],
        shuffle: true,  // Critical for good mixing
        max_lines: None,
        num_threads: Some(8),
    }),
    artifacts: ArtifactsCfg {
        dir: PathBuf::from("data/tokenizer_artifacts"),
        tokenizer_json: Some(PathBuf::from("mixed_tokenizer.json")),
        // ... other config
    },
    // ... rest of config
};

let tokenizer = train_from_corpus(&config)?;
```

## 📂 Directory Structure

After setup, your directory structure will look like:

```
crates/pretraining-data/
├── data/
│   ├── raw/                    # Your original HBS files
│   │   ├── HBS_article1.txt
│   │   └── ...
│   ├── raw_hf/                # Downloaded HF datasets
│   │   ├── openwebtext_chunk_0001.txt
│   │   ├── wikipedia_chunk_0001.txt
│   │   └── download_manifest.json
│   ├── processed/             # Combined processed data
│   │   ├── hbs_corpus.txt     # Local only
│   │   └── mixed_corpus.txt   # Local + HF
│   └── tokenizer_artifacts/   # Trained tokenizers
│       ├── mixed_tokenizer.json
│       └── ...
├── scripts/
│   ├── download_datasets.py   # HF download script
│   ├── requirements.txt       # Python dependencies
│   └── setup.sh/setup.bat     # Setup scripts
└── venv/                      # Python virtual environment
```

## ⚡ Performance Tips

1. **Start Small**: Use `--max-examples` to test your pipeline before downloading TBs of data
2. **Incremental Scaling**: Start with Wikipedia (20GB), then add OpenWebText (38GB), then larger datasets
3. **SSD Storage**: Large datasets benefit significantly from SSD storage
4. **RAM**: Processing mixed datasets uses ~2-4GB RAM efficiently due to streaming
5. **Parallel Processing**: Adjust `num_threads` in your tokenizer config based on your CPU

## 🐛 Troubleshooting

**"No HF datasets found" error:**
- Run the Python download script first
- Check that `data/raw_hf/` directory exists and contains `*_chunk_*.txt` files

**Python import errors:**
- Make sure you activated the virtual environment
- Reinstall requirements: `pip install -r scripts/requirements.txt`

**Out of disk space:**
- Use `--max-examples` to limit dataset size
- Process datasets one at a time instead of downloading all at once

**Slow processing:**
- Increase chunk size with `--chunk-size` flag
- Process datasets on SSD storage if available
- Reduce dataset weights to sample fewer files

## 🎯 Next Steps

1. **Test the Pipeline**: Start with small datasets to validate your setup
2. **Scale Gradually**: Incrementally add larger datasets
3. **Monitor Quality**: Check tokenizer performance as you scale up data
4. **Optimize Mixing**: Adjust dataset weights based on your model's target domain

Your tokenizer training pipeline is now ready to scale from HBS articles to production-quality LLM training data!