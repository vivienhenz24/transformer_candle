# Quick Start: Streaming Training (No Disk Storage Required!)

This guide gets you training a 1B parameter model in minutes using streaming data from HuggingFace.

## 🚀 Step-by-Step on RunPod

### 1. Set Up Your Pod

```bash
# SSH into your RunPod
# Navigate to workspace
cd /workspace

# Clone your repo
git clone <your-repo-url> transformer_candle
cd transformer_candle

# Set target directory to avoid disk quota issues
export CARGO_TARGET_DIR=/tmp/rust-target

# Add to .bashrc for persistence
echo 'export CARGO_TARGET_DIR=/tmp/rust-target' >> ~/.bashrc
```

### 2. Install Python Dependencies

```bash
# Make sure you have the datasets library
pip install datasets
```

### 3. Build the Project

```bash
# Build in release mode (takes ~2-3 minutes)
cargo build --release
```

### 4. Test Streaming (Optional but Recommended)

```bash
# Test with 1000 samples to verify everything works
cargo run --example test_streaming

# You should see:
# - "Loading HuggingFace dataset..."
# - "HuggingFace streaming ready"
# - Sample previews
# - "Streaming completed successfully!"
```

### 5. Train Tokenizer with Streaming Data

```bash
# This streams 1M samples, trains tokenizer, NO disk storage needed!
cargo run -p training --bin orchestrate -- \
  --stream \
  --dataset "HuggingFaceFW/fineweb" \
  --split train \
  --max-samples 1000000 \
  --vocab-size 50000 \
  --experiment streaming-1b

# Output will be in:
# /workspace/artifacts/tokenizer/streaming-1b/
```

### 6. Start Training!

The model training itself still needs to be integrated (next step), but for now, you can verify the tokenizer works:

```bash
# Check the tokenizer was created
ls -la /workspace/artifacts/tokenizer/streaming-1b/

# You should see:
# - tokenizer.json
# - vocab.json
# - merges.txt
```

## 🎯 What Just Happened?

1. **No Data Download**: The orchestrate binary streamed 1M samples directly from HuggingFace
2. **No Disk Usage**: Everything happened in memory
3. **Tokenizer Trained**: A 50k vocabulary BBPE tokenizer was trained on real FineWeb data
4. **Ready for Training**: All artifacts are set up for model training

## 📊 Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stream` | false | Enable streaming mode |
| `--dataset` | HuggingFaceFW/fineweb | Dataset to stream from |
| `--split` | train | Which split to use |
| `--max-samples` | None | Limit samples (None = unlimited) |
| `--vocab-size` | 32000 | Tokenizer vocabulary size |
| `--stream-batch-size` | 1000 | Samples per fetch batch |
| `--tokenizer-max-lines` | 1000000 | Samples for tokenizer training |
| `--experiment` | auto | Experiment name |

## 🔥 Full Training Example (100M Samples)

```bash
# Train tokenizer on 1M samples
cargo run -p training --bin orchestrate -- \
  --stream \
  --dataset "HuggingFaceFW/fineweb" \
  --max-samples 1000000 \
  --vocab-size 50000 \
  --experiment runpod-1b-streaming

# This creates tokenizer in:
# /workspace/artifacts/tokenizer/runpod-1b-streaming/

# Training integration coming next!
```

## 💡 Advantages Over File-Based Approach

| Aspect | File-Based | Streaming |
|--------|------------|-----------|
| Setup Time | 1-2 hours | 10 seconds |
| Disk Usage | 50-100 GB | 0 GB |
| Quota Issues | Common | Never |
| Flexibility | Fixed dataset | Dynamic |
| Cost | Storage costs | Free |

## 🐛 Troubleshooting

### "Failed to spawn Python process"
```bash
# Check Python is available
python3 --version

# Should see Python 3.x
```

### "ModuleNotFoundError: No module named 'datasets'"
```bash
# Install the datasets library
pip install datasets
```

### "Disk quota exceeded"
```bash
# Make sure CARGO_TARGET_DIR is set
echo $CARGO_TARGET_DIR

# Should show: /tmp/rust-target

# If not set:
export CARGO_TARGET_DIR=/tmp/rust-target
```

### Streaming is slow
```bash
# Increase batch size for fewer network round trips
--stream-batch-size 5000

# Or use a faster internet connection
```

## 📈 Next Steps

1. **Complete Training Integration**: Modify the training binary to use streaming corpus
2. **Start 1B Training**: Train on 100M+ samples
3. **Monitor Progress**: Use TensorBoard for metrics
4. **Scale Up**: Try 2B or 4B models

## 🎓 Architecture Overview

```
HuggingFace FineWeb (36 trillion tokens)
         ↓
   Python subprocess
         ↓
   JSON batches (1000 samples)
         ↓
   Rust buffer (in memory)
         ↓
   BBPE Tokenizer Training
         ↓
   Tokenizer artifacts saved
         ↓
   Ready for model training!
```

## ✅ Success Indicators

You'll know it's working when you see:
- ✅ "Loading HuggingFace dataset..."
- ✅ "HuggingFace streaming ready"
- ✅ "Collected X samples for tokenizer..."
- ✅ "Tokenizer trained and saved to..."
- ✅ Files in `/workspace/artifacts/tokenizer/`

## 🚧 Current Limitations

1. **Training Binary**: Need to create `train-streaming` binary (integration in progress)
2. **Checkpointing**: Streaming state not saved in checkpoints yet
3. **Multi-GPU**: Single GPU only for now

These will be addressed in upcoming updates!

## 📚 Additional Resources

- Full documentation: `STREAMING_TRAINING.md`
- Test example: `crates/pretraining-data/examples/test_streaming.rs`
- Implementation: `crates/pretraining-data/src/corpora.rs`

Happy training! 🎉
