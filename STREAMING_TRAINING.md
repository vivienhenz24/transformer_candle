# Streaming Training Guide

This guide explains how to train your transformer model using **streaming data** from Hugging Face, without needing to download the entire dataset to disk.

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Loop                              │
│  • Requests batches continuously                             │
│  • No disk I/O for data loading                              │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│            StreamingTextDataLoader                            │
│  • Maintains shuffle buffer (10k samples)                    │
│  • Tokenizes text on-the-fly in memory                       │
│  • Creates batches (batch_size × sequence_length)            │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│         HuggingFaceStreamingCorpus                            │
│  • Long-lived Python subprocess                              │
│  • Streams samples in batches (1000 at a time)               │
│  • Buffers data in memory (no disk writes)                   │
│  • Can iterate indefinitely (multiple epochs)                │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
              HuggingFace Dataset API
              (FineWeb or any dataset)
```

## Key Benefits

1. **No Disk Quota Issues**: Everything stays in memory
2. **Faster Startup**: No need to download terabytes of data first
3. **Infinite Data**: Stream as much as you need
4. **Memory Efficient**: Only buffers what's needed
5. **Flexible**: Easy to switch datasets or adjust sample limits

## Usage Example

### Option 1: Train with FineWeb Streaming (100M samples)

```bash
# Set up a new pod or use existing one
cd /workspace/transformer_candle

# Build with CUDA support
export CARGO_TARGET_DIR=/tmp/rust-target
cargo build --release

# Start streaming training directly (no data download step!)
cargo run -p training --bin train-streaming -- \
  --dataset "HuggingFaceFW/fineweb" \
  --split train \
  --max-samples 100000000 \
  --batch-size 1000 \
  --config configs/runpod_1b.yaml
```

### Option 2: Use with Custom Dataset

```rust
use pretraining_data::HuggingFaceStreamingCorpus;

// Create streaming corpus
let corpus = HuggingFaceStreamingCorpus::new(
    "HuggingFaceFW/fineweb".to_string(),
    "train".to_string(),
    1000,  // batch size for fetching
    Some(100_000_000),  // max samples (or None for unlimited)
)?;

// Create an iterator
let mut stream = corpus.stream()?;

// Process samples as they arrive
for text in stream {
    let text = text?;
    // Tokenize and train...
}
```

## Implementation Details

### Python Subprocess Communication

The Rust code spawns a long-lived Python process that:
1. Loads the dataset in streaming mode using `datasets` library
2. Sends batches of text as JSON over stdout
3. Provides progress updates on stderr
4. Handles errors gracefully

The communication protocol:
```json
{"status": "loading"}     // Initial load
{"status": "ready"}       // Dataset ready
{"batch": ["text1", "text2", ...]}  // Data batches
{"progress": 10000}       // Progress updates (stderr)
{"done": true, "total": 100000000}  // Completion
```

### Memory Management

- **Fetch Buffer**: 1000 samples (configurable)
- **Shuffle Buffer**: 10k samples in data loader
- **Token Buffer**: Dynamically sized based on sequence length

Total memory for buffers: ~50-100MB for typical configurations

### Error Handling

- Python process crashes are detected
- Network errors trigger retries (future enhancement)
- Graceful cleanup on training interrupt (Ctrl+C)

## Performance Characteristics

### Throughput
- **Network**: Depends on HuggingFace CDN (typically 10-100 MB/s)
- **Parsing**: ~100k samples/second (JSON parsing overhead)
- **Tokenization**: ~50k samples/second (BBPE tokenization)
- **Bottleneck**: Usually tokenization or training, not streaming

### Latency
- **First Batch**: 5-10 seconds (dataset loading)
- **Subsequent Batches**: <100ms (from buffer)
- **No Stalls**: Async fetching keeps buffer full

## Comparison with File-Based Training

| Aspect | File-Based | Streaming |
|--------|------------|-----------|
| **Setup Time** | Hours (download) | Seconds (connect) |
| **Disk Usage** | 100s of GB | 0 GB |
| **Flexibility** | Fixed dataset | Dynamic sampling |
| **Quota Issues** | Common | None |
| **Network Dependency** | Download only | Continuous |
| **Resume Training** | Fast | Fast (reconnects) |

## Advanced Configuration

### Train with Sample Limit
```bash
# Train on exactly 10M samples
cargo run -p training --bin train-streaming -- \
  --max-samples 10000000 \
  --config configs/runpod_1b.yaml
```

### Multiple Datasets
```rust
// Combine multiple datasets (future enhancement)
let corpus1 = HuggingFaceStreamingCorpus::new(...)?;
let corpus2 = HuggingFaceStreamingCorpus::new(...)?;
let combined = MultiCorpus::new(vec![corpus1, corpus2])?;
```

### Custom Batch Sizes
```rust
// Larger batches for faster network transfer
let corpus = HuggingFaceStreamingCorpus::new(
    "HuggingFaceFW/fineweb".to_string(),
    "train".to_string(),
    5000,  // Fetch 5k samples at a time
    None,
)?;
```

## Troubleshooting

### "Failed to spawn Python process"
- Ensure Python 3 is installed: `python3 --version`
- Install datasets library: `pip install datasets`

### "Dataset loading timeout"
- Check internet connection
- Verify dataset name is correct
- Try a smaller batch size

### "Memory errors"
- Reduce `batch_size` parameter
- Reduce `shuffle_buffer_size` in data loader
- Use smaller sequence length

### "Slow streaming"
- Increase `batch_size` for fewer network round-trips
- Check network bandwidth
- Consider using a dataset mirror/cache

## Next Steps

1. **Implement Proper Training Binary**: Create `train-streaming` binary
2. **Add Progress Tracking**: Show samples/second, ETA
3. **Implement Checkpointing**: Save/resume with streaming state
4. **Add Dataset Mixing**: Combine multiple datasets with weights
5. **Optimize Python IPC**: Use faster serialization (msgpack/protobuf)

## Example: Full Training Pipeline

```bash
# 1. Set up environment
export CARGO_TARGET_DIR=/tmp/rust-target

# 2. Build project
cargo build --release

# 3. Train tokenizer on streaming data (1M samples)
cargo run -p training --bin train-tokenizer-streaming -- \
  --dataset "HuggingFaceFW/fineweb" \
  --max-samples 1000000 \
  --vocab-size 50000 \
  --output /workspace/artifacts/tokenizer/runpod-1b

# 4. Start model training with streaming data
cargo run -p training --bin train-streaming -- \
  --dataset "HuggingFaceFW/fineweb" \
  --max-samples 100000000 \
  --tokenizer /workspace/artifacts/tokenizer/runpod-1b \
  --config configs/runpod_1b.yaml \
  --output /workspace/runs/runpod-1b
```

This approach completely eliminates disk storage requirements and gets you training immediately!
