# 🎮 RTX 5090 Training Guide (32GB VRAM)

## 🔧 Optimized 1B Model for RTX 5090

The RTX 5090 has 32GB VRAM (vs A100's 80GB), so we've optimized the config:

### Key Optimizations

| Setting | A100 (80GB) | RTX 5090 (32GB) | Reason |
|---------|-------------|-----------------|--------|
| **Batch Size** | 2 | 1 | Half VRAM → half batch |
| **Gradient Accum** | 32 | 64 | Double to maintain effective batch |
| **Shuffle Buffer** | 65536 | 32768 | Reduce memory overhead |
| **Num Workers** | 8 | 6 | Fewer workers = less memory |
| **Max Checkpoints** | 10 | 5 | Save disk space |
| **Eval Batches** | 32 | 16 | Less memory during eval |

**Effective batch size remains the same:** `batch_size × gradient_accumulation_steps = 64`

## 🚀 Training Workflow

### 1. Environment Setup

```bash
# Set critical env vars
export CARGO_TARGET_DIR=/tmp/rust-target
echo 'export CARGO_TARGET_DIR=/tmp/rust-target' >> ~/.bashrc

export HF_HOME=/workspace/hf_cache
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# Install Python deps
pip install datasets
```

### 2. Build

```bash
cd /workspace/transformer_candle
cargo build --release
```

### 3. Train Tokenizer (Streaming)

```bash
cargo run -p training --bin orchestrate -- \
  --stream \
  --dataset "HuggingFaceFW/fineweb" \
  --split train \
  --max-samples 1000000 \
  --vocab-size 50000 \
  --experiment rtx5090-1b
```

This creates:
- `/workspace/artifacts/tokenizer/rtx5090-1b/` - Tokenizer files
- `/workspace/runs/rtx5090-1b/streaming_config.json` - Streaming metadata

### 4. Create Training Config

```bash
# Copy the optimized config
cp configs/rtx5090_1b.yaml /workspace/runs/rtx5090-1b/training.yaml

# Create special tokens
cat > /workspace/runs/rtx5090-1b/special_tokens.txt <<'EOF'
["<|endoftext|>", "<|pad|>"]
EOF
```

### 5. Start Training!

```bash
cargo run -p training --bin train -- \
  --config /workspace/runs/rtx5090-1b/training.yaml
```

The train binary will automatically detect `streaming_config.json` and use streaming mode!

## 📊 Expected Memory Usage

```
Model Parameters:     ~4-5 GB   (1B params × BF16)
Optimizer States:     ~8-10 GB  (AdamW: 2 states per param)
Gradients:            ~4-5 GB   (Same as model)
Activations:          ~8-10 GB  (batch=1, seq=2048)
Training Overhead:    ~2-3 GB   (misc buffers)
─────────────────────────────────────────────
Total:                ~26-33 GB (fits in 32GB!)
```

## ⚠️ If You Get OOM (Out of Memory)

Try these in order:

### Option 1: Reduce Sequence Length
```yaml
data:
  sequence_length: 1024  # Instead of 2048
```
**Saves:** ~4-5 GB

### Option 2: Use Grouped Query Attention (GQA)
```yaml
model:
  num_key_value_heads: 8  # Instead of 32
```
**Saves:** ~1-2 GB in KV cache

### Option 3: Increase Gradient Accumulation
```yaml
data:
  batch_size: 1
  gradient_accumulation_steps: 128  # Instead of 64
```
**Saves:** 0 GB (but slower training)

### Option 4: Reduce Workers
```yaml
data:
  num_workers: 4  # Instead of 6
```
**Saves:** ~500 MB

## 🎯 Training Speed Estimate

- **RTX 5090**: ~0.8-1.0 steps/sec (batch=1, grad_accum=64)
- **A100 80GB**: ~1.2-1.5 steps/sec (batch=2, grad_accum=32)

**Total training time (300k steps):**
- RTX 5090: ~3.5-4 days
- A100: ~2.5-3 days

## 💡 Pro Tips

1. **Monitor VRAM Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Use BF16** (already configured):
   - RTX 5090 supports BF16 natively
   - Better numerical stability than FP16
   - Same memory as FP16

3. **Gradient Accumulation is Free**:
   - No memory cost (activations recomputed)
   - Same effective batch size
   - Just slower steps

4. **Stream Forever**:
   ```bash
   # No --max-samples = infinite data!
   cargo run -p training --bin orchestrate -- \
     --stream \
     --vocab-size 50000 \
     --experiment rtx5090-1b-unlimited
   ```

## 📈 Scaling Options

### Smaller Model (500M - more headroom)
```yaml
model:
  hidden_size: 1536       # Instead of 2048
  intermediate_size: 6144 # 4x hidden
  num_layers: 16          # Instead of 20
```

### Larger Batch (if memory allows)
```yaml
data:
  batch_size: 2
  gradient_accumulation_steps: 32
```

## ✅ Ready to Train!

Your RTX 5090 is perfectly capable of training a 1B parameter model with these optimizations. The key is:
- ✅ Smaller batch size (1 vs 2)
- ✅ More gradient accumulation (64 vs 32)
- ✅ Same effective batch size (64)
- ✅ Streaming data (no disk usage)

**Start training and monitor `nvidia-smi` for the first few minutes to ensure VRAM stays under 32GB!** 🚀
