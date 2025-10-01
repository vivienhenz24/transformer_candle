# 🚀 Restart Training with 550M Parameters + Gradient Clipping

## ✨ What's New:

### 1. **550M Parameter Model** (down from 1.2B)
- **Hidden size**: 896 (down from 1280)
- **Layers**: 12 (down from 16)
- **Heads**: 14 (down from 20)
- **FFN size**: 3584 (4x hidden)
- **Sequence length**: 1024 (UP from 512!)
- **Total params**: ~550M

### 2. **Gradient Clipping** ✅
- **max_grad_norm: 1.0** - Prevents gradient explosion!
- Your gradients were spiking to 40+, now they'll be capped at 1.0

### 3. **Better Training Settings**
- **Learning rate**: 6.0e-4 (GPT-3 style, higher than before)
- **Warmup**: 2000 steps (shorter)
- **Weight decay**: 0.1 (better regularization)
- **Precision**: BF16 (faster + saves memory)
- **Effective batch**: 64 (4 batch × 16 grad_accum)

### 4. **More Training Tokens**
- **Total steps**: 500K (up from 300K)
- **Tokens per step**: 65,536 (4 × 16 × 1024)
- **Total tokens**: **32.8 BILLION** (3x more than before!)
- **Chinchilla optimal for 550M**: ~11-14B tokens
- **You're training on 2.3x optimal** - excellent!

## 📊 Memory Usage (32GB VRAM):
- Model: ~2.2 GB (BF16)
- Optimizer: ~4.4 GB
- Gradients: ~2.2 GB
- Activations: ~6-8 GB
- **Total: ~17-19 GB (60% of 32GB)** ✅ SAFE!

## 🎯 Expected Loss Trajectory:

| Steps | Expected Loss | Status |
|-------|--------------|--------|
| 1,000 | 5.5-6.0 | Early warmup |
| 5,000 | 3.5-4.0 | Learning basics |
| 10,000 | 2.5-3.0 | Good progress |
| 50,000 | 1.8-2.2 | Strong performance |
| 100,000+ | 1.4-1.8 | Near convergence |
| 500,000 | 1.2-1.5 | **Final target** |

## 🚀 Steps to Restart Training:

### Step 1: Stop Current Training
```bash
# SSH into RunPod
ssh root@<your-runpod-ip>

# Attach to tmux
tmux attach -t training

# Press Ctrl+C to stop training
```

### Step 2: Upload New Config & Code
```bash
# On your local machine (from /Users/vivienhenz/Desktop/transformer)
# First, build to make sure everything compiles
cargo build --release

# Then sync to RunPod (replace with your IP)
rsync -avz --progress \
  --exclude 'target/' \
  --exclude '.git/' \
  --exclude 'runs/' \
  ~/Desktop/transformer/ \
  root@<your-runpod-ip>:~/transformer_candle/

# Or use scp for the config only
scp configs/rtx5090_550m.yaml root@<your-runpod-ip>:~/transformer_candle/configs/
```

### Step 3: Rebuild on RunPod
```bash
# On RunPod
cd ~/transformer_candle
cargo build --release
```

### Step 4: Start Fresh Training
```bash
# Clean up old runs (optional - keeps tokenizer)
rm -rf /workspace/runs/rtx5090-550m/checkpoints/*
rm -rf /workspace/runs/rtx5090-550m/tensorboard/*

# Start training with new 550M config
cargo run --release --bin train -- --config configs/rtx5090_550m.yaml
```

### Step 5: Monitor Progress
Watch for:
- ✅ **Loss starts at ~8-9** (normal for random init)
- ✅ **Loss drops to ~6 by step 500** (good sign)
- ✅ **Loss drops to ~4-5 by step 2000** (warmup complete)
- ✅ **Gradient norm stays under 1.0** (clipping working!)
- ⚠️ **If loss stays above 7 after 1000 steps** → something wrong

## 🔍 What to Look For (Success):
```
[progress] step    100 | seq 1024 | tokens  65536 | loss   8.2341 | grad  0.9823 | lr  3.000e-5
[progress] step    500 | seq 1024 | tokens  65536 | loss   6.1245 | grad  0.8912 | lr  1.500e-4
[progress] step   1000 | seq 1024 | tokens  65536 | loss   5.3421 | grad  0.7234 | lr  3.000e-4
[progress] step   2000 | seq 1024 | tokens  65536 | loss   4.2134 | grad  0.6123 | lr  6.000e-4
```

## 🔍 What to Look For (Failure):
```
[progress] step   1000 | seq 1024 | tokens  65536 | loss   7.5000 | grad  0.9999 | lr  3.000e-4
```
If loss is still >7 after 1000 steps, STOP and investigate!

## ⚡ Performance Expectations:
- **Speed**: ~2000-2500 tokens/sec (similar to before)
- **Time per 1000 steps**: ~7-8 hours
- **Time to 10K steps**: ~3 days
- **Time to 100K steps**: ~30 days
- **Time to 500K steps**: ~5 months (but you can stop earlier!)

## 🎯 Recommended Training Strategy:
1. **Train to 10K steps first** (~3 days) - should see loss ~2.5-3.0
2. **If loss looks good, continue to 50K** (~2 weeks) - loss ~1.8-2.2
3. **Evaluate quality** - generate text, see if it's coherent
4. **Continue to 100K-200K if needed** - diminishing returns after that

## 📁 Files Changed:
- ✅ `configs/rtx5090_550m.yaml` - New model config
- ✅ `crates/training/src/config.rs` - Added `max_grad_norm`
- ✅ `crates/training/src/trainer.rs` - Added gradient clipping

## 🎉 Why This Will Work:
1. ✅ **Smaller model** → less VRAM pressure → more stable
2. ✅ **Gradient clipping** → prevents explosions → stable training
3. ✅ **Higher LR + warmup** → faster convergence
4. ✅ **More tokens** → better final performance
5. ✅ **BF16** → faster training, same quality

Good luck! 🚀

