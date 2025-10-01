# 🎯 Simple RunPod Steps - Just Copy/Paste These

## Step 1️⃣: SSH into RunPod
```bash
ssh root@YOUR_RUNPOD_IP
```

## Step 2️⃣: Stop Current Training
```bash
# See if training is running
tmux ls

# Attach to training session
tmux attach -t training

# Press Ctrl+C to STOP training
# Press Ctrl+D to exit tmux
```

## Step 3️⃣: Sync Code from Your Mac
**⚠️ Run this on YOUR MAC (not on RunPod):**
```bash
cd ~/Desktop/transformer

# Replace YOUR_RUNPOD_IP with actual IP
rsync -avz --progress \
  --exclude 'target/' \
  --exclude '.git/' \
  --exclude 'runs/' \
  . \
  root@YOUR_RUNPOD_IP:~/transformer_candle/
```

## Step 4️⃣: Back on RunPod - Rebuild
```bash
cd ~/transformer_candle

# Rebuild with gradient clipping
cargo build --release
```

## Step 5️⃣: Start New Training
```bash
# Clean old runs (optional)
rm -rf /workspace/runs/rtx5090-550m/checkpoints/*
rm -rf /workspace/runs/rtx5090-550m/tensorboard/*

# Start training in tmux
tmux new-session -d -s training \
  "cd ~/transformer_candle && cargo run --release --bin train -- --config configs/rtx5090_550m.yaml"

# Attach to see it running
tmux attach -t training
```

## Step 6️⃣: Monitor Training
```bash
# To attach to see logs
tmux attach -t training

# To detach (leave training running)
# Press: Ctrl+B, then D

# To monitor GPU
watch -n 1 nvidia-smi
```

---

## 🎯 What You Should See:

### ✅ GOOD (Learning):
```
[progress] step    100 | seq 1024 | tokens  65536 | loss   8.1234 | grad  0.9234 | lr  3.000e-5
[progress] step    500 | seq 1024 | tokens  65536 | loss   6.2341 | grad  0.8123 | lr  1.500e-4
[progress] step   1000 | seq 1024 | tokens  65536 | loss   5.1234 | grad  0.7234 | lr  3.000e-4
[progress] step   2000 | seq 1024 | tokens  65536 | loss   4.0123 | grad  0.6345 | lr  6.000e-4
```
**Loss is going DOWN**, gradient stays under 1.0 ✅

### ❌ BAD (Not Learning):
```
[progress] step   1000 | seq 1024 | tokens  65536 | loss   7.5000 | grad  0.9999 | lr  3.000e-4
```
**Loss stuck at 7+** → STOP and ask for help ❌

---

## 🆘 Quick Commands:

### See if training is running:
```bash
tmux ls
ps aux | grep train
```

### Stop training:
```bash
tmux attach -t training
# Press Ctrl+C
```

### Check GPU usage:
```bash
nvidia-smi
```

### Check disk space:
```bash
df -h /workspace
```

### Tail the logs:
```bash
tmux capture-pane -t training -p | tail -100
```

---

## 📊 Key Config Changes (550M Model):

| Setting | Old (1.2B) | New (550M) | Why |
|---------|-----------|-----------|-----|
| hidden_size | 1280 | 896 | Smaller model |
| num_layers | 16 | 12 | Faster training |
| sequence_length | 512 | 1024 | More tokens/step |
| learning_rate | 1.8e-4 | 6.0e-4 | Faster learning |
| **max_grad_norm** | ❌ None | ✅ **1.0** | **Prevents explosion!** |
| precision | fp32 | bf16 | Faster + saves memory |
| total_steps | 300K | 500K | More training |

---

## 🎉 Why This Will Work:

1. ✅ **Gradient clipping** prevents explosions (your main issue!)
2. ✅ **Smaller model** = more stable
3. ✅ **Higher LR** = faster convergence
4. ✅ **More tokens** = better quality
5. ✅ **BF16** = 2x faster training

Expected timeline:
- **3 days** → 10K steps → loss ~2.5-3.0
- **2 weeks** → 50K steps → loss ~1.8-2.2
- **1 month** → 100K steps → loss ~1.5-1.8

Good luck! 🚀

