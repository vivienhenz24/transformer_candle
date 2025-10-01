# 🚀 Complete RunPod Setup (Including Rust Installation)

## Step 1: SSH Into Your Pod

```bash
ssh root@<pod-ip> -p <port>
```

---

## Step 2: Install Rust/Cargo (5 minutes)

```bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Load Rust into current session
source $HOME/.cargo/env

# Verify installation
cargo --version
# Should output: cargo 1.xx.x

rustc --version
# Should output: rustc 1.xx.x
```

---

## Step 3: Set Critical Environment Variables

```bash
# Navigate to workspace
cd /workspace

# Set CARGO_TARGET_DIR (CRITICAL - avoids disk quota issues)
export CARGO_TARGET_DIR=/tmp/rust-target
echo 'export CARGO_TARGET_DIR=/tmp/rust-target' >> ~/.bashrc

# Set HuggingFace cache
export HF_HOME=/workspace/hf_cache
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# Make sure Cargo is in PATH (persist across sessions)
echo 'source $HOME/.cargo/env' >> ~/.bashrc

# Reload environment
source ~/.bashrc

# Verify
echo $CARGO_TARGET_DIR
# Should output: /tmp/rust-target

cargo --version
# Should output cargo version
```

---

## Step 4: Check GPU and System

```bash
# Check GPU
nvidia-smi
# Should show your RTX 5090 or A100

# Check disk space
df -h /workspace
# Should show ~500TB available

df -h /tmp
# Should show plenty of space for build artifacts
```

---

## Step 5: Clone Repository

```bash
cd /workspace

# Clone your repo (replace with your actual repo URL)
git clone https://github.com/yourusername/transformer.git transformer_candle

# Enter directory
cd transformer_candle

# Verify code is there
ls -la
# Should see: Cargo.toml, crates/, configs/, README.md, etc.
```

---

## Step 6: Install Python Dependencies

```bash
# Install datasets library (needed for streaming)
pip install datasets

# Verify
python -c "import datasets; print('✅ datasets installed')"
# Should output: ✅ datasets installed
```

---

## Step 7: Build the Project (3-5 minutes)

```bash
cd /workspace/transformer_candle

# Build in release mode
cargo build --release

# This will take 3-5 minutes
# Lots of "Compiling..." messages
# Should end with "Finished release [optimized] target(s)"
```

---

## Step 8: Create Directory Structure

```bash
mkdir -p /workspace/artifacts/tokenizer
mkdir -p /workspace/runs
mkdir -p /workspace/cache
mkdir -p /workspace/hf_cache

# Verify
ls -la /workspace/
```

---

## Step 9: Train Tokenizer (5-10 minutes)

### For RTX 5090 (32GB):

```bash
cd /workspace/transformer_candle

cargo run --release -p training --bin orchestrate -- \
  --stream \
  --dataset "HuggingFaceFW/fineweb" \
  --split train \
  --max-samples 1000000 \
  --vocab-size 50000 \
  --experiment rtx5090-1b
```

### For A100 (80GB):

```bash
cd /workspace/transformer_candle

cargo run --release -p training --bin orchestrate -- \
  --stream \
  --dataset "HuggingFaceFW/fineweb" \
  --split train \
  --max-samples 1000000 \
  --vocab-size 50000 \
  --experiment runpod-1b
```

---

## Step 10: Setup Training Configuration

### For RTX 5090:

```bash
# Copy config
cp configs/rtx5090_1b.yaml /workspace/runs/rtx5090-1b/training.yaml

# Create special tokens
cat > /workspace/runs/rtx5090-1b/special_tokens.txt <<'EOF'
["<|endoftext|>", "<|pad|>"]
EOF

# Verify
ls -la /workspace/runs/rtx5090-1b/
```

### For A100:

```bash
# Copy config
cp configs/runpod_1b.yaml /workspace/runs/runpod-1b/training.yaml

# Create special tokens
cat > /workspace/runs/runpod-1b/special_tokens.txt <<'EOF'
["<|endoftext|>", "<|pad|>"]
EOF

# Verify
ls -la /workspace/runs/runpod-1b/
```

---

## Step 11: Start Training! 🚀

### For RTX 5090:

```bash
cd /workspace/transformer_candle

cargo run --release -p training --bin train -- \
  --config /workspace/runs/rtx5090-1b/training.yaml
```

### For A100:

```bash
cd /workspace/transformer_candle

cargo run --release -p training --bin train -- \
  --config /workspace/runs/runpod-1b/training.yaml
```

---

## Step 12: Run in Background (Recommended)

### Using tmux (Recommended):

```bash
# Install tmux if not available
apt-get update && apt-get install -y tmux

# Create a new tmux session
tmux new -s training

# Inside tmux, start training:
cd /workspace/transformer_candle
cargo run --release -p training --bin train -- \
  --config /workspace/runs/rtx5090-1b/training.yaml

# Detach from tmux: Press Ctrl+B, then press D

# Reattach later:
tmux attach -t training

# List sessions:
tmux ls
```

### Using screen (Alternative):

```bash
# Install screen if not available
apt-get update && apt-get install -y screen

# Create new screen session
screen -S training

# Inside screen, start training:
cd /workspace/transformer_candle
cargo run --release -p training --bin train -- \
  --config /workspace/runs/rtx5090-1b/training.yaml

# Detach: Press Ctrl+A, then press D

# Reattach:
screen -r training
```

---

## 📋 Quick Copy-Paste Full Setup Script

Save this as `setup.sh` and run it:

```bash
#!/bin/bash
set -e

echo "🚀 RunPod Transformer Training Setup"
echo "===================================="

# 1. Install Rust
echo "📦 Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 2. Set environment variables
echo "⚙️  Setting environment variables..."
export CARGO_TARGET_DIR=/tmp/rust-target
export HF_HOME=/workspace/hf_cache
echo 'export CARGO_TARGET_DIR=/tmp/rust-target' >> ~/.bashrc
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
echo 'source $HOME/.cargo/env' >> ~/.bashrc

# 3. Install Python deps
echo "🐍 Installing Python dependencies..."
pip install datasets

# 4. Create directories
echo "📁 Creating directories..."
mkdir -p /workspace/artifacts/tokenizer
mkdir -p /workspace/runs
mkdir -p /workspace/cache
mkdir -p /workspace/hf_cache

# 5. Install tmux
echo "🖥️  Installing tmux..."
apt-get update && apt-get install -y tmux

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Clone your repo to /workspace/transformer_candle"
echo "2. cd /workspace/transformer_candle"
echo "3. cargo build --release"
echo "4. Follow the training steps in the guide!"
```

To use it:

```bash
cd /workspace
nano setup.sh
# Paste the script above, save (Ctrl+X, Y, Enter)

chmod +x setup.sh
./setup.sh
```

---

## ⚠️ Common Issues

### "cargo: command not found"
**Solution:**
```bash
source $HOME/.cargo/env
```

### "rustc: command not found"
**Solution:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### "Disk quota exceeded"
**Solution:**
```bash
export CARGO_TARGET_DIR=/tmp/rust-target
echo 'export CARGO_TARGET_DIR=/tmp/rust-target' >> ~/.bashrc
```

### "ModuleNotFoundError: No module named 'datasets'"
**Solution:**
```bash
pip install datasets
```

---

## ✅ Verification Checklist

Before starting training, verify:

- [ ] `cargo --version` works
- [ ] `rustc --version` works
- [ ] `echo $CARGO_TARGET_DIR` shows `/tmp/rust-target`
- [ ] `python -c "import datasets"` works
- [ ] `nvidia-smi` shows your GPU
- [ ] Code is in `/workspace/transformer_candle`
- [ ] `cargo build --release` completes successfully

---

## 🎯 Full Timeline

| Step | Duration |
|------|----------|
| Install Rust | 2-3 min |
| Install Python deps | 1 min |
| Clone repo | 1 min |
| Build project | 3-5 min |
| Train tokenizer | 5-10 min |
| Setup config | 1 min |
| **Start training!** | 3-4 days |

**Total setup time: ~15-20 minutes**

---

## 🚀 You're Ready!

Once Rust is installed, everything else will work smoothly. Let's get training! 🎉
