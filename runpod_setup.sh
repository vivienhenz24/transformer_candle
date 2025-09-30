#!/bin/bash

# RunPod Complete Setup Script for 1B Transformer Training
# Run this after SSH'ing into your new RunPod pod

set -e  # Exit on any error

echo "🚀 Starting RunPod Setup for 1B Transformer Training..."

# Update system
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y tmux curl wget git build-essential

# Install Rust
echo "🦀 Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default stable

# Set up environment
echo "🔧 Setting up environment..."
export CARGO_TARGET_DIR=/tmp/rust-target
export CUDA_VISIBLE_DEVICES=0
export CUDA_COMPUTE_CAP=89
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to bashrc for persistence
echo 'export CARGO_TARGET_DIR=/tmp/rust-target' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export CUDA_COMPUTE_CAP=89' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'source $HOME/.cargo/env' >> ~/.bashrc

# Navigate to workspace
cd /workspace

# Clone the repository (if not already there)
if [ ! -d "transformer_candle" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/vivienhenz24/transformer_candle.git
fi

cd transformer_candle

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install datasets huggingface-hub tokenizers transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/runs/rtx5090-1b
mkdir -p /workspace/cache/rtx5090-1b
mkdir -p /workspace/tmp_streaming_shards
mkdir -p /workspace/artifacts/tokenizer/rtx5090-1b

# Build the project
echo "🔨 Building Rust project (this takes ~5-10 minutes)..."
cargo build --release

# Train tokenizer from streaming data
echo "📝 Training tokenizer from streaming data..."
cargo run --release --bin orchestrate -- \
    --experiment rtx5090-1b \
    --stream \
    --dataset "HuggingFaceFW/fineweb" \
    --split train \
    --max-samples 5000000 \
    --vocab-size 50000

# Update training config for streaming
echo "📝 Updating training configuration..."
sed -i 's|train_shards: \[.*\]|train_shards: ["/workspace/tmp_streaming_shards/shard_0000.txt"]|' configs/rtx5090_1b.yaml
sed -i 's|validation_shards: \[.*\]|validation_shards: ["/workspace/tmp_streaming_shards/shard_0000.txt"]|' configs/rtx5090_1b.yaml
sed -i 's|batch_size: 1|batch_size: 64|' configs/rtx5090_1b.yaml
sed -i 's|gradient_accumulation_steps: 64|gradient_accumulation_steps: 1|' configs/rtx5090_1b.yaml

# Test GPU access
echo "🎮 Testing GPU access..."
python3 -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try:
    import torch
    print('✅ PyTorch CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✅ GPU name:', torch.cuda.get_device_name(0))
        print('✅ GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
    else:
        print('❌ GPU not available - check NVML')
except Exception as e:
    print('❌ PyTorch error:', e)
"

# Create a training script
cat > start_training.sh << 'EOF'
#!/bin/bash
cd /workspace/transformer_candle
export CARGO_TARGET_DIR=/tmp/rust-target
export CUDA_VISIBLE_DEVICES=0
export CUDA_COMPUTE_CAP=89
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "🚀 Starting 1B Transformer Training..."
echo "📊 Using RTX 5090 with 32GB VRAM"
echo "📈 Training will take ~3.5 days"
echo "💾 Checkpoints saved every 1500 steps"

cargo run --release --bin train -- --config configs/rtx5090_1b.yaml
EOF

chmod +x start_training.sh

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Test GPU: python3 -c \"import torch; print(torch.cuda.is_available())\""
echo "2. Start tmux: tmux new-session -s training"
echo "3. Run training: cd /workspace/transformer_candle && ./start_training.sh"
echo "4. Detach from tmux: Ctrl+B, then D"
echo ""
echo "🔍 Monitor training:"
echo "- Reattach to tmux: tmux attach -t training"
echo "- List tmux sessions: tmux ls"
echo "- Kill tmux session: tmux kill-session -t training"
echo ""
echo "💡 Training will:"
echo "- Stream 30M samples from HuggingFace"
echo "- Initialize 1B parameter model"
echo "- Train for ~3.5 days"
echo "- Save checkpoints every 1500 steps"
echo ""
echo "⚠️  IMPORTANT:"
echo "- Make sure GPU is detected before training!"
echo "- If nvidia-smi fails, restart the pod"
echo "- Training on CPU will take months (not feasible)"
echo ""
echo "🚀 Ready to train your 1B transformer!"
