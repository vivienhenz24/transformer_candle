#!/bin/bash
# 🚀 RunPod Restart Commands - Copy/Paste This Once You SSH In

echo "=== Step 1: Stop Current Training ==="
# First, let's check if training is running
tmux ls 2>/dev/null || echo "No tmux sessions found"

# Attach to training session and you'll need to press Ctrl+C manually
echo "About to attach to training session..."
echo "⚠️  PRESS Ctrl+C TO STOP TRAINING, then Ctrl+D to detach"
read -p "Press Enter to continue..."
tmux attach -t training || echo "No training session to stop"

echo ""
echo "=== Step 2: Navigate to Project ==="
cd ~/transformer_candle
pwd

echo ""
echo "=== Step 3: Check Current Status ==="
echo "Current config files:"
ls -lh configs/*.yaml

echo ""
echo "Git status:"
git status --short || echo "Not a git repo"

echo ""
echo "=== Step 4: Pull Latest Code (from your local machine) ==="
echo "⚠️  STOP HERE - Run this command from your LOCAL machine (not RunPod):"
echo ""
echo "rsync -avz --progress \\"
echo "  --exclude 'target/' \\"
echo "  --exclude '.git/' \\"
echo "  --exclude 'runs/' \\"
echo "  ~/Desktop/transformer/ \\"
echo "  root@YOUR_RUNPOD_IP:~/transformer_candle/"
echo ""
read -p "Have you run rsync from your local machine? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please run rsync first, then run this script again"
    exit 1
fi

echo ""
echo "=== Step 5: Rebuild with Gradient Clipping ==="
cargo build --release 2>&1 | tee build.log

# Check if build succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed! Check build.log"
    exit 1
fi

echo ""
echo "=== Step 6: Clean Old Runs (Optional) ==="
echo "This will delete old checkpoints to start fresh"
read -p "Delete old 550M checkpoints? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf /workspace/runs/rtx5090-550m/checkpoints/*
    rm -rf /workspace/runs/rtx5090-550m/tensorboard/*
    echo "✅ Old checkpoints deleted"
fi

echo ""
echo "=== Step 7: Verify Config ==="
echo "Checking rtx5090_550m.yaml config..."
if [ -f configs/rtx5090_550m.yaml ]; then
    echo "✅ Config exists!"
    echo ""
    echo "Key settings:"
    grep -E "hidden_size|num_layers|learning_rate|max_grad_norm|batch_size|sequence_length" configs/rtx5090_550m.yaml
else
    echo "❌ Config not found! Make sure you synced it from local machine"
    exit 1
fi

echo ""
echo "=== Step 8: Start Training in tmux ==="
echo "This will start training in a background tmux session"
read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Kill old session if exists
    tmux kill-session -t training 2>/dev/null || true
    
    # Start new training session
    tmux new-session -d -s training "cd ~/transformer_candle && cargo run --release --bin train -- --config configs/rtx5090_550m.yaml"
    
    echo "✅ Training started in tmux session 'training'"
    echo ""
    echo "To monitor progress:"
    echo "  tmux attach -t training"
    echo ""
    echo "To detach from tmux (leave training running):"
    echo "  Press Ctrl+B, then D"
    echo ""
    echo "Waiting 5 seconds, then showing initial output..."
    sleep 5
    tmux capture-pane -t training -p | tail -50
else
    echo "Skipped training start"
    echo ""
    echo "To start manually:"
    echo "  tmux new-session -s training"
    echo "  cd ~/transformer_candle"
    echo "  cargo run --release --bin train -- --config configs/rtx5090_550m.yaml"
fi

echo ""
echo "=== ✅ Setup Complete! ==="
echo ""
echo "📊 Monitor your training:"
echo "  tmux attach -t training          # Attach to see live logs"
echo "  watch -n 1 nvidia-smi            # Monitor GPU usage"
echo "  tail -f /workspace/runs/rtx5090-550m/tensorboard/events.out.tfevents.*  # TensorBoard logs"
echo ""
echo "🎯 What to expect:"
echo "  Step 100:  loss ~8.0-8.5, grad ~0.9"
echo "  Step 500:  loss ~6.0-6.5, grad ~0.8"
echo "  Step 1000: loss ~5.0-5.5, grad ~0.7"
echo "  Step 2000: loss ~4.0-4.5, grad ~0.6"
echo ""
echo "⚠️  If loss is still >7 after 1000 steps, STOP and investigate!"

