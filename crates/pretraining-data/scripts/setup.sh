#!/bin/bash
# Setup script for Hugging Face dataset integration

set -e

echo "🚀 Setting up Python environment for HF dataset downloads..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r scripts/requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Download datasets: python scripts/download_datasets.py --datasets openwebtext wikipedia"
echo "3. Process data: cargo run --bin preprocess -- --mixed"