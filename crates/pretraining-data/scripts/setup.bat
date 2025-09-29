@echo off
REM Setup script for Hugging Face dataset integration (Windows)

echo 🚀 Setting up Python environment for HF dataset downloads...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing Python dependencies...
pip install -r scripts\requirements.txt

echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo 1. Activate the environment: venv\Scripts\activate.bat
echo 2. Download datasets: python scripts\download_datasets.py --datasets openwebtext wikipedia
echo 3. Process data: cargo run --bin preprocess -- --mixed