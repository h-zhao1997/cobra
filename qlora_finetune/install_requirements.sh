#!/bin/bash
# Staged installation script to handle dependency conflicts
# This installs packages in the correct order to avoid build-time dependency issues
# All packages are installed in the virtual environment at ./env

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists, create if not
if [ ! -d "env" ]; then
    echo "Creating virtual environment at ./env..."
    python3 -m venv env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

echo "Step 1: Installing build dependencies..."
pip install --upgrade pip setuptools wheel packaging

echo "Step 2: Installing PyTorch and core dependencies..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio triton==2.1.0

echo "Step 3: Installing transformers ecosystem..."
pip install transformers==4.34.1 \
    'tokenizers>=0.14,<0.15' \
    accelerate==0.26.1 \
    'huggingface_hub>=0.16.4,<0.18' \
    'numpy>=1.17,<2.0.0' \
    'dill<0.3.9,>=0.3.0' \
    'fsspec>=2023.1.0,<2023.11.0'

echo "Step 4: Installing vision and utility libraries..."
pip install einops \
    timm==0.9.10 \
    pillow \
    wandb \
    jsonlines \
    rich \
    tqdm

echo "Step 5: Installing QLoRA dependencies..."
pip install peft==0.7.1 \
    'bitsandbytes>=0.41.0' \
    trl==0.7.4 \
    'datasets>=2.14.0,<2.18.0'

echo "Step 6: Installing Mamba SSM (requires torch to be installed first)..."
echo "Note: mamba-ssm requires --no-build-isolation to see torch in build environment"
pip install --no-build-isolation 'mamba-ssm<2.0.0' causal-conv1d

echo "Installation complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source $SCRIPT_DIR/env/bin/activate"

