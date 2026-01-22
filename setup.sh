#!/bin/bash

# Fin Whale SSL Setup Script
# This script sets up the environment for the fin_whale_ssl project

set -e  # Exit on error

echo "========================================="
echo "Fin Whale SSL Environment Setup"
echo "========================================="

# Step 1: Install uv
echo ""
echo "[1/9] Installing uv..."
pip install uv

# Step 2: Clone stable-pretraining dependency
echo ""
echo "[2/9] Cloning stable-pretraining repository..."
git clone git@github.com:galilai-group/stable-pretraining.git

# Step 3: Create virtual environment with uv
echo ""
echo "[3/9] Creating virtual environment with uv..."
uv venv

# Step 4: Activate virtual environment
echo ""
echo "[4/9] Activating virtual environment..."
source .venv/bin/activate

# Step 5: Install PyTorch
echo ""
echo "[5/9] Installing PyTorch with uv..."
uv pip install torch torchvision torchaudio

# Step 6: Install project dependencies
echo ""
echo "[6/9] Installing project dependencies with uv..."
cd stable-pretraining
uv pip install -e ".[all]"

# Step 7: Install soundfile
echo ""
echo "[7/9] Installing soundfile..."
uv pip install soundfile

# Step 8: Login to WandB
echo ""
echo "[8/9] Setting up WandB login..."
echo "Please login to WandB (you'll need your API key from https://wandb.ai/authorize)"
wandb login

# Step 9: Login to Hugging Face
echo ""
echo "[9/9] Setting up Hugging Face login..."
echo "Please login to Hugging Face (you'll need your token from https://huggingface.co/settings/tokens)"
huggingface-cli login

echo ""
echo "========================================="
echo "✅ Setup completed successfully!"
echo "========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
