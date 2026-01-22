#!/bin/bash

# Fin Whale SSL Setup Script
# This script sets up the environment for the fin_whale_ssl project

set -e  # Exit on error

echo "========================================="
echo "Fin Whale SSL Environment Setup"
echo "========================================="

# Step 1: Install uv
echo ""
echo "[1/11] Installing uv..."
pip install uv

# Step 2: Clone the fin_whale_ssl repository
echo ""
echo "[2/11] Cloning fin_whale_ssl repository..."
cd ~/Desktop
git clone git@github.com:haodongzhang0118/fin_whale_ssl.git

# Step 3: Navigate to project directory
echo ""
echo "[3/11] Navigating to project directory..."
cd fin_whale_ssl

# Step 4: Clone stable-pretraining dependency
echo ""
echo "[4/11] Cloning stable-pretraining repository..."
git clone git@github.com:galilai-group/stable-pretraining.git

# Step 5: Create virtual environment with uv
echo ""
echo "[5/11] Creating virtual environment with uv..."
uv venv

# Step 6: Activate virtual environment
echo ""
echo "[6/11] Activating virtual environment..."
source .venv/bin/activate

# Step 7: Install PyTorch
echo ""
echo "[7/11] Installing PyTorch with uv..."
uv pip install torch torchvision torchaudio

# Step 8: Install project dependencies
echo ""
echo "[8/11] Installing project dependencies with uv..."
uv pip install -e ".[all]"

# Step 9: Install soundfile
echo ""
echo "[9/11] Installing soundfile..."
uv pip install soundfile

# Step 10: Login to WandB
echo ""
echo "[10/11] Setting up WandB login..."
echo "Please login to WandB (you'll need your API key from https://wandb.ai/authorize)"
wandb login

# Step 11: Login to Hugging Face
echo ""
echo "[11/11] Setting up Hugging Face login..."
echo "Please login to Hugging Face (you'll need your token from https://huggingface.co/settings/tokens)"
huggingface-cli login

echo ""
echo "========================================="
echo "✅ Setup completed successfully!"
echo "========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  cd ~/Desktop/fin_whale_ssl"
echo "  source .venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
