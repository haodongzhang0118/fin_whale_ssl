#!/bin/bash

# Fin Whale SSL Setup Script
# This script sets up the environment for the fin_whale_ssl project

set -e  # Exit on error

echo "========================================="
echo "Fin Whale SSL Environment Setup"
echo "========================================="

# Step 1: Install uv
echo ""
echo "[1/10] Installing uv..."
pip install uv

# Step 2: Clone stable-pretraining dependency
echo ""
echo "[2/10] Cloning stable-pretraining repository..."
git clone git@github.com:galilai-group/stable-pretraining.git

# Step 3: Create virtual environment with uv
echo ""
echo "[3/10] Creating virtual environment with uv..."
uv venv

# Step 4: Activate virtual environment
echo ""
echo "[4/10] Activating virtual environment..."
source .venv/bin/activate

# Step 5: Install PyTorch
echo ""
echo "[5/10] Installing PyTorch with uv..."
uv pip install torch torchvision torchaudio

# Step 6: Install project dependencies
echo ""
echo "[6/10] Installing project dependencies with uv..."
cd stable-pretraining
uv pip install -e ".[all]"

# Step 7: Install soundfile
echo ""
echo "[7/10] Installing soundfile..."
uv pip install soundfile

# Step 8: Login to WandB
echo ""
echo "[8/10] Setting up WandB login..."
echo "Please login to WandB (you'll need your API key from https://wandb.ai/authorize)"
wandb login

# Step 9: Login to Hugging Face
echo ""
echo "[9/10] Setting up Hugging Face login..."
echo "Please login to Hugging Face (you'll need your token from https://huggingface.co/settings/tokens)"
huggingface-cli login

# Step 10: Download dataset
echo ""
echo "[10/10] Downloading dataset..."
apt install git-lfs
git lfs install

# Prompt user for HF_TOKEN
echo ""
echo "Please enter your Hugging Face token (get it from https://huggingface.co/settings/tokens):"
read -r HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN cannot be empty. Skipping dataset download."
    echo "You can download it later by running:"
    echo "  export HF_TOKEN='your_token'"
    echo "  git clone https://user:\$HF_TOKEN@huggingface.co/datasets/CIANLabxBROWNUniv/ICML_2026_FIN_HUMPBACK_WHALE"
else
    export HF_TOKEN
    git clone https://user:$HF_TOKEN@huggingface.co/datasets/CIANLabxBROWNUniv/ICML_2026_FIN_HUMPBACK_WHALE
    echo "✅ Dataset downloaded successfully!"
fi

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
