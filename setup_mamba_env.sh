#!/bin/bash
# Setup Mamba SSM environment with Python 3.11 (required for CUDA build)
# Python 3.14 doesn't have prebuilt wheels yet

set -e

echo "=== Setting up Mamba SSM Environment with Python 3.11 ==="
echo ""

# Create separate venv with Python 3.11
if [ -d ".venv_mamba" ]; then
    echo "Removing old Mamba environment..."
    rm -rf .venv_mamba
fi

echo "Creating Python 3.11 virtual environment..."
python3.11 -m venv .venv_mamba

# Activate
source .venv_mamba/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install Mamba SSM and dependencies
echo ""
echo "Installing Mamba SSM (this will compile CUDA kernels)..."
pip install mamba-ssm==1.2.0.post1 causal-conv1d==1.2.0.post2

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install transformers datasets numpy pandas matplotlib tqdm psutil sentence-transformers

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Activate the environment:"
echo "  source .venv_mamba/bin/activate"
echo ""
echo "Test Mamba import:"
echo "  python3 -c 'from mamba_ssm import Mamba; print(\"✓ Mamba SSM ready!\")'"
echo ""
