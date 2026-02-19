#!/bin/bash
# Quick demo of dual-stream architecture using existing sample data
# This runs in < 5 minutes to prove the concept

set -e

echo "=== Dual-Stream Architecture Quick Demo ==="
echo ""

# Auto-detect compatible GCC for CUDA compilation
if command -v gcc-11 &> /dev/null; then
    export CC=gcc-11
    export CXX=g++-11
    echo "✓ Using GCC 11 for CUDA compatibility"
elif command -v gcc-14 &> /dev/null; then
    export CC=gcc-14
    export CXX=g++-14
    echo "✓ Using GCC 14 for CUDA compatibility"
else
    echo "⚠ Warning: No compatible GCC found. CUDA compilation may fail."
    echo "  Please install gcc-14 or gcc-11"
fi

# Set optimal CUDA arch for RTX 3080 Ti
export TORCH_CUDA_ARCH_LIST="8.6"

# cd structural-manifold-compression

# Check dependencies
echo "Step 1: Checking dependencies..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
fi

if ! python3 -c "import mamba_ssm" 2>/dev/null; then
    echo "Installing Mamba SSM..."
    pip install mamba-ssm causal-conv1d
fi

echo "✓ Dependencies ready"
echo ""

# Use existing sample data
echo "Step 2: Using existing sample corpus..."
SAMPLE_DATA="data/sample_corpus.jsonl"
if [ ! -f "$SAMPLE_DATA" ]; then
    echo "Creating minimal sample corpus..."
    cat > $SAMPLE_DATA << 'EOF'
{"text": "The quick brown fox jumps over the lazy dog.", "doc_id": "doc1"}
{"text": "Machine learning models process data efficiently.", "doc_id": "doc2"}
{"text": "Quantum mechanics describes atomic behavior.", "doc_id": "doc3"}
EOF
fi
echo "✓ Sample data ready"
echo ""

# Step 3: Prepare tiny manifold dataset (< 1 min)
echo "Step 3: Preparing manifold dataset (< 1 min)..."
python3 scripts/data/prepare_causal_dataset.py \
  --text-root $SAMPLE_DATA \
  --json-text-key text \
  --output-dir output/quick_demo/manifold_dataset \
  --window-bytes 128 \
  --stride-bytes 96 \
  --precision 2 \
  --sequence-length 64 \
  --min-sequence-length 4 \
  --export-signatures \
  --concat-documents \
  --reset-output

echo "✓ Manifold dataset ready"
echo ""

# Step 4: Train tiny SSM (< 2 min on GPU)
echo "Step 4: Training tiny Mamba SSM (< 2 min on GPU)..."
python3 scripts/training/mamba_ssm_trainer.py \
  --dataset-path output/quick_demo/manifold_dataset/hf_dataset \
  --vocab-path output/quick_demo/manifold_dataset/vocab.json \
  --output-dir output/quick_demo/mamba_checkpoint \
  --d-model 256 \
  --n-layer 4 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-3 \
  --num-epochs 1 \
  --eval-holdout 0.1

echo "✓ SSM trained"
echo ""

# Step 5: Build simple codebook
echo "Step 5: Building codebook..."
# For quick demo, we'll use a synthetic codebook
python3 << 'PYEOF'
import json
import sys
from pathlib import Path

# Simple synthetic codebook for demo
codebook = {
    "window_size": 128,
    "decay_factor": 0.95,
    "global_position": 100,
    "entries": {
        "c0.8_s0.2_e0.4": {
            "signature": "c0.8_s0.2_e0.4",
            "tokens": ["the", "quick", "machine"],
            "positions": [0, 10, 20],
            "frequency": 5,
            "last_seen": 50
        },
        "c0.6_s0.4_e0.5": {
            "signature": "c0.6_s0.4_e0.5",
            "tokens": ["quantum", "learning", "fox"],
            "positions": [5, 15, 25],
            "frequency": 3,
            "last_seen": 75
        },
        "c0.9_s0.1_e0.3": {
            "signature": "c0.9_s0.1_e0.3",
            "tokens": ["model", "mechanics", "brown"],
            "positions": [8, 18, 28],
            "frequency": 4,
            "last_seen": 80
        }
    },
    "spatial_index": {
        "c0.8_s0.2_e0.4": ["c0.6_s0.4_e0.5", "c0.9_s0.1_e0.3"],
        "c0.6_s0.4_e0.5": ["c0.8_s0.2_e0.4", "c0.9_s0.1_e0.3"],
        "c0.9_s0.1_e0.3": ["c0.8_s0.2_e0.4", "c0.6_s0.4_e0.5"]
    }
}

output_path = Path("output/quick_demo/codebook.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(codebook, f, indent=2)

print("✓ Codebook created")
PYEOF

echo ""

# Step 6: Run inference
echo "Step 6: Running inference..."
python3 scripts/inference/dual_stream_inference.py \
  --ssm-checkpoint output/quick_demo/mamba_checkpoint/checkpoint-1000 \
  --codebook output/quick_demo/codebook.json \
  --vocab output/quick_demo/manifold_dataset/vocab.json \
  --prompt "The quantum" \
  --max-tokens 20 \
  --output output/quick_demo/inference_result.json \
  --device cuda 2>/dev/null || \
python3 scripts/inference/dual_stream_inference.py \
  --ssm-checkpoint output/quick_demo/mamba_checkpoint/checkpoint-1000 \
  --codebook output/quick_demo/codebook.json \
  --vocab output/quick_demo/manifold_dataset/vocab.json \
  --prompt "The quantum" \
  --max-tokens 20 \
  --output output/quick_demo/inference_result.json \
  --device cpu

echo ""
echo "=== Demo Complete! ==="
echo ""
echo "Results saved to: output/quick_demo/"
echo ""
echo "Next steps:"
echo "  1. View inference results: cat output/quick_demo/inference_result.json"
echo "  2. Run full training (see COMMANDS_TO_RUN.md)"
echo "  3. Run benchmarks to get preliminary findings"
echo ""
