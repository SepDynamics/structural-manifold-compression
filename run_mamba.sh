#!/bin/bash
source /sep/structural-manifold-compression/.venv/bin/activate
pip install mamba-ssm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ensure fp16 or bf16 is used, reduce batch size if needed for 12GB VRAM
# A quick parameter adjustment per the "If training fails with OOM" guide:
python3 scripts/training/mamba_ssm_trainer.py \
  --dataset-path output/production/manifold_dataset/hf_dataset \
  --vocab-path output/production/manifold_dataset/vocab.json \
  --output-dir output/production/mamba_checkpoint \
  --d-model 768 \
  --n-layer 16 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --num-epochs 3 \
  --eval-holdout 0.02 \
  --checkpoint-every 1000 \
  --resume
