# Dual-Stream Architecture Implementation Notes

This file describes runnable components for the dual-stream experiments. It is an implementation guide, not a proof document. Claims in this file follow the evidence ladder in [evidence_ladder.md](./evidence_ladder.md).

## Current Status

Level 1: measured result

- The repository contains a locked corpus benchmark for structural retrieval.
- The best committed large-scale benchmark result on that benchmark path comes from structural nodes without sidecar reranking on the `200`-paper / `250`-question arXiv checkpoint.
- The current benchmark does not support a strong compression claim.
- The flagship repo claim is retrieval-oriented: structural-node manifold indexing improves document retrieval on leakage-aware corpus benchmarks.

Level 3: hypothesis

- The dual-stream path may still be useful as an experimental framework for studying structural retrieval and bounded-context language-model answers.
- Compression should be treated as a separate unresolved research track rather than as the main repo claim.

## Architecture Overview

The implementation separates four technical concerns:

1. Manifold encoder
   - converts raw bytes into structural signatures
2. Sequence model
   - predicts over structural representations
3. Codebook or retrieval index
   - maps structural states to downstream outputs
4. Inference pipeline
   - coordinates generation, retrieval, and benchmarking

These components are available in the repo, but their combined scientific value still depends on further benchmarking.

## Main Components

1. Manifold encoder
   - [`SMC-Demo/sep_text_manifold/encode.py`](../SMC-Demo/sep_text_manifold/encode.py)
2. Mamba training path
   - [`scripts/training/mamba_ssm_trainer.py`](../scripts/training/mamba_ssm_trainer.py)
3. Dynamic codebook
   - [`scripts/inference/dynamic_codebook.py`](../scripts/inference/dynamic_codebook.py)
4. End-to-end inference
   - [`scripts/inference/dual_stream_inference.py`](../scripts/inference/dual_stream_inference.py)

## Quick Start

### Installation

```bash
cd structural-manifold-compression
pip install -r requirements-mamba.txt
make native
```

### Prepare a manifold dataset

```bash
python scripts/data/prepare_causal_dataset.py \
  --text-root data/raw_math/synthetic_linear_qa.jsonl \
  --output-dir output/manifold_dataset \
  --window-bytes 512 \
  --stride-bytes 384 \
  --precision 3 \
  --use-native \
  --sequence-length 512 \
  --export-signatures
```

### Train the sequence model

```bash
python scripts/training/mamba_ssm_trainer.py \
  --dataset-path output/manifold_dataset/hf_dataset \
  --vocab-path output/manifold_dataset/vocab.json \
  --output-dir output/mamba_checkpoint \
  --d-model 768 \
  --n-layer 16 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --num-epochs 3 \
  --eval-holdout 0.02 \
  --resume
```

### Run inference

```bash
python scripts/inference/dual_stream_inference.py \
  --ssm-checkpoint output/mamba_checkpoint/checkpoint-final \
  --codebook output/codebook.json \
  --vocab output/manifold_dataset/vocab.json \
  --prompt "The quick brown fox jumps over" \
  --max-tokens 50 \
  --temperature 1.0 \
  --output output/generation_result.json
```

## How To Read Metrics In This Folder

Level 2: observed behavior

- Local latency measurements in these scripts can tell you whether the implementation runs and whether scaling looks favorable in the tested setup.
- They do not, by themselves, establish the corpus-retrieval or compression claim in the main benchmark.

Level 4: research direction

- Treat these scripts as experimental infrastructure for future comparisons, not as the main source of repo-level claims.

## Evaluation Targets

Use these scripts to answer questions such as:

- Can the sequence-model path run end to end on structural representations?
- Do local latency and memory trends look favorable in the tested setup?
- Can dynamic-codebook updates be exercised without full retraining?

Do not use this document to claim:

- general architectural superiority over transformer systems
- unbounded scaling claims
- universal zero-shot behavior
- production cost savings without a benchmark artifact

## Troubleshooting

### OOM during training

```bash
--batch-size 2 --gradient-accumulation-steps 16
```

### Slow manifold encoding

```bash
cd SMC-Demo
make build
```

### Low generation quality

- train longer
- increase model size
- improve the training corpus
- tune codebook parameters

## Citation

```bibtex
@misc{nagy2026dualstream,
  author       = {Alexander Nagy},
  title        = {Dual-Stream Architecture: Experimental Implementation Notes},
  year         = {2026},
  howpublished = {\url{https://github.com/SepDynamics/structural-manifold-compression}}
}
```
