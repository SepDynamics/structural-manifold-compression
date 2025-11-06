#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CUDA_IMAGE:-nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04}"
DATASET="${DATASET:-wikitext=data/raw_text/wikitext_train.jsonl}"
JSON_KEY="${JSON_KEY:-text}"
WINDOW="${WINDOW:-512}"
STRIDE="${STRIDE:-384}"
PRECISION="${PRECISION:-3}"
OUTDIR="${OUTDIR:-output/benchmark_runs/wikitext_custom}"
SCRIPT="${SCRIPT:-python scripts/experiments/benchmark_eval.py}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[docker] docker CLI not found. Install Docker before running this script." >&2
  exit 1
fi

CMD=$(cat <<EOF
set -euo pipefail
apt-get update >/dev/null && apt-get install -y --no-install-recommends build-essential >/dev/null
make native
CUDA_VISIBLE_DEVICES=0 ${SCRIPT} \
  --dataset ${DATASET} \
  --json-text-key ${JSON_KEY} \
  --window-bytes ${WINDOW} \
  --stride-bytes ${STRIDE} \
  --precision ${PRECISION} \
  --output-dir ${OUTDIR} \
  --use-native
EOF
)

docker run --rm --gpus all -v "${PWD}":/workspace -w /workspace "${IMAGE}" bash -lc "${CMD}"
