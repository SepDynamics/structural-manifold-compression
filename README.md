# Structural Manifold Compression

**Text-only manifold signatures that compress Fox EN/CN and OmniDocBench by 42× on bytes / 85–90× on tokens while preserving ≥ 94.9 % token accuracy, ≤ 5.1 % normalized edit distance, and 80–97 % verification precision with < 0.09 % false-positive rate.** Runs complete in < 1 hour on a single RTX 3080 Ti. Full methodology and results live in [`docs/manifold_vs_optical/report.pdf`](docs/manifold_vs_optical/report.pdf).

---

## 1. Overview & Contributions

- **Sliding-window manifold signatures:** 512 B windows, 384 B stride, quantized coherence/stability/entropy/hazard packed into a 9 B payload (+ repetition count).
- **Perfect-recall hazard gating:** Cross-document verifier reuses the hazard prior to audit collisions; precision hits 91.2 % (Fox EN), 97.2 % (Fox CN), 80.9 % (OmniDoc) with FPR < 0.09 %.
- **End-to-end reproducibility:** `scripts/experiments/benchmark_eval.py` regenerates all CSV/JSON metrics cited in the report; `make report` rebuilds the PDF.
- **Optical baseline harness:** `scripts/experiments/deepseek_ocr_runner.py` replays DeepSeek-OCR on the same manifest for apples-to-apples comparisons.

If you only want the narrative, figures, and tables, read the PDF:  
📄 [`docs/manifold_vs_optical/report.pdf`](docs/manifold_vs_optical/report.pdf)

---

## 2. Benchmark Snapshot (Full Run @ RTX 3080 Ti)

| Dataset | Docs | Byte × | Token × | Token Acc. | Char Acc. | Verif. Precision | Verif. FPR |
|---------|-----:|-------:|--------:|-----------:|----------:|-----------------:|-----------:|
| Fox EN  | 112 | 42.03 | 85.48 | 95.35 % | 95.62 % | 91.21 % | 0.087 % |
| Fox CN  | 100 | 42.01 | 88.08 | 94.94 % | 95.04 % | 97.19 % | 0.029 % |
| OmniDoc | 1 349 | 41.59 | 89.49 | 94.90 % | 94.94 % | 80.85 % | 0.017 % |

Source: [`output/benchmark_runs/full_benchmark/summary.csv`](output/benchmark_runs/full_benchmark/summary.csv)

---

## 3. Repository Layout

```
data/benchmark_corpus/      # Fox / OmniDoc text dumps (symlink; not committed)
docs/
  manifold_vs_optical/
    report.tex              # LaTeX source
    report.pdf              # Ready-to-share manuscript
scripts/
  experiments/
    benchmark_eval.py       # Structural manifold benchmark
    deepseek_ocr_runner.py  # Optical baseline (DeepSeek-OCR)
    plot_manifold_sweep.py  # Curves for the report
  data/
    download_corpora.py     # Fox + OmniDocBench downloader + manifest builder
  training/
    ocr_trainer.py          # LoRA-ready OCR finetuning harness
    run_pipeline.py         # Stageable download/train orchestrator
src/                        # Encoder + manifold helpers
output/                     # Generated summaries/plots
Makefile                    # install | native | full-run | report
docs/03_training_playbook.md# Deployment + training roadmap
```

---

## 4. Setup

```bash
git clone https://github.com/SepDynamics/structural-manifold-compression.git
cd structural-manifold-compression
python3 -m venv .venv && source .venv/bin/activate
make install            # installs Python deps
make native             # optional, builds CUDA kernel if nvcc is present
```

### Dataset & Weights

1. **Fox benchmark** (English + Chinese) text manifests → place under `data/benchmark_corpus/fox/text/{en_page_ocr,cn_page_ocr}`.
2. **OmniDocBench** page-level text → `data/benchmark_corpus/omnidocbench/text`.
3. Keep datasets outside Git; symlink them in if needed: `ln -s /data/share benchmark_corpus/data`.
4. Place the DeepSeek-OCR weights under `external/DeepSeek-OCR/weights` (symlink `external` if you reuse a global models directory).

To pull the corpora automatically (Fox + OmniDoc) and regenerate manifests, run:

```bash
./scripts/training/run_pipeline.py download
# or with filters, e.g.:
# ./scripts/training/run_pipeline.py download --datasets fox
```

The helper wraps `scripts/data/download_corpora.py`, which uses Hugging Face snapshots. Set `HF_TOKEN=...` when downloading from private mirrors.

---

## 5. Reproduce the Structural Benchmark

```bash
python scripts/experiments/benchmark_eval.py \
  --dataset fox=data/benchmark_corpus/fox/text/en_page_ocr \
  --dataset fox_cn=data/benchmark_corpus/fox/text/cn_page_ocr \
  --dataset omnidoc=data/benchmark_corpus/omnidocbench/text \
  --window-bytes 512 --stride-bytes 384 --precision 3 \
  --tokenizer external/DeepSeek-OCR/weights --tokenizer-trust-remote-code \
  --output-dir output/benchmark_runs/full_benchmark
```

Outputs:
- CSV: `output/benchmark_runs/full_benchmark/summary.csv` (table above)
- JSON: dataset- and per-document stats (`fox.json`, `fox_cn.json`, `omnidoc.json`)

### Optional: Optical Baseline (Subset)

```bash
python scripts/experiments/deepseek_ocr_runner.py \
  --dataset fox=data/benchmark_corpus/fox/metadata/text_manifest.jsonl:data/benchmark_corpus/fox/raw \
  --dataset omnidoc=data/benchmark_corpus/omnidocbench/metadata/text_manifest.jsonl:data/benchmark_corpus/omnidocbench/raw/OmniDocBench \
  --prompt "<image>\nFree OCR." \
  --model-name external/DeepSeek-OCR/weights \
  --trust-remote-code --dtype bfloat16 --device cuda --attn-impl eager \
  --max-records 150 \
  --output output/deepseek_runs
```

---

## 6. Rebuild the Report

```bash
make report   # runs pdflatex twice, emits docs/manifold_vs_optical/report.pdf
```

The PDF includes methodology, metric definitions, full benchmark tables, DeepSeek comparison, limitations, and step-by-step reproducibility instructions.

---

## 7. Make Targets

| Target          | Description |
|-----------------|-------------|
| `make install`  | Install Python dependencies into `.venv`. |
| `make native`   | Build the optional CUDA kernel (`scripts/utils/native_kernel.cu`). |
| `make full-run` | Shortcut for the structural benchmark command above. |
| `make report`   | Compile the LaTeX report into `docs/manifold_vs_optical/report.pdf`. |
| `make docker`   | Build a `manifold-compression:latest` image (requires datasets mounted at runtime). |

---

## 8. Citation

```bibtex
@misc{nagy2025manifold,
  author       = {Alexander Nagy},
  title        = {Structural Manifold Compression: A Text-Only Alternative to Optical Context Encoding},
  year         = {2025},
  howpublished = {\url{https://github.com/SepDynamics/structural-manifold-compression}}
}
```

## 9. OCR Fine‑Tuning & Deployment

- Read `docs/03_training_playbook.md` for the end-to-end plan (production integration → language expansion → hardware scaling → robustness).
- Spin up finetuning runs with `scripts/training/ocr_trainer.py`, which consumes the existing Fox/OmniDoc manifests and supports mixed precision, gradient checkpointing, LoRA, and token/character F1 tracking.
- Example (single RTX 3080 Ti, English Fox pages):

  ```bash
  python scripts/training/ocr_trainer.py \
    --train-dataset fox_en=data/benchmark_corpus/fox/metadata/text_manifest.jsonl:data/benchmark_corpus/fox/raw \
    --include-language english \
    --val-split 0.08 \
    --model-id microsoft/trocr-base-printed \
    --output-dir output/training_runs/trocr_fox_en \
    --epochs 3 \
    --train-batch-size 1 \
    --gradient-accumulation 8 \
    --learning-rate 1e-4 \
    --gradient-checkpointing \
    --fp16 \
    --lora-rank 8
  ```

- TensorBoard logs, checkpoints, and eval summaries will land in `output/training_runs/<run-name>`; feed the resulting adapters back into the benchmarking scripts to compare against DeepSeek-OCR and the manifold baselines.

- ### One-Command Pipeline + Resume

- `./scripts/training/run_pipeline.py all` → downloads (if needed), prepares manifests, and launches training with the default Fox+OmniDoc mix.
- `./scripts/training/run_pipeline.py train` → starts (or resumes) training only. Pass `RESUME_FROM=output/training_runs/<run>/checkpoint-XXXX` to pick up after an interruption.
- Tweak env vars instead of editing the script:
  - `RUN_NAME=my_run` (changes output directory).
  - `TRAIN_DATASETS="fox_en=...:...,omnidoc=..."` (choose subsets).
  - `INCLUDE_LANGUAGES=english,chinese` (language filters) or leave unset for all text.
  - `PRECISION=bf16`, `LORA_RANK=4`, `VAL_DATASETS=...` etc.
- Any extra CLI flags appended after the stage name are forwarded straight to `ocr_trainer.py`.

Questions or reproducibility issues? File an issue or ping **@alexandernagy**. Every figure and table is derived directly from the scripts and datasets above. Happy verifying!
