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
src/                        # Encoder + manifold helpers
output/                     # Generated summaries/plots
Makefile                    # install | native | full-run | report
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

Questions or reproducibility issues? File an issue or ping **@alexandernagy**. Every figure and table is derived directly from the scripts and datasets above. Happy verifying!
