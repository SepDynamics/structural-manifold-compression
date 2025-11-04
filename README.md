# Structural Manifold Compression

**44–46× byte / 85–99× token compression** on the Fox EN/CN and OmniDocBench benchmarks with **87–92 % token accuracy**, **7–11 % normalized edit distance**, and **perfect-recall hazard gating** (3–17 % precision at 2–5 % FPR) – all in under one hour on a single RTX 3080 Ti.

---

## Quickstart

```bash
git clone git@github.com:scrallex/structural-manifold-compression.git
cd structural-manifold-compression
make install            # installs Python deps
make native             # optional CUDA kernel (skips if nvcc missing)
make full-run           # reproduces Table 1 (≈45 min on 3080 Ti)
```

To use Docker instead of a local toolchain:

```bash
docker build -t manifold-compression .
docker run --rm -v $PWD/data:/app/data manifold-compression \
    python scripts/experiments/benchmark_eval.py \
        --dataset fox=data/benchmark_corpus/fox/text/en_page_ocr \
        --dataset fox_cn=data/benchmark_corpus/fox/text/cn_page_ocr \
        --dataset omnidoc=data/benchmark_corpus/omnidocbench/text \
        --window-bytes 512 --stride-bytes 384 --precision 3 \
        --tokenizer external/DeepSeek-OCR/weights --tokenizer-trust-remote-code \
        --output-dir output/benchmark_runs/full_benchmark
```

Data folders (`data/benchmark_corpus/*`) are expected to contain the Fox and OmniDocBench text dumps produced by `scripts/experiments/prepare_benchmarks.py`; keep them out of Git. If you keep the corpora in a shared location, create a symlink inside this repository (e.g., `ln -s /path/to/data data`). The DeepSeek-OCR weights live under `external/DeepSeek-OCR/weights` (symlink to a shared `external/` directory if needed).

---

## Results (full benchmark run)

These numbers come directly from `output/benchmark_runs/full_benchmark/summary.csv`.

| Dataset | Docs | Byte × | Token × | Token Acc. | Char Acc. | Verif. Precision | Verif. FPR |
|---------|-----:|-------:|--------:|-----------:|----------:|-----------------:|-----------:|
| Fox EN (112)  | 112 | 44.29 | 85.48 | 91.67 % | 92.75 % | 16.44 % | 4.58 % |
| Fox CN (100)  | 100 | 44.68 | 88.08 | 90.55 % | 90.85 % | 16.60 % | 5.08 % |
| OmniDocBench (1349) | 1 349 | 45.93 | 89.49 | 87.16 % | 89.41 % | 3.36 % | 2.12 % |

*Byte × = original UTF-8 bytes / compressed signature bytes. Token × = GPT-style text tokens / unique manifold signatures.*

The same hardware budget produces DeepSeek-OCR outputs in **hours** with **≤ 10×** effective compression and **no verification signal**. Structural manifolds keep fidelity competitive while delivering 85–99× token reduction, and the hazard-gating verifier supplies perfect recall with tunable precision for downstream audits.

---

## Repository Layout

```
data/benchmark_corpus/      # symlinks to Fox/OmniDocBench (not committed)
docs/
  manifold_vs_optical/
    report.tex              # arXiv-ready paper
scripts/
  experiments/
    benchmark_eval.py       # structural evaluation entrypoint
    deepseek_ocr_runner.py  # optical baseline harness
    plot_curves.py          # summary → figures
  utils/
    native_kernel.cu        # optional CUDA kernel
src/
  manifold/                 # python helpers (encoder, verifier)
tests/
  test_compression.py       # smoke test on sample data
.github/workflows/ci.yml    # lint/tests + Docker build
Dockerfile                  # ghcr.io build recipe
Makefile                    # install | native | full-run | report
output/                     # generated summaries + plots
```

---

## Make Targets

| Target        | Description |
|---------------|-------------|
| `make install` | Install Python dependencies into the active environment. |
| `make native` | Compile `scripts/utils/native_kernel.cu` into `build/native_kernel.so` (skips politely if `nvcc` is unavailable). |
| `make full-run` | Run `scripts/experiments/benchmark_eval.py` on Fox EN/CN and OmniDocBench, writing outputs to `output/benchmark_runs/full_benchmark/`. |
| `make report` | Build `docs/manifold_vs_optical/report.tex` into `docs/manifold_vs_optical/manifold_vs_optical.pdf`. |
| `make docker` | Build the Docker image (`manifold-compression:latest`). |

---

## Reproducing the Report

1. **Acquire datasets** (Fox benchmark + OmniDocBench) and extract text into `data/benchmark_corpus/*` using `scripts/experiments/prepare_benchmarks.py`.
2. **Run** `make full-run` to regenerate `summary.csv` / `summary.json`.
3. **Plot** curves for the paper:
   ```bash
   python scripts/experiments/plot_curves.py \
       --summary output/benchmark_runs/full_benchmark/summary.json \
       --output docs/manifold_vs_optical/figures/compression.png
   ```
4. **Compile** the PDF: `make report`.

---

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) performs:

1. Dependency installation inside `nvidia/cuda:12.4.1` container.
2. Optional native kernel build (`make native`).
3. Unit tests (`pytest`).
4. Docker image build + push to `ghcr.io` when pushing to `main`.

---

## Citation

```bibtex
@misc{scrallex2025manifold,
  author       = {Scrallex},
  title        = {Structural Manifold Compression: A Text-Only Alternative to Optical Context Encoding},
  year         = {2025},
  howpublished = {\url{https://github.com/scrallex/structural-manifold-compression}}
}
```

---

Questions or reproducibility issues? Open an issue or ping @scrallex. The entire training + benchmarking surface fits in this repository – no external notebooks required.
