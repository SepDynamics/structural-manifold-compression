.PHONY: install native full-run report docker test

PYTHON ?= python
FOX_TEXT ?= data/benchmark_corpus/fox/text/en_page_ocr
FOX_CN_TEXT ?= data/benchmark_corpus/fox/text/cn_page_ocr
OMNIDOC_TEXT ?= data/benchmark_corpus/omnidocbench/text
OUTPUT_DIR ?= output/benchmark_runs/full_benchmark
TOKENIZER ?= external/DeepSeek-OCR/weights

install:
	pip install --no-cache-dir -r requirements.txt

native:
	@echo "[native] building optional CUDA kernel"
	@mkdir -p build
	@if command -v nvcc >/dev/null 2>&1; then \
		nvcc -allow-unsupported-compiler -ccbin g++-14 -shared -Xcompiler -fPIC -arch=sm_80 scripts/utils/native_kernel.cu -o build/native_kernel.so; \
	else \
		echo "nvcc not found; skipping native build"; \
	fi

full-run:
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) scripts/experiments/benchmark_eval.py \
		--dataset fox=$(FOX_TEXT) \
		--dataset fox_cn=$(FOX_CN_TEXT) \
		--dataset omnidoc=$(OMNIDOC_TEXT) \
		--window-bytes 512 \
		--stride-bytes 384 \
		--precision 3 \
		--tokenizer $(TOKENIZER) \
		--tokenizer-trust-remote-code \
		--output-dir $(OUTPUT_DIR) \
		--use-native

report:
	cd docs/manifold_vs_optical && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex

plot:
	$(PYTHON) scripts/experiments/plot_curves.py \
		--summary $(OUTPUT_DIR)/summary.json \
		--output docs/manifold_vs_optical/figures/compression.png

docker:
	docker build -t manifold-compression .

test:
	pytest -v

MANIFOLD_RUN_DIR ?= output/training_runs/wikitext_manifold_gpt
MANIFOLD_DATASET ?= output/wikitext_manifold/hf_dataset
MANIFOLD_VOCAB ?= output/wikitext_manifold/vocab.json
MANIFOLD_PY ?= .venv/bin/python
CUDA ?= 0

train-manifold-gpt:
	@echo "[train] launching manifold LM on GPU $(CUDA)"
	CUDA_VISIBLE_DEVICES=$(CUDA) $(MANIFOLD_PY) scripts/training/manifold_lm_trainer.py \
		--dataset-path $(MANIFOLD_DATASET) \
		--vocab-path $(MANIFOLD_VOCAB) \
		--output-dir $(MANIFOLD_RUN_DIR) \
		--n-layer 16 --n-head 16 --n-embd 1024 \
		--context-length 512 \
		--per-device-train-batch-size 2 \
		--per-device-eval-batch-size 2 \
		--gradient-accumulation-steps 16 \
		--learning-rate 2e-4 \
		--num-train-epochs 3 \
		--warmup-steps 500 \
		--eval-holdout 0.02 \
		--gradient-checkpointing \
		--fp16 \
		--resume
