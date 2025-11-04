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
		nvcc -shared -Xcompiler -fPIC -arch=sm_80 scripts/utils/native_kernel.cu -o build/native_kernel.so; \
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
