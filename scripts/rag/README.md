# Manifold Sidecar (RAG Prototype)

Lightweight tooling to build a structural manifold index and hazard-gated verifier for RAG/agent sidecars.

## Prepare a corpus

```bash
python scripts/rag/prepare_corpus.py \
  --input-dir data/raw_docs \
  --output-jsonl data/corpus.jsonl
```

- Walks `input-dir`, ingests `.txt`, `.md`, and `.pdf` (page-by-page), and emits JSONL rows of `{"doc_id": "...", "text": "..."}`.
- Document IDs are relative paths (PDFs append `#page=N`).

## Build an index

```bash
python scripts/rag/build_manifold_index.py \
  --dataset examples/structured_demo/news_sample.jsonl \
  --json-text-key text \
  --window-bytes 512 --stride-bytes 384 --precision 3 \
  --use-native \
  --output output/manifold_index/news_demo.json
```

- Accepts any JSONL/JSON/txt corpus; `--json-text-key` picks the field to read.
- Use `--omit-windows` to drop per-document window streams if you only need signature→occurrence lookup.
- The index stores prototype spans, occurrences (doc_id + offsets + hazard), and an 80th-percentile hazard gate.

End-to-end with your own files:

```bash
python scripts/rag/prepare_corpus.py --input-dir data/raw_docs --output-jsonl data/corpus.jsonl
python scripts/rag/build_manifold_index.py --dataset data/corpus.jsonl --output output/manifold_index/corpus.json
```

## Verify a snippet

```bash
python scripts/rag/verify_snippet.py \
  --index output/manifold_index/news_demo.json \
  --text "Example paragraph to verify" \
  --coverage-threshold 0.5 \
  --reconstruct
```

- Returns coverage (low-hazard matches / windows), matched documents, and per-window occurrences.
- `--reconstruct` rebuilds the snippet using only prototype spans from the index (no raw text dependency).

## Demo: naive vs manifold-verified RAG

```bash
python scripts/rag/demo_rag.py \
  --dataset data/corpus.jsonl \
  --index output/manifold_index/corpus.json \
  --question "What does the Q3 risk section say about liquidity?" \
  --top-k 5
```

- Builds embeddings over chunked passages, prints naive top-k retrieval, then filters with `verify_snippet` to show only hazard-gated matches.

## One-step demo corpus

```bash
make demo-corpus
```

- Prepares `data/sample_docs` → `data/sample_corpus.jsonl`
- Builds `output/manifold_index/sample_corpus.json`
- Runs the CLI RAG demo with a sample liquidity question
