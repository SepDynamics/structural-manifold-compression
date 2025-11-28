# Manifold Sidecar (RAG Prototype)

Lightweight tooling to build a structural manifold index and hazard-gated verifier for RAG/agent sidecars.

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
