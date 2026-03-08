# Corpus Benchmark Plan

## Objective
Run the new leakage-aware benchmark in stages until the 200-paper result is credible enough to cite.

## Recommended path

### Stage 0: Local smoke corpus
Use 3-10 local `.txt` files only to verify wiring.

Command:
```bash
python run_full_demo.py \
  --input-dir /path/to/local/texts \
  --qa-backend extractive \
  --embedding-model hash \
  --force
```

Purpose:
- confirm the pipeline runs
- confirm artifacts land in `data/`, `manifold/`, and `results/`
- do not interpret these numbers scientifically

### Stage 1: arXiv pilot (25 papers)
Use the built-in arXiv downloader.

Command:
```bash
python run_full_demo.py \
  --paper-count 25 \
  --question-count 40 \
  --categories cs.LG cs.AI math.OC math.PR hep-th cond-mat.stat-mech \
  --qa-backend ollama \
  --force
```

Purpose:
- check PDF extraction quality
- inspect `data/questions.json`
- inspect whether retrieved chunks are sensible
- verify shuffled-manifold accuracy collapses

Exit criteria:
- at least 80% of sampled papers have usable extracted text
- frozen questions are document-specific rather than generic
- shuffled-manifold retrieval materially degrades

### Stage 2: arXiv mid-scale run (50-100 papers)
Increase only after Stage 1 artifacts look clean.

Suggested commands:
```bash
python run_full_demo.py --paper-count 50 --question-count 60 --qa-backend ollama --force
python run_full_demo.py --paper-count 100 --question-count 80 --qa-backend ollama --force
```

Purpose:
- estimate runtime and storage growth
- see whether manifold retrieval stays above chance
- see whether QA stays within striking distance of baseline RAG

Metrics that matter:
- `results/compression_metrics.json`
- `results/baseline_rag_results.json`
- `results/manifold_results.json`
- `results/manifold_results_shuffled.json`
- `results/qa_results.json`

### Stage 3: flagship arXiv run (200 papers)
Only run this after the 50-100 paper stage looks legitimate.

Suggested command:
```bash
python run_full_demo.py \
  --paper-count 200 \
  --question-count 100 \
  --categories cs.LG cs.AI math.OC math.PR hep-th cond-mat.stat-mech \
  --qa-backend ollama \
  --force
```

This is the first run worth presenting externally.

## What corpus to obtain

### For the flagship claim
Use arXiv only. You do not need a second corpus before this one is finished.

Reasons:
- public and easy to re-download
- technically dense
- structurally rich
- hard to accuse of cherry-picking
- already supported by the new demo scripts

### After the flagship run
Only add follow-on corpora if the arXiv result is promising.

Recommended order:
1. Public code/docs corpus
2. Legal opinions or SEC filings
3. Long-form logs or incident reports

The point is to test transfer after the core claim works once, not before.

## Manual review checklist

Before trusting a run, inspect:
- 10 random files in `data/corpus/`
- 10 random entries in `data/questions.json`
- 10 baseline retrieval results
- 10 manifold retrieval results
- shuffled-manifold degradation

Reject the run if:
- extracted PDFs are mostly broken
- questions can be answered from generic world knowledge
- titles or ids leak the answer path
- manifold retrieval remains strong after shuffling

## Practical advice

- Start with `--qa-backend extractive` if you only want to debug retrieval.
- Switch to `--qa-backend ollama` once retrieval looks sane.
- Do not jump straight to 200 papers; most failure modes show up at 25-50.
- Freeze `data/questions.json` and keep it untouched once the run starts.
