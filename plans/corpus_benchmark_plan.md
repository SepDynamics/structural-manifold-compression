# Corpus Benchmark Plan

## Objective
Run the leakage-aware benchmark in stages, keeping claims tied to committed artifacts as the corpus scales.

## Current checkpoint

The latest completed checkpoints are:

- `25` arXiv papers
- `40` frozen questions
- both `extractive` and `ollama` answer backends on the locked pilot
- a follow-on `50`-paper / `60`-question Ollama checkpoint with sidecar reranking disabled
- a `200`-paper / `250`-question Ollama checkpoint with sidecar reranking disabled

Observed results:

- extractive baseline RAG: `QA=0.650`, `Top-1=0.650`, `Top-5=0.875`
- extractive structural manifold: `QA=0.900`, `Top-1=0.900`, `Top-5=0.950`
- extractive shuffled manifold: `QA=0.025`, `Top-1=0.025`, `Top-5=0.050`
- ollama baseline RAG: `QA=0.775`, `Top-1=0.650`, `Top-5=0.875`
- ollama structural manifold: `QA=0.825`, `Top-1=0.900`, `Top-5=0.950`
- ollama shuffled manifold: `QA=0.050`, `Top-1=0.025`, `Top-5=0.050`
- 50-paper ollama baseline RAG: `QA=0.750`, `Top-1=0.617`, `Top-5=0.767`
- 50-paper ollama structural manifold (no sidecar rerank): `QA=0.883`, `Top-1=0.867`, `Top-5=0.950`
- 50-paper ollama shuffled manifold: `QA=0.050`, `Top-1=0.000`, `Top-5=0.050`
- compression at 50 papers: `2.24x` on structural tokens, with serialized manifold bytes still larger than the corpus
- 200-paper ollama baseline RAG: `QA=0.504`, `Top-1=0.392`, `Top-5=0.536`
- 200-paper ollama structural manifold (no sidecar rerank): `QA=0.716`, `Top-1=0.728`, `Top-5=0.828`
- 200-paper ollama shuffled manifold: `QA=0.016`, `Top-1=0.008`, `Top-5=0.020`
- compression at 200 papers: `2.66x` on structural tokens, with serialized manifold bytes still larger than the corpus

Interpretation:

- structural retrieval now looks legitimate through the `200`-paper checkpoint
- the shuffled control is behaving correctly
- answer-path tightening worked materially on the same locked corpus/questions
- the manifold still beats baseline under LLM answering at `200` papers
- the retrieval-to-QA gap is now small at `200` papers (`Top-1=0.728`, `QA=0.716`)
- the compression claim is still weak
- the next highest-value steps are stronger baseline comparisons, harder question sets, and compression redesign, not more prompt cleanup

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
  --node-chars 1500 \
  --node-overlap 180 \
  --window-bytes 16 \
  --stride-bytes 4 \
  --qa-backend extractive \
  --force
```

Purpose:
- check PDF extraction quality
- inspect `data/questions.json`
- inspect whether retrieved structural nodes are sensible
- verify shuffled-manifold accuracy collapses

Exit criteria:
- at least 80% of sampled papers have usable extracted text
- frozen questions are document-specific rather than generic
- shuffled-manifold retrieval materially degrades

### Stage 1b: locked 25-paper Ollama run
Do this on the same corpus and frozen questions after Stage 1 looks clean.

Command:
```bash
python run_full_demo.py \
  --paper-count 25 \
  --question-count 40 \
  --categories cs.LG cs.AI math.OC math.PR hep-th cond-mat.stat-mech \
  --node-chars 1500 \
  --node-overlap 180 \
  --window-bytes 16 \
  --stride-bytes 4 \
  --qa-backend ollama
```

Purpose:
- test whether the manifold still wins when answers must be generated from reconstructed context
- measure how much of the pilot win survives beyond title-level extractive retrieval

Exit criteria:
- manifold remains materially above shuffled
- manifold stays competitive with or better than baseline on the locked question set
- answer quality still tracks retrieved evidence rather than generic prior knowledge

Status:
- Completed.
- Result: after answer-path tightening on the same locked corpus/questions, manifold reached `0.825` versus baseline `0.775` and shuffled `0.050`.

### Stage 1c: locked 25-paper answer-path tightening
Do this before scaling corpus size.

Purpose:
- separate retrieval quality from answer-path quality
- reduce answer losses caused by prompt format, `INSUFFICIENT_CONTEXT`, or abbreviated model outputs

Concrete work items:
- inspect failed Ollama answers where `retrieval_top1=True` but `correct=False`
- tighten scoring aliases and answer prompt formatting
- keep the corpus and frozen questions unchanged during this phase

Status:
- Completed.
- Result: the locked pilot improved from `0.700` to `0.825` manifold QA and from `0.625` to `0.775` baseline QA. Remaining misses are now mostly true `INSUFFICIENT_CONTEXT` outputs rather than formatting noise.

### Stage 1d: locked 25-paper ablation / reconstruction tuning
Do this before scaling corpus size.

Purpose:
- measure whether sidecar reranking is actually buying useful accuracy
- see whether the remaining `INSUFFICIENT_CONTEXT` misses are caused by too little or too noisy reconstruction
- preserve the same frozen corpus/questions so the comparison stays interpretable

Concrete work items:
- compare structural nodes with and without sidecar reranking
- vary `top-k`, per-paper snippet count, and `max_context_tokens`
- inspect whether extra context closes the remaining manifold `0.90 -> 0.825` gap
- keep the corpus and frozen questions unchanged during this phase

Status:
- Completed.
- Sidecar ablation result: with sidecar `QA=0.850`; without sidecar `QA=0.900`; Top-1 retrieval remained `0.900` in both arms.
- Reconstruction sweep result: best tested configuration was already the current default `top_k=5`, `per_paper_snippets=3`, `max_context_tokens=2000`, with sidecar reranking disabled.
- Best follow-on locked-pilot result: `QA=0.900`, `Top-1=0.900`, `Top-5=0.975`, `4` `INSUFFICIENT_CONTEXT` answers.

### Stage 2: arXiv mid-scale run (50-100 papers)
Increase only after Stage 1d improves or at least explains the remaining Ollama gap.

Suggested commands:
```bash
python run_full_demo.py --paper-count 50 --question-count 60 --qa-backend ollama
python run_full_demo.py --paper-count 100 --question-count 80 --qa-backend ollama
```

Purpose:
- estimate runtime and storage growth
- see whether manifold retrieval stays above chance at larger scale
- see whether QA stays within striking distance of baseline RAG
- see whether compression improves or remains too weak to support the flagship claim

Recommended next checkpoint:
```bash
python run_full_demo.py \
  --paper-count 50 \
  --question-count 60 \
  --categories cs.LG cs.AI math.OC math.PR hep-th cond-mat.stat-mech \
  --node-chars 1500 \
  --node-overlap 180 \
  --window-bytes 16 \
  --stride-bytes 4 \
  --qa-backend ollama \
  --disable-sidecar-rerank \
  --force
```

Status:
- Partially completed.
- `50`-paper result: baseline `QA=0.750`, manifold `QA=0.883`, shuffled `QA=0.050`.
- Retrieval also remained strong at `50` papers: baseline `Top-1=0.617`, manifold `Top-1=0.867`, shuffled `Top-1=0.000`.
- Compression improved only modestly to `2.24x` and remains the weak part of the story.

Metrics that matter:
- `results/compression_metrics.json`
- `results/baseline_rag_results.json`
- `results/manifold_results.json`
- `results/manifold_results_no_sidecar.json`
- `results/manifold_results_shuffled.json`
- `results/qa_results.json`

### Stage 3: flagship arXiv run (200 papers)
Only run this after the 50-100 paper stage looks legitimate.

Suggested command:
```bash
python run_full_demo.py \
  --paper-count 200 \
  --question-count 250 \
  --categories cs.LG cs.AI math.OC math.PR hep-th cond-mat.stat-mech \
  --node-chars 1500 \
  --node-overlap 180 \
  --window-bytes 16 \
  --stride-bytes 4 \
  --qa-backend ollama \
  --disable-sidecar-rerank \
  --force
```

Status:
- Completed.
- `200`-paper result: baseline `QA=0.504`, manifold `QA=0.716`, shuffled `QA=0.016`.
- Retrieval remained materially stronger for the manifold at `200` papers: baseline `Top-1=0.392`, manifold `Top-1=0.728`, shuffled `Top-1=0.008`.
- Compression improved only modestly to `2.66x`, and serialized manifold storage remains larger than the raw corpus bytes.
- Mean latency was slightly better for the manifold path in this run (`0.397s` vs `0.452s`).

This is the first large-scale run worth presenting externally, but it supports a retrieval claim rather than a strong compression claim.

### Stage 4: post-200 validation
Do this after the flagship run, using the same evidence discipline.

Purpose:
- determine whether the current win survives stronger baselines
- measure whether the result survives harder question types
- separate retrieval quality from compression quality in follow-on work

Recommended next work:
1. compare against stronger embedding and hybrid retrieval baselines
2. add harder cross-document and structural questions to reduce title-lookup bias
3. redesign the compression format so serialized artifacts are smaller than the source corpus
4. only then test transfer to a second public corpus

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
- compression remains too weak for the claim being made

## Practical advice

- Start with `--qa-backend extractive` if you only want to debug retrieval.
- Switch to `--qa-backend ollama` once retrieval looks sane, while keeping the same frozen corpus/questions.
- Use the `25`- and `50`-paper stages to debug, but treat the `200`-paper result as the current external checkpoint.
- Freeze `data/questions.json` and keep it untouched once the run starts.
