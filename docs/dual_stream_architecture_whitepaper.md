# Dual-Stream Architecture: Current Evidence and Open Questions

**Technical note**  
**Version 2.1**  
**Date: March 2026**

This document describes the dual-stream architecture as a research framework. It distinguishes measured benchmark results from observed behavior, hypotheses, and research directions. See [evidence_ladder.md](./evidence_ladder.md) for the repo-wide standard.

## 1. Measured Results

Current claim-bearing artifacts:

- `results/qa_results.json`
- `results/baseline_suite.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`
- `results/manifold_results_no_sidecar.json`

### Latest large-scale arXiv checkpoint

- baseline RAG (`ollama`): `QA=0.504`, `Top-1=0.392`, `Top-5=0.536`
- structural manifold, no sidecar rerank: `QA=0.716`, `Top-1=0.728`, `Top-5=0.828`
- shuffled manifold control: `QA=0.016`, `Top-1=0.008`, `Top-5=0.020`
- compression: about `2.66x` on structural tokens, with serialized manifold bytes larger than the corpus
- retrieval baseline suite on the same locked corpus/questions:
  - dense `all-MiniLM-L6-v2`: `Top-1=0.412`, `Top-5=0.536`
  - BM25: `Top-1=0.480`, `Top-5=0.632`
  - hybrid `MiniLM + BM25`: `Top-1=0.460`, `Top-5=0.624`
  - reranked hybrid `MiniLM + BM25 + cross-encoder`: `Top-1=0.496`, `Top-5=0.648`

Artifact note:

- `results/qa_results.json` is the claim-bearing summary for the flagship `200`-paper `ollama` checkpoint.
- `results/baseline_suite.json` is the retrieval-only stronger-baseline comparison.

### Measured interpretation

- Section-level structural indexing provides a strong retrieval signal through the current `200`-paper checkpoint.
- Section-level structural indexing remains stronger than the current BM25/MiniLM baseline suite.
- The shuffled control materially degrades performance, which supports the retrieval result.
- The current sidecar reranker does not improve the best committed benchmark configuration.
- Compression is not yet strong enough to headline the project.

## 2. Observed Behavior

- The remaining manifold misses are mostly `INSUFFICIENT_CONTEXT`.
- The sidecar-disabled configuration continues to outperform the baseline at `200` papers.
- The best tested reconstruction settings are `top_k=5`, `per_paper_snippets=3`, and `max_context_tokens=2000`.
- In the retrieval-only suite, BM25 is the strongest non-manifold committed baseline, and the reranked hybrid improves on plain MiniLM but not on manifold retrieval.

These are useful engineering observations, but they are still limited to early large-scale checkpoints rather than broad scaling studies.

## 3. Architecture Hypothesis

The dual-stream architecture is a proposed framework for combining:

1. structural encoding
2. structural-node retrieval
3. optional sidecar verification features
4. a bounded-context language-model answer layer

The working hypothesis is that this separation may preserve useful retrieval structure without requiring the answer stage to consume the entire source corpus.

This should be read as a research hypothesis, not as a proved replacement for token-based language models.

## 4. Prior Reported Measurements

The repository also contains earlier internal measurements on narrower tasks:

- dual-stream latency experiments
- manifold verification experiments
- local Hebbian-learning experiments
- zero-shot codebook and injection experiments

These measurements are relevant background, but they do not establish the current arXiv benchmark claim.

## 5. Current Limitations

- The current benchmark claim rests primarily on a single `200`-paper / `250`-question checkpoint, with earlier `25`- and `50`-paper runs serving as development stages.
- Compression is weak in the committed pilot artifacts.
- Serialized manifold storage is still larger than the source corpus bytes.
- The stronger-baseline comparison is still narrow: BM25, MiniLM chunk retrieval, and one cross-encoder reranked hybrid.
- The current sidecar layer needs redesign before it can be presented as a helpful retrieval component.

## 6. Research Directions

- compare against stronger dense retrieval baselines beyond the current BM25/MiniLM suite
- validate the current retrieval result on broader question types and additional corpora
- redesign the sidecar layer as verifier-first rather than reranker-first
- test whether compression can improve without losing the current retrieval signal
- evaluate transfer to other public corpora after the arXiv path is stable
