# Dual-Stream Architecture: Current Evidence and Open Questions

**Technical note**  
**Version 2.1**  
**Date: March 2026**

This document describes the dual-stream architecture as a research framework. It distinguishes measured benchmark results from observed behavior, hypotheses, and research directions. See [evidence_ladder.md](./evidence_ladder.md) for the repo-wide standard.

## 1. Measured Results

Current claim-bearing artifacts:

- `results/qa_results.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`
- `results/manifold_results_no_sidecar.json`

### Latest mid-scale arXiv checkpoint

- baseline RAG (`ollama`): `QA=0.750`, `Top-1=0.617`, `Top-5=0.767`
- structural manifold, no sidecar rerank: `QA=0.883`, `Top-1=0.867`, `Top-5=0.950`
- shuffled manifold control: `QA=0.050`, `Top-1=0.000`, `Top-5=0.050`
- compression: about `2.24x` on structural tokens, with serialized manifold bytes larger than the corpus

### Measured interpretation

- Section-level structural indexing provides a strong retrieval signal through the current `50`-paper checkpoint.
- The shuffled control materially degrades performance, which supports the retrieval result.
- The current sidecar reranker does not improve the best committed benchmark configuration.
- Compression is not yet strong enough to headline the project.

## 2. Observed Behavior

- The remaining manifold misses are mostly `INSUFFICIENT_CONTEXT`.
- The sidecar-disabled configuration continues to outperform the baseline at `50` papers.
- The best tested reconstruction settings are `top_k=5`, `per_paper_snippets=3`, and `max_context_tokens=2000`.

These are useful engineering observations, but they are still limited to early mid-scale checkpoints rather than broad scaling studies.

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

- The current checkpoints are still modest in scale: `25` papers / `40` questions and `50` papers / `60` questions.
- Compression is weak in the committed pilot artifacts.
- Serialized manifold storage is still larger than the source corpus bytes.
- The baseline comparison is a simple chunked RAG pipeline, not a broad external benchmark suite.
- The current sidecar layer needs redesign before it can be presented as a helpful retrieval component.

## 6. Research Directions

- validate the benchmark at `100` and `200` papers
- compare against stronger embedding and hybrid retrieval baselines
- redesign the sidecar layer as verifier-first rather than reranker-first
- test whether compression can improve without losing the current retrieval signal
- evaluate transfer to other public corpora after the arXiv path is stable
