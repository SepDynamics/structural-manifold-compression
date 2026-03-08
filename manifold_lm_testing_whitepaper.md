# Structural Retrieval and Compression Notes

This document summarizes what the repository currently measures, what it only suggests, and what remains speculative. It follows the evidence ladder in [docs/evidence_ladder.md](docs/evidence_ladder.md).

## 1. Measured Results

Artifacts:

- `results/qa_results.json`
- `results/baseline_suite.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`
- `results/manifold_results_no_sidecar.json`

### Latest large-scale arXiv checkpoint

- corpus: `200` papers
- evaluation set: `250` frozen questions
- protocol: leakage-aware, bounded reconstruction, shuffled-manifold control

Measured outputs:

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

- Structural node retrieval remains stronger than the current chunked RAG baseline at `200` papers.
- Structural node retrieval remains stronger than the current stronger-baseline suite at `200` papers.
- The shuffled control collapses, which supports the integrity of the retrieval result.
- The current best committed manifold result is achieved with sidecar reranking disabled.
- Compression remains weak by the standard required for a strong corpus-compression claim.

## 2. Observed Behavior

- Most remaining manifold misses in the `200`-paper checkpoint are `INSUFFICIENT_CONTEXT`.
- The sidecar-disabled manifold configuration remains better than the current baseline at `200` papers.
- The reconstruction sweep did not find a better setting than `top_k=5`, `per_paper_snippets=3`, and `max_context_tokens=2000`.
- In the retrieval-only suite, BM25 outperformed the plain MiniLM chunk baseline and the unrereanked hybrid, while cross-encoder reranking helped but still remained well below manifold retrieval.

These are meaningful observations, but they are still early large-scale observations rather than robust scaling laws.

## 3. Architecture Hypotheses

The current repository tests the following hypothesis:

Structural document nodes may preserve enough information for strong retrieval under bounded reconstruction, even when the answer stage never sees the full corpus.

The current implementation has four technical layers:

1. Structural encoder
   - measures byte-level structural patterns
2. Structural node index
   - stores heading-aware document segments with short text sketches
3. Sidecar signature layer
   - provides optional secondary overlap features
4. Language-model answer layer
   - answers from reconstructed evidence only

This is a hypothesis about retrieval architecture, not a proved model of language or cognition.

## 4. What Is Not Yet Proven

- strong corpus compression under the arXiv benchmark
- storage reduction relative to the source corpus
- performance beyond the current `200`-paper checkpoint
- parity with a broader external baseline field beyond the current BM25/MiniLM/cross-encoder suite
- a useful role for the sidecar layer in its current form
- any claim of general context-handling superiority over current transformer systems

## 5. Prior Reported Measurements

The repo also contains earlier internal measurements from narrower settings. These are background observations, not proof of the current claim.

Examples include:

- earlier signature-compression tables on OCR-style datasets
- internal latency and verification measurements in manifold-specific experiments
- exploratory dual-stream and Hebbian-learning experiments

These results are useful for hypothesis generation, but the current benchmark claim should stand on the committed arXiv benchmark artifacts instead.

## 6. Research Directions

- compare structural nodes against stronger dense retrieval baselines beyond the current BM25/MiniLM/cross-encoder suite
- redesign the sidecar layer as verifier-only or verifier-first
- improve compression without losing the current retrieval signal
- test whether the current `200`-paper behavior transfers to other public corpora
