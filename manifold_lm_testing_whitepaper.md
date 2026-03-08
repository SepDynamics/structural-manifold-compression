# Structural Retrieval and Compression Notes

This document summarizes what the repository currently measures, what it only suggests, and what remains speculative. It follows the evidence ladder in [docs/evidence_ladder.md](docs/evidence_ladder.md).

## 1. Measured Results

Artifacts:

- `results/qa_results.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`
- `results/manifold_results_no_sidecar.json`

### Latest mid-scale arXiv checkpoint

- corpus: `50` papers
- evaluation set: `60` frozen questions
- protocol: leakage-aware, bounded reconstruction, shuffled-manifold control

Measured outputs:

- baseline RAG (`ollama`): `QA=0.750`, `Top-1=0.617`, `Top-5=0.767`
- structural manifold, no sidecar rerank: `QA=0.883`, `Top-1=0.867`, `Top-5=0.950`
- shuffled manifold control: `QA=0.050`, `Top-1=0.000`, `Top-5=0.050`
- compression: about `2.24x` on structural tokens, with serialized manifold bytes larger than the corpus

### Measured interpretation

- Structural node retrieval remains stronger than the current chunked RAG baseline at `50` papers.
- The shuffled control collapses, which supports the integrity of the retrieval result.
- The current best committed manifold result is achieved with sidecar reranking disabled.
- Compression remains weak by the standard required for a strong corpus-compression claim.

## 2. Observed Behavior

- Most remaining manifold misses in the `50`-paper checkpoint are `INSUFFICIENT_CONTEXT`.
- The sidecar-disabled manifold configuration remains better than the current baseline at `50` papers.
- The reconstruction sweep did not find a better setting than `top_k=5`, `per_paper_snippets=3`, and `max_context_tokens=2000`.

These are meaningful observations, but they are still early mid-scale observations rather than robust scaling laws.

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
- performance at `100-200` papers
- parity with stronger RAG baselines
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

- scale the benchmark to `100` and `200` papers
- compare structural nodes against stronger embedding baselines
- redesign the sidecar layer as verifier-only or verifier-first
- improve compression without losing the current retrieval signal
- test whether the pilot behavior transfers to other public corpora
