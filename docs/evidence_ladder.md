# Evidence Ladder

This repository uses four evidence levels for technical claims.

## Level 1: Measured Result

Directly supported by a committed artifact, benchmark output, test result, or reproducible script.

Examples:

- `results/qa_results.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`
- `tests/test_demo_pipeline.py`

Recommended phrasing:

- "On the locked 25-paper arXiv pilot, structural manifold retrieval reached Top-1 retrieval of 0.90."
- "Disabling sidecar reranking improved Ollama QA from 0.85 to 0.90 in the committed ablation artifact."

## Level 2: Observed Behavior

Seen in one or more experiments, but not yet tested broadly enough to treat as a stable result.

Recommended phrasing:

- "In the current pilot, most remaining manifold misses are `INSUFFICIENT_CONTEXT` rather than malformed titles."
- "The current sidecar reranker does not improve QA on the locked pilot."

## Level 3: Hypothesis

A proposed explanation for a measured result or observed behavior.

Recommended phrasing:

- "This suggests the structural node representation is carrying most of the retrieval signal."
- "The sidecar signatures may be better suited for verification than reranking in the current design."

## Level 4: Research Direction

Future work, open questions, or speculative potential.

Recommended phrasing:

- "Structural manifold indexing may support larger retrieval benchmarks if the pilot behavior survives scale-out."
- "A redesigned sidecar layer may become useful for post-retrieval verification."

## Repo Rules

- Prefer direct artifact references for Level 1 claims.
- Avoid absolute language that implies solved status, universal superiority, or unbounded scaling unless a claim is proved in the narrow sense used.
- Keep architecture descriptions at Level 3 unless they are directly benchmarked.
- Put speculative ideas in a dedicated research section instead of mixing them with results.
