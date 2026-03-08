# Structural Manifold Compression

Experimental tooling for testing whether structural document representations can support retrieval and bounded reconstruction over corpora that exceed normal LLM context limits.

## What This Project Is

This repository explores a structural retrieval pipeline built from:

- a byte-level signature encoder
- section-level document nodes
- an optional sidecar signature layer
- a bounded-context answer stage using either extractive scoring or an LLM

The current evidence supports a retrieval claim on a leakage-aware `200`-paper arXiv benchmark. It does not yet support a strong corpus-compression claim.

## Evidence Levels

The repo uses the evidence ladder defined in [docs/evidence_ladder.md](docs/evidence_ladder.md):

- Level 1: measured result
- Level 2: observed behavior
- Level 3: hypothesis
- Level 4: research direction

## Current Benchmark Snapshot

Level 1: measured result

Dataset and protocol:

- `200` arXiv papers
- `250` frozen questions
- leakage-aware setup with neutral `paper_###` ids
- bounded reconstruction only; the model never sees the full corpus
- shuffled-manifold integrity control

Primary artifacts:

- `results/qa_results.json`
- `results/baseline_rag_results.json`
- `results/manifold_results.json`
- `results/manifold_results_no_sidecar.json`
- `results/manifold_results_shuffled.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`

### Current benchmark results

| System | QA | Top-1 | Top-5 | Artifact |
|--------|---:|------:|------:|----------|
| Baseline RAG (`ollama`) | 0.504 | 0.392 | 0.536 | `results/baseline_rag_results.json` |
| Structural manifold, no sidecar rerank | 0.716 | 0.728 | 0.828 | `results/manifold_results_no_sidecar.json` |
| Shuffled manifold control | 0.016 | 0.008 | 0.020 | `results/manifold_results_shuffled.json` |

Compression on the same pilot:

- original corpus tokens: `2,957,826`
- structural manifold tokens: `1,111,688`
- token compression: `2.66x`
- serialized manifold size: larger than the raw corpus bytes

## Interpretation

Level 1: measured result

- Structural node retrieval continues to outperform the current baseline at `200` papers.
- The shuffled control collapses, so the pilot is not surviving randomization.
- Disabling the current sidecar reranker improves QA from `0.850` to `0.900` in the committed ablation artifact.
- The best tested reconstruction settings are `top_k=5`, `per_paper_snippets=3`, and `max_context_tokens=2000`.

Level 2: observed behavior

- Most remaining manifold misses in the `200`-paper run are `INSUFFICIENT_CONTEXT`.
- The current sidecar layer behaves more like a noisy reranker than a helpful one on this benchmark.
- The manifold retrieval-to-QA gap is now small (`0.728` Top-1 vs `0.716` QA), so retrieval rather than answer formatting is carrying most of the remaining error.

Level 3: hypothesis

- The structural node representation appears to carry most of the useful retrieval signal in the current design.
- The sidecar signatures may be more useful as a verification layer than as a primary reranker.

Level 4: research direction

- Stronger embedding and hybrid retrieval baselines may show how much of the current win is specific to the simple baseline used here.
- Harder question sets may show whether the retrieval advantage survives beyond document-identification-heavy evaluation.
- A redesigned sidecar layer may still become useful if it is decoupled from ranking.

## What Is Established

Level 1: measured result

- Structural manifold encoding, indexing, and verification code exists in the repo.
- The corpus benchmark is leakage-aware and end-to-end reproducible.
- The `25`-paper, `50`-paper, and `200`-paper arXiv checkpoints all support a credible retrieval claim.
- The shuffled control, sidecar ablation, and reconstruction sweep are all implemented and committed as artifacts.

## What Is Not Yet Established

Level 4: research direction

- strong compression under the arXiv benchmark
- serialized storage reduction relative to the source corpus
- stronger baseline comparisons at the `200`-paper scale
- performance on broader question types than the current generated set
- value added by the sidecar layer after redesign
- any claim of general context-handling superiority over current transformer systems

## Current Limitations

Level 1: measured result

- Compression is currently weak at about `2.66x` token reduction in the latest `200`-paper checkpoint.
- Serialized manifold artifacts are larger than the source corpus bytes.
- The question set is generated automatically and is still weighted toward document identification rather than rich cross-document reasoning.
- The baseline comparison is a simple chunked RAG pipeline, not a broad leaderboard of embedding systems.

Level 2: observed behavior

- The current answer bottleneck is reconstruction sufficiency, not output normalization.
- The sidecar reranker is currently unnecessary on the best-performing pilot configuration.

## Architecture Overview

Level 3: hypothesis

The current architecture under test has four technical layers:

1. Structural encoder
   - encodes local byte-level patterns into compact signatures
2. Structural node index
   - segments documents into section-level nodes with headings, section types, salient phrases, and short text sketches
3. Sidecar signature layer
   - provides optional secondary overlap features for verification or reranking
4. Language-model answer layer
   - answers from reconstructed evidence only, with a bounded context budget

This is a proposed framework for structural retrieval and bounded reconstruction. It should not be read as a proven replacement for token-based language models.

## Reproducible Commands

Run the full pipeline:

```bash
python run_full_demo.py
```

Recommended 200-paper checkpoint:

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

Run the sidecar ablation on the locked corpus/questions:

```bash
python demo/run_manifold_ablation.py \
  --qa-backend ollama \
  --top-k 5 \
  --per-paper-snippets 3 \
  --max-context-tokens 2000
```

Run the reconstruction sweep on the locked corpus/questions:

```bash
python demo/run_manifold_sweep.py \
  --qa-backend ollama \
  --disable-sidecar-rerank
```

## Repo Layout

```text
demo/
   build_corpus.py
   generate_manifold.py
   run_baseline_rag.py
   run_manifold_system.py
   run_manifold_ablation.py
   run_manifold_sweep.py
   evaluate.py

data/
   corpus/
   corpus_manifest.json
   corpus_full.txt
   questions.json

manifold/
   manifold.json
   manifold_index.bin

results/
   compression_metrics.json
   baseline_rag_results.json
   manifold_results.json
   manifold_results_no_sidecar.json
   manifold_results_shuffled.json
   manifold_ablation.json
   manifold_reconstruction_sweep.json
   qa_results.json
   graphs/
```

## Other Experimental Components

Level 2: observed behavior

- The repo also contains pair-programmer and anomaly-monitoring demos.
- These are exploratory applications of the structural tooling, not evidence for the corpus-compression claim.

Level 4: research direction

- structural retrieval as agent memory
- structural verification layers for retrieval pipelines
- larger document benchmarks beyond arXiv

## Prior Reported Results

Level 2: observed behavior

The repository also contains earlier internal measurements from narrower settings. They are useful background, but they do not establish the current corpus-scale claim.

| Dataset | Docs | Byte x | Token x | Token Acc. | Char Acc. | Verif. Precision | Verif. FPR |
|---------|-----:|-------:|--------:|-----------:|----------:|-----------------:|-----------:|
| Fox EN  | 112 | 42.03 | 85.48 | 95.35 % | 95.62 % | 91.21 % | 0.087 % |
| Fox CN  | 100 | 42.01 | 88.08 | 94.94 % | 95.04 % | 97.19 % | 0.029 % |
| OmniDoc | 1,349 | 41.59 | 89.49 | 94.90 % | 94.94 % | 80.85 % | 0.017 % |

## Research Directions

Level 4: research direction

- validate the locked benchmark at `50`, `100`, and `200` papers
- redesign the sidecar layer as verifier-first rather than reranker-first
- compare against stronger embedding and hybrid retrieval baselines
- test whether compression can improve without losing the current retrieval signal

## License and Citation

This project is released under the [MIT License](LICENSE).

```bibtex
@misc{nagy2025manifold,
  author       = {Alexander Nagy},
  title        = {Structural Manifold Compression: Experimental Structural Retrieval and Bounded Reconstruction},
  year         = {2025},
  howpublished = {\url{https://github.com/SepDynamics/structural-manifold-compression}}
}
```
