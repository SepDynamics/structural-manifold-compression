#!/usr/bin/env python3
"""
LLM FEP Saturation Benchmark
Simulates a massive sequence of High Hazard (Structural Tension) collisions
to stress-test the LLM 'ADHD Burst' heuristic resolver over the saturated
Valkey index.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter


def test_fep_saturation(router: TripartiteRouter, query: str) -> None:
    print(f"\n[FEP Saturation Test] Query: '{query}'")
    start_time = time.perf_counter()

    verified, response, coverage, matched_documents = router.process_query(
        query=query,
        hazard_threshold=0.8,
        coverage_threshold=0.5,
        llm_endpoint="http://localhost:11434/api/generate",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "Deterministically Resolved" if verified else "High Hazard (LLM Fallback)"
    print(f"Status: {status} (Coverage: {coverage:.2f}%)")
    print(f"Latency: {elapsed_ms:.2f} ms")
    print(f"Nodes Matched in Valkey: {len(matched_documents)}")

    if not verified:
        print("\n🤖 LLM Heuristic Resolution Snippet:")
        # Print first 200 chars to avoid flooding terminal
        print(f"'{str(response)[:200]}...'")
    else:
        print(
            "⚠️ WARNING: Query did not trigger an LLM FEP saturation. Hazard was too low."
        )


def main():
    print("Initializing Tripartite Router against live Valkey DB...")
    router = TripartiteRouter()

    if not router.wm.ping():
        print("❌ CRITICAL: Valkey Work Memory is offline.")
        sys.exit(1)

    index = router.wm.get_cached_index()
    if index is None:
        print("❌ CRITICAL: Valkey index is empty. Run bulk ingestion first.")
        sys.exit(1)

    meta = index.meta
    print(
        f"Valkey State: {meta.get('total_signatures', 0)} Signatures | {meta.get('total_windows', 0)} Windows"
    )

    print("\n--- LLM FEP Saturation Sequence ---")

    # We will run highly complex, multi-scale abstract queries that use some physical elements from the corpus
    # (e.g. math syntax, continuous functions) but in illogical overlapping ways to generate High Hazard Tension.
    # This will trigger the Latent Semantic Adapter to intercept the prompt with the Recency Buffer instead of raw text.

    import json

    corpus_path = REPO_ROOT / "data/raw_math/synthetic_linear_qa.jsonl"
    with open(corpus_path, "r") as f:
        real_text = json.loads(f.readline())["text"]

    queries = [
        # Query 1: Exactly matches the first 600 bytes to overlap structurally,
        # then collapses rapidly with philosophical ambiguity to trigger the LLM Latent Semantic Adapter.
        real_text[:600]
        + "\n\nAND THEN THE UNIVERSE EXPANDED INTO A KALEIDOSCOPE OF PHILOSOPHICAL EPISTEMOLOGY BEYOND THE TENSION GATE!"
        * 5,
    ]

    for q in queries:
        test_fep_saturation(router, q)
        print("-" * 60)
        time.sleep(1)  # Breath between LLM generations


if __name__ == "__main__":
    main()
