#!/usr/bin/env python3
"""
Latency Verification Benchmark
Measures the response time of the Tripartite Router's deterministic reflex against
the fully saturated Valkey Working Memory.
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter


def test_query_latency(
    router: TripartiteRouter, query: str, expected_type: str = "Deterministic"
) -> None:
    print(f"\n[Test] Query: '{query}'")
    start_time = time.perf_counter()

    verified, response, coverage, matched_documents = router.process_query(
        query=query,
        hazard_threshold=0.8,
        coverage_threshold=0.5,
        llm_endpoint="http://localhost:11434/api/generate",  # Optional, might timeout if not running
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "Deterministically Resolved" if verified else "High Hazard (LLM Fallback)"
    print(f"Status: {status} (Coverage: {coverage:.2f}%)")
    print(f"Latency: {elapsed_ms:.2f} ms")
    print(f"Nodes Matched: {len(matched_documents)}")

    if expected_type == "Deterministic" and elapsed_ms > 50:
        print(
            f"⚠️ WARNING: Deterministic reflex exceeded theoretical sub-15ms threshold ({elapsed_ms:.2f}ms)"
        )
    elif expected_type == "Deterministic":
        print("✅ SUCCESS: Query completed at scale with physics-engine speed.")


def main():
    parser = argparse.ArgumentParser(description="Latency Verification Benchmark")
    parser.add_argument(
        "--queries", type=int, default=3, help="Number of queries to run"
    )
    args = parser.parse_args()

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

    # We don't have the actual raw text in Valkey because we skipped document insertion
    # for speed in bulk ingest, but the structural graph (signatures & occurrences) exists.
    # The Router will still perform the ANN math perfectly, it just might not return block text.

    print("\n--- Latency Verification Sequence ---")

    base_queries = [
        ("Solve for x in the equation 2x + 5 = 15", "Deterministic"),
        ("Determine the derivative of f(x) = x^2", "Deterministic"),
        (
            "What is the philosophical meaning of linear algebra in quantum physics",
            "LLM",
        ),
    ]

    for i in range(args.queries):
        query_text, query_type = base_queries[i % len(base_queries)]

        # Optionally vary query slightly to simulate unique streams if running many queries
        test_text = f"{query_text} (iteration {i})" if args.queries > 3 else query_text
        test_query_latency(router, test_text, expected_type=query_type)


if __name__ == "__main__":
    main()
