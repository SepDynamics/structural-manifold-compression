#!/usr/bin/env python3
"""FEP learning test: measure free-energy spike to assimilation latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter


def measure_fep_learning(
    router: TripartiteRouter,
    baseline_text: str,
    novel_text: str,
    query: str,
    hazard_threshold: float,
    coverage_threshold: float,
) -> Dict[str, float]:
    router.wm.clear_all()
    router.wm.add_document("baseline", baseline_text)

    start = time.time()
    router.process_query(
        query,
        hazard_threshold=hazard_threshold,
        coverage_threshold=coverage_threshold,
    )
    baseline_time = time.time() - start

    spike_start = time.time()
    router.wm.add_document("novel_fact", novel_text)
    router.process_query(
        query,
        hazard_threshold=hazard_threshold,
        coverage_threshold=coverage_threshold,
    )
    spike_time = time.time() - spike_start

    return {
        "baseline_seconds": baseline_time,
        "spike_to_assimilate_seconds": spike_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Free-Energy Principle learning benchmark")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline text file")
    parser.add_argument("--novel", type=Path, required=True, help="Novel contradiction text file")
    parser.add_argument("--query", type=str, required=True, help="Query to measure learning response")
    parser.add_argument("--hazard-threshold", type=float, default=0.8)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/fep_learning_test.json"))
    args = parser.parse_args()

    router = TripartiteRouter()
    baseline_text = args.baseline.read_text(encoding="utf-8")
    novel_text = args.novel.read_text(encoding="utf-8")

    metrics = measure_fep_learning(
        router,
        baseline_text,
        novel_text,
        args.query,
        args.hazard_threshold,
        args.coverage_threshold,
    )

    payload = {
        "query": args.query,
        "hazard_threshold": args.hazard_threshold,
        "coverage_threshold": args.coverage_threshold,
        **metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
