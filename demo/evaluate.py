#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.common import (
    BASELINE_RESULTS_PATH,
    COMPRESSION_METRICS_PATH,
    GRAPHS_DIR,
    MANIFOLD_NO_SIDECAR_RESULTS_PATH,
    MANIFOLD_RESULTS_PATH,
    QA_RESULTS_PATH,
    SHUFFLED_MANIFOLD_RESULTS_PATH,
    ensure_directories,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compression, retrieval, QA, and latency metrics.")
    parser.add_argument("--manifold-results-path", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_manifold_results_path(
    *,
    preferred_path: Path | None,
    baseline: dict[str, object],
) -> Path:
    if preferred_path is not None:
        return preferred_path

    baseline_summary = baseline.get("summary", {})
    baseline_questions = int(baseline_summary.get("questions", 0))
    baseline_backend = str(baseline_summary.get("qa_backend", ""))

    candidates: list[tuple[tuple[int, int, int, float], Path]] = []
    for path in (MANIFOLD_RESULTS_PATH, MANIFOLD_NO_SIDECAR_RESULTS_PATH):
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        summary = payload.get("summary", {})
        score = (
            int(int(summary.get("questions", 0)) == baseline_questions),
            int(str(summary.get("qa_backend", "")) == baseline_backend),
            int(int(summary.get("questions", 0)) > 0),
            path.stat().st_mtime,
        )
        candidates.append((score, path))

    if not candidates:
        return MANIFOLD_RESULTS_PATH
    return max(candidates, key=lambda item: item[0])[1]


def _plot_compression(metrics: dict[str, object]) -> str:
    output = GRAPHS_DIR / "compression_ratio.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Corpus", "Manifold"]
    values = [metrics["original_tokens"], metrics["manifold_signature_tokens"]]
    ax.bar(labels, values, color=["#23395d", "#ff7f11"])
    ax.set_ylabel("Estimated tokens")
    ax.set_title("Corpus Tokens vs Manifold Tokens")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def _plot_accuracy(baseline: dict[str, object], manifold: dict[str, object]) -> str:
    output = GRAPHS_DIR / "qa_accuracy.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Full corpus\n(oracle)", "Baseline RAG", "Structural manifold"]
    values = [1.0, baseline["summary"]["qa_accuracy"], manifold["summary"]["qa_accuracy"]]
    ax.bar(labels, values, color=["#b8c0ff", "#4361ee", "#ff7f11"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("QA Accuracy")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def _plot_latency(baseline: dict[str, object], manifold: dict[str, object]) -> str:
    output = GRAPHS_DIR / "latency.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Baseline RAG", "Structural manifold"]
    values = [baseline["summary"]["mean_latency_seconds"], manifold["summary"]["mean_latency_seconds"]]
    ax.bar(labels, values, color=["#4361ee", "#ff7f11"])
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Query Latency")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def evaluate(_: argparse.Namespace) -> dict[str, object]:
    ensure_directories()
    compression = _load_json(COMPRESSION_METRICS_PATH)
    baseline = _load_json(BASELINE_RESULTS_PATH)
    preferred_manifold_path = getattr(_, "manifold_results_path", None)
    manifold_path = _select_manifold_results_path(
        preferred_path=preferred_manifold_path,
        baseline=baseline,
    )
    manifold = _load_json(manifold_path)
    shuffled = _load_json(SHUFFLED_MANIFOLD_RESULTS_PATH) if SHUFFLED_MANIFOLD_RESULTS_PATH.exists() else None

    graphs = {
        "compression_ratio": _plot_compression(compression),
        "qa_accuracy": _plot_accuracy(baseline, manifold),
        "latency": _plot_latency(baseline, manifold),
    }

    payload = {
        "compression": compression,
        "qa_accuracy": {
            "full_corpus_oracle": 1.0,
            "baseline_rag": baseline["summary"]["qa_accuracy"],
            "structural_manifold": manifold["summary"]["qa_accuracy"],
        },
        "retrieval_accuracy": {
            "baseline_top1": baseline["summary"]["retrieval_top1"],
            "baseline_top5": baseline["summary"]["retrieval_top5"],
            "manifold_top1": manifold["summary"]["retrieval_top1"],
            "manifold_top5": manifold["summary"]["retrieval_top5"],
        },
        "latency_seconds": {
            "baseline_rag": baseline["summary"]["mean_latency_seconds"],
            "structural_manifold": manifold["summary"]["mean_latency_seconds"],
        },
        "integrity_checks": {
            "question_hash_locked": compression["question_hash"],
            "randomized_manifold_qa_accuracy": (
                shuffled["summary"]["qa_accuracy"] if shuffled is not None else None
            ),
            "randomized_manifold_top1": (
                shuffled["summary"]["retrieval_top1"] if shuffled is not None else None
            ),
        },
        "graphs": graphs,
        "manifold_results_path": str(manifold_path),
    }
    write_metrics(QA_RESULTS_PATH, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    print(json.dumps(payload["qa_accuracy"], indent=2))


if __name__ == "__main__":
    main()
