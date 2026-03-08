#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.common import (
    DEFAULT_OLLAMA_GENERATE_ENDPOINT,
    MANIFOLD_ABLATION_PATH,
    ensure_directories,
    write_metrics,
)
from demo.run_manifold_system import run_manifold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a locked-corpus ablation comparing structural nodes with and without sidecar reranking."
    )
    parser.add_argument("--qa-backend", choices=("ollama", "extractive"), default="ollama")
    parser.add_argument("--ollama-endpoint", default=DEFAULT_OLLAMA_GENERATE_ENDPOINT)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--per-paper-snippets", type=int, default=3)
    parser.add_argument("--sidecar-weight", type=float, default=0.25)
    parser.add_argument("--phrase-weight", type=float, default=0.15)
    parser.add_argument("--output-path", type=Path, default=MANIFOLD_ABLATION_PATH)
    return parser.parse_args()


def _manifold_args(args: argparse.Namespace, *, disable_sidecar_rerank: bool) -> SimpleNamespace:
    return SimpleNamespace(
        qa_backend=args.qa_backend,
        ollama_endpoint=args.ollama_endpoint,
        ollama_model=args.ollama_model,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        per_paper_snippets=args.per_paper_snippets,
        disable_sidecar_rerank=disable_sidecar_rerank,
        sidecar_weight=args.sidecar_weight,
        phrase_weight=args.phrase_weight,
        output_path=None,
        shuffle_index=False,
        write_output=False,
    )


def _error_stats(payload: dict[str, object]) -> dict[str, int]:
    rows = payload["results"]
    retrieval_true_but_wrong = sum(1 for row in rows if row["retrieval_top1"] and not row["correct"])
    insufficient_context = sum(1 for row in rows if row["predicted_answer"] == "INSUFFICIENT_CONTEXT")
    return {
        "retrieval_top1_but_wrong": retrieval_true_but_wrong,
        "insufficient_context": insufficient_context,
    }


def run_ablation(args: argparse.Namespace) -> dict[str, object]:
    ensure_directories()

    with_sidecar = run_manifold(_manifold_args(args, disable_sidecar_rerank=False))
    without_sidecar = run_manifold(_manifold_args(args, disable_sidecar_rerank=True))

    comparison = {
        "qa_accuracy_delta": with_sidecar["summary"]["qa_accuracy"] - without_sidecar["summary"]["qa_accuracy"],
        "retrieval_top1_delta": with_sidecar["summary"]["retrieval_top1"] - without_sidecar["summary"]["retrieval_top1"],
        "retrieval_top5_delta": with_sidecar["summary"]["retrieval_top5"] - without_sidecar["summary"]["retrieval_top5"],
        "latency_delta_seconds": with_sidecar["summary"]["mean_latency_seconds"] - without_sidecar["summary"]["mean_latency_seconds"],
    }

    payload = {
        "study": "sidecar_ablation",
        "config": {
            "qa_backend": args.qa_backend,
            "top_k": args.top_k,
            "max_context_tokens": args.max_context_tokens,
            "per_paper_snippets": args.per_paper_snippets,
            "sidecar_weight": args.sidecar_weight,
            "phrase_weight": args.phrase_weight,
        },
        "with_sidecar": {
            "summary": with_sidecar["summary"],
            "error_stats": _error_stats(with_sidecar),
            "results": with_sidecar["results"],
        },
        "without_sidecar": {
            "summary": without_sidecar["summary"],
            "error_stats": _error_stats(without_sidecar),
            "results": without_sidecar["results"],
        },
        "comparison": comparison,
    }
    write_metrics(args.output_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_ablation(args)
    delta = payload["comparison"]
    print(
        "Ablation complete:\n"
        f"  With sidecar QA: {payload['with_sidecar']['summary']['qa_accuracy']:.3f}\n"
        f"  Without sidecar QA: {payload['without_sidecar']['summary']['qa_accuracy']:.3f}\n"
        f"  QA delta: {delta['qa_accuracy_delta']:+.3f}"
    )


if __name__ == "__main__":
    main()
