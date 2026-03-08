#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.common import (
    DEFAULT_OLLAMA_GENERATE_ENDPOINT,
    MANIFOLD_SWEEP_PATH,
    resolve_ollama_model,
    write_metrics,
)
from demo.run_manifold_system import run_manifold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep manifold reconstruction settings on the current locked corpus/questions."
    )
    parser.add_argument("--qa-backend", choices=("ollama", "extractive"), default="ollama")
    parser.add_argument("--ollama-endpoint", default=DEFAULT_OLLAMA_GENERATE_ENDPOINT)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--per-paper-snippet-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-context-token-values", type=int, nargs="+", default=[1200, 2000, 2800])
    parser.add_argument("--disable-sidecar-rerank", action="store_true")
    parser.add_argument("--sidecar-weight", type=float, default=0.25)
    parser.add_argument("--phrase-weight", type=float, default=0.15)
    parser.add_argument("--output-path", type=Path, default=MANIFOLD_SWEEP_PATH)
    return parser.parse_args()


def _run_args(
    args: argparse.Namespace,
    *,
    top_k: int,
    per_paper_snippets: int,
    max_context_tokens: int,
    ollama_model: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        qa_backend=args.qa_backend,
        ollama_endpoint=args.ollama_endpoint,
        ollama_model=ollama_model,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        per_paper_snippets=per_paper_snippets,
        disable_sidecar_rerank=args.disable_sidecar_rerank,
        sidecar_weight=args.sidecar_weight,
        phrase_weight=args.phrase_weight,
        output_path=None,
        shuffle_index=False,
        write_output=False,
    )


def _summary_row(
    payload: dict[str, object],
    *,
    top_k: int,
    per_paper_snippets: int,
    max_context_tokens: int,
) -> dict[str, object]:
    rows = payload["results"]
    insufficient_context = sum(1 for row in rows if row["predicted_answer"] == "INSUFFICIENT_CONTEXT")
    retrieval_true_but_wrong = sum(1 for row in rows if row["retrieval_top1"] and not row["correct"])
    summary = payload["summary"]
    return {
        "top_k": top_k,
        "per_paper_snippets": per_paper_snippets,
        "max_context_tokens": max_context_tokens,
        "qa_accuracy": summary["qa_accuracy"],
        "retrieval_top1": summary["retrieval_top1"],
        "retrieval_top5": summary["retrieval_top5"],
        "mean_latency_seconds": summary["mean_latency_seconds"],
        "insufficient_context": insufficient_context,
        "retrieval_top1_but_wrong": retrieval_true_but_wrong,
        "sidecar_rerank": summary["sidecar_rerank"],
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    resolved_model = args.ollama_model
    if args.qa_backend == "ollama" and not resolved_model:
        resolved_model = resolve_ollama_model()

    runs: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_payload: dict[str, object] | None = None

    for top_k, per_paper_snippets, max_context_tokens in product(
        args.top_k_values,
        args.per_paper_snippet_values,
        args.max_context_token_values,
    ):
        payload = run_manifold(
            _run_args(
                args,
                top_k=top_k,
                per_paper_snippets=per_paper_snippets,
                max_context_tokens=max_context_tokens,
                ollama_model=resolved_model,
            )
        )
        row = _summary_row(
            payload,
            top_k=top_k,
            per_paper_snippets=per_paper_snippets,
            max_context_tokens=max_context_tokens,
        )
        runs.append(row)
        if best_row is None or (
            row["qa_accuracy"],
            row["retrieval_top1"],
            -row["insufficient_context"],
            -row["mean_latency_seconds"],
        ) > (
            best_row["qa_accuracy"],
            best_row["retrieval_top1"],
            -best_row["insufficient_context"],
            -best_row["mean_latency_seconds"],
        ):
            best_row = row
            best_payload = payload

    runs.sort(
        key=lambda row: (
            row["qa_accuracy"],
            row["retrieval_top1"],
            -row["insufficient_context"],
            -row["mean_latency_seconds"],
        ),
        reverse=True,
    )

    payload = {
        "study": "manifold_reconstruction_sweep",
        "config": {
            "qa_backend": args.qa_backend,
            "top_k_values": args.top_k_values,
            "per_paper_snippet_values": args.per_paper_snippet_values,
            "max_context_token_values": args.max_context_token_values,
            "sidecar_rerank": not args.disable_sidecar_rerank,
            "sidecar_weight": args.sidecar_weight if not args.disable_sidecar_rerank else 0.0,
            "phrase_weight": args.phrase_weight,
        },
        "best_run": best_row,
        "best_run_results": best_payload["results"] if best_payload else [],
        "runs": runs,
    }
    write_metrics(args.output_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_sweep(args)
    best = payload["best_run"]
    print(
        "Sweep complete:\n"
        f"  Best QA: {best['qa_accuracy']:.3f}\n"
        f"  Best config: top_k={best['top_k']}, "
        f"per_paper_snippets={best['per_paper_snippets']}, "
        f"max_context_tokens={best['max_context_tokens']}"
    )


if __name__ == "__main__":
    main()
