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
    BASELINE_BM25_RESULTS_PATH,
    BASELINE_HYBRID_RESULTS_PATH,
    BASELINE_HYBRID_RERANKED_RESULTS_PATH,
    BASELINE_RESULTS_PATH,
    BASELINE_STRONG_DENSE_RESULTS_PATH,
    BASELINE_SUITE_PATH,
    ensure_directories,
    write_metrics,
)
from demo.run_baseline_rag import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stronger baseline comparison suite on the locked corpus/questions."
    )
    parser.add_argument("--qa-backend", choices=("extractive", "ollama"), default="extractive")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--chunk-chars", type=int, default=2200)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--include-cross-encoder", action="store_true")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--strong-dense-model", default=None, help="Optional additional dense model to benchmark.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    return parser.parse_args()


def _baseline_args(args: argparse.Namespace, **overrides: object) -> SimpleNamespace:
    payload = {
        "retrieval_method": "dense",
        "embedding_model": "all-MiniLM-L6-v2",
        "device": args.device,
        "reranker_model": None,
        "reranker_device": args.reranker_device,
        "rerank_top_n": args.rerank_top_n,
        "rrf_k": 60,
        "qa_backend": args.qa_backend,
        "ollama_endpoint": args.ollama_endpoint,
        "ollama_model": args.ollama_model,
        "top_k": args.top_k,
        "max_context_tokens": args.max_context_tokens,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap": args.chunk_overlap,
        "output_path": BASELINE_RESULTS_PATH,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def run_suite(args: argparse.Namespace) -> dict[str, object]:
    ensure_directories()

    suite_configs: list[tuple[str, SimpleNamespace]] = [
        (
            "dense_minilm",
            _baseline_args(
                args,
                retrieval_method="dense",
                embedding_model="all-MiniLM-L6-v2",
                output_path=BASELINE_RESULTS_PATH,
            ),
        ),
        (
            "bm25",
            _baseline_args(
                args,
                retrieval_method="bm25",
                embedding_model="all-MiniLM-L6-v2",
                output_path=BASELINE_BM25_RESULTS_PATH,
            ),
        ),
        (
            "hybrid_minilm_bm25",
            _baseline_args(
                args,
                retrieval_method="hybrid",
                embedding_model="all-MiniLM-L6-v2",
                output_path=BASELINE_HYBRID_RESULTS_PATH,
            ),
        ),
    ]
    if args.strong_dense_model:
        suite_configs.append(
            (
                "dense_strong",
                _baseline_args(
                    args,
                    retrieval_method="dense",
                    embedding_model=args.strong_dense_model,
                    output_path=BASELINE_STRONG_DENSE_RESULTS_PATH,
                ),
            )
        )
    if args.include_cross_encoder:
        suite_configs.append(
            (
                "hybrid_minilm_bm25_crossencoder",
                _baseline_args(
                    args,
                    retrieval_method="hybrid",
                    embedding_model="all-MiniLM-L6-v2",
                    reranker_model=args.cross_encoder_model,
                    output_path=BASELINE_HYBRID_RERANKED_RESULTS_PATH,
                ),
            )
        )

    summaries: list[dict[str, object]] = []
    for label, baseline_args in suite_configs:
        payload = run_baseline(baseline_args)
        summary = dict(payload["summary"])
        summary["label"] = label
        summary["artifact_path"] = str(baseline_args.output_path)
        summaries.append(summary)

    payload = {
        "qa_backend": args.qa_backend,
        "comparisons": summaries,
    }
    write_metrics(BASELINE_SUITE_PATH, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_suite(args)
    for summary in payload["comparisons"]:
        print(
            f"{summary['label']}: QA {summary['qa_accuracy']:.3f} | "
            f"Top-1 {summary['retrieval_top1']:.3f} | "
            f"Top-5 {summary['retrieval_top5']:.3f}"
        )


if __name__ == "__main__":
    main()
