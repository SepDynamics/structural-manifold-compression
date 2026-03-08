#!/usr/bin/env python3
from __future__ import annotations

import argparse

from demo.common import MANIFOLD_NO_SIDECAR_RESULTS_PATH, MANIFOLD_RESULTS_PATH
from demo.build_corpus import build_corpus
from demo.evaluate import evaluate
from demo.generate_manifold import generate_manifold
from demo.run_baseline_rag import run_baseline
from demo.run_manifold_system import run_manifold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full structural manifold corpus demo.")
    parser.add_argument("--paper-count", type=int, default=60)
    parser.add_argument("--question-count", type=int, default=60)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-chars", type=int, default=2200)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--node-chars", type=int, default=1500)
    parser.add_argument("--node-overlap", type=int, default=180)
    parser.add_argument("--window-bytes", type=int, default=16)
    parser.add_argument("--stride-bytes", type=int, default=4)
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--per-paper-snippets", type=int, default=3)
    parser.add_argument("--disable-sidecar-rerank", action="store_true")
    parser.add_argument("--sidecar-weight", type=float, default=0.25)
    parser.add_argument("--phrase-weight", type=float, default=0.15)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--qa-backend", choices=("ollama", "extractive"), default="ollama")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--ollama-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_corpus(
        argparse.Namespace(
            paper_count=args.paper_count,
            question_count=args.question_count,
            categories=args.categories,
            input_dir=args.input_dir,
            force=args.force,
        )
    )
    generate_manifold(
        argparse.Namespace(
            node_chars=args.node_chars,
            node_overlap=args.node_overlap,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=args.precision,
            embedding_model=args.embedding_model,
        )
    )
    run_baseline(
        argparse.Namespace(
            embedding_model=args.embedding_model,
            qa_backend=args.qa_backend,
            ollama_endpoint=args.ollama_endpoint,
            ollama_model=args.ollama_model,
            top_k=args.top_k,
            max_context_tokens=args.max_context_tokens,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
        )
    )
    run_manifold(
        argparse.Namespace(
            qa_backend=args.qa_backend,
            ollama_endpoint=args.ollama_endpoint,
            ollama_model=args.ollama_model,
            top_k=args.top_k,
            max_context_tokens=args.max_context_tokens,
            per_paper_snippets=args.per_paper_snippets,
            disable_sidecar_rerank=args.disable_sidecar_rerank,
            sidecar_weight=args.sidecar_weight,
            phrase_weight=args.phrase_weight,
            output_path=None,
            shuffle_index=False,
        )
    )
    run_manifold(
        argparse.Namespace(
            qa_backend=args.qa_backend,
            ollama_endpoint=args.ollama_endpoint,
            ollama_model=args.ollama_model,
            top_k=args.top_k,
            max_context_tokens=args.max_context_tokens,
            per_paper_snippets=args.per_paper_snippets,
            disable_sidecar_rerank=args.disable_sidecar_rerank,
            sidecar_weight=args.sidecar_weight,
            phrase_weight=args.phrase_weight,
            output_path=None,
            shuffle_index=True,
        )
    )
    manifold_results_path = (
        MANIFOLD_NO_SIDECAR_RESULTS_PATH if args.disable_sidecar_rerank else MANIFOLD_RESULTS_PATH
    )
    payload = evaluate(argparse.Namespace(manifold_results_path=manifold_results_path))
    accuracy = payload["qa_accuracy"]
    print(
        "Demo complete:\n"
        f"  Baseline RAG QA accuracy: {accuracy['baseline_rag']:.3f}\n"
        f"  Structural manifold QA accuracy: {accuracy['structural_manifold']:.3f}"
    )


if __name__ == "__main__":
    main()
