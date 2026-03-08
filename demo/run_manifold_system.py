#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.common import (
    DEFAULT_OLLAMA_GENERATE_ENDPOINT,
    MANIFOLD_INDEX_PATH,
    MANIFOLD_JSON_PATH,
    MANIFOLD_NO_SIDECAR_RESULTS_PATH,
    MANIFOLD_RESULTS_PATH,
    SHUFFLED_MANIFOLD_RESULTS_PATH,
    answer_from_context,
    build_retrieval_query,
    ensure_directories,
    load_questions,
    resolve_ollama_model,
    score_prediction,
    write_metrics,
)
from demo.retrieval import load_manifold_index, rank_documents, rank_manifold_nodes_detailed
from demo.structure import build_node_contexts, deserialize_nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the structural manifold retrieval system.")
    parser.add_argument("--qa-backend", choices=("ollama", "extractive"), default="ollama")
    parser.add_argument("--ollama-endpoint", default=DEFAULT_OLLAMA_GENERATE_ENDPOINT)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--per-paper-snippets", type=int, default=3)
    parser.add_argument("--disable-sidecar-rerank", action="store_true")
    parser.add_argument("--sidecar-weight", type=float, default=0.25)
    parser.add_argument("--phrase-weight", type=float, default=0.15)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--shuffle-index", action="store_true", help="Permute retrieved structural nodes across papers as an integrity check.")
    return parser.parse_args()


def _retrieval_hits(ranked_docs: list[tuple[str, float]], expected: list[str]) -> tuple[bool, bool]:
    doc_ids = [doc_id for doc_id, _ in ranked_docs]
    top1 = bool(doc_ids and doc_ids[0] in expected)
    if len(expected) > 1:
        top5 = set(expected).issubset(set(doc_ids[:5]))
    else:
        top5 = any(doc_id in expected for doc_id in doc_ids[:5])
    return top1, top5


def run_manifold(args: argparse.Namespace) -> dict[str, object]:
    ensure_directories()
    if not MANIFOLD_JSON_PATH.exists() or not MANIFOLD_INDEX_PATH.exists():
        raise FileNotFoundError("Manifold artifacts are missing. Run generate_manifold.py first.")

    metadata = json.loads(MANIFOLD_JSON_PATH.read_text(encoding="utf-8"))
    questions = load_questions()
    nodes = deserialize_nodes(metadata.get("nodes", []))
    index_payload = load_manifold_index(MANIFOLD_INDEX_PATH)
    ollama_model = resolve_ollama_model(args.ollama_model) if args.qa_backend == "ollama" else None
    embedding_model = str(metadata.get("embedding_model", "all-MiniLM-L6-v2"))

    results: list[dict[str, object]] = []
    total_latency = 0.0

    for question in questions:
        started = time.perf_counter()
        retrieval_query = build_retrieval_query(question)
        ranked_node_details = rank_manifold_nodes_detailed(
            question,
            nodes=nodes,
            index_payload=index_payload,
            embedding_model=embedding_model,
            window_bytes=int(metadata["window_bytes"]),
            stride_bytes=int(metadata["stride_bytes"]),
            precision=int(metadata["precision"]),
            top_k=args.top_k,
            shuffle_nodes=args.shuffle_index,
            use_sidecar_rerank=not args.disable_sidecar_rerank,
            sidecar_weight=args.sidecar_weight,
            phrase_weight=args.phrase_weight,
        )
        ranked_node_scores = [
            (int(item["node_index"]), float(item["score"])) for item in ranked_node_details
        ]
        ranked_nodes = [nodes[idx] for idx, _ in ranked_node_scores]
        ranked_docs = rank_documents(ranked_node_scores, nodes)
        contexts = build_node_contexts(
            ranked_nodes,
            max_context_tokens=args.max_context_tokens,
            snippets_per_paper=args.per_paper_snippets,
        )
        answer = answer_from_context(
            question,
            contexts,
            qa_backend=args.qa_backend,
            ollama_endpoint=args.ollama_endpoint,
            ollama_model=ollama_model,
        )
        elapsed = time.perf_counter() - started
        top1, top5 = _retrieval_hits(ranked_docs, question.source_papers)
        total_latency += elapsed
        results.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "retrieval_query": retrieval_query,
                "expected_answer": question.answer,
                "predicted_answer": answer,
                "correct": score_prediction(answer, question),
                "source_papers": question.source_papers,
                "retrieved_papers": [doc_id for doc_id, _ in ranked_docs[: args.top_k]],
                "retrieved_nodes": [node.node_id for node in ranked_nodes[: args.top_k]],
                "retrieval_top1": top1,
                "retrieval_top5": top5,
                "latency_seconds": elapsed,
                "shuffle_index": args.shuffle_index,
                "sidecar_rerank": not args.disable_sidecar_rerank,
                "sidecar_verified": any(bool(item["verified"]) for item in ranked_node_details),
                "sidecar_scores": [float(item["sidecar_score"]) for item in ranked_node_details],
            }
        )

    question_count = max(1, len(results))
    summary = {
        "system": "manifold_system",
        "qa_backend": args.qa_backend,
        "questions": len(results),
        "qa_accuracy": sum(1 for row in results if row["correct"]) / question_count,
        "retrieval_top1": sum(1 for row in results if row["retrieval_top1"]) / question_count,
        "retrieval_top5": sum(1 for row in results if row["retrieval_top5"]) / question_count,
        "mean_latency_seconds": total_latency / question_count,
        "max_context_tokens": args.max_context_tokens,
        "per_paper_snippets": args.per_paper_snippets,
        "top_k": args.top_k,
        "shuffle_index": args.shuffle_index,
        "sidecar_rerank": not args.disable_sidecar_rerank,
        "sidecar_weight": args.sidecar_weight if not args.disable_sidecar_rerank else 0.0,
        "phrase_weight": args.phrase_weight,
    }
    payload = {"summary": summary, "results": results}
    output_arg = getattr(args, "output_path", None)
    if output_arg:
        output_path = Path(output_arg)
    elif args.shuffle_index:
        output_path = SHUFFLED_MANIFOLD_RESULTS_PATH
    elif args.disable_sidecar_rerank:
        output_path = MANIFOLD_NO_SIDECAR_RESULTS_PATH
    else:
        output_path = MANIFOLD_RESULTS_PATH

    if getattr(args, "write_output", True):
        write_metrics(output_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_manifold(args)
    summary = payload["summary"]
    suffix = " (shuffled)" if summary["shuffle_index"] else ""
    print(
        f"Manifold{suffix}: QA {summary['qa_accuracy']:.3f} | "
        f"Top-1 {summary['retrieval_top1']:.3f} | "
        f"Top-5 {summary['retrieval_top5']:.3f} | "
        f"Sidecar={'on' if summary['sidecar_rerank'] else 'off'}"
    )


if __name__ == "__main__":
    main()
