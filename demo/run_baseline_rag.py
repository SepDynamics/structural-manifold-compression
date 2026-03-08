#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.common import (
    BASELINE_RESULTS_PATH,
    CORPUS_MANIFEST_PATH,
    DEFAULT_OLLAMA_GENERATE_ENDPOINT,
    answer_from_context,
    build_retrieval_query,
    build_chunks,
    build_retrieved_contexts,
    ensure_directories,
    load_manifest,
    load_questions,
    resolve_ollama_model,
    score_prediction,
    write_metrics,
)
from demo.retrieval import (
    build_lexical_index,
    encode_embeddings,
    rank_bm25_chunks,
    rank_documents,
    rank_embedding_chunks,
    rank_hybrid_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline embedding RAG evaluation.")
    parser.add_argument(
        "--retrieval-method",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
        help="Baseline retrieval method.",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Dense retrieval model name.")
    parser.add_argument("--device", default="cpu", help="Device for dense retrieval models.")
    parser.add_argument("--reranker-model", default=None, help="Optional cross-encoder reranker model.")
    parser.add_argument("--reranker-device", default=None, help="Device for the optional cross-encoder reranker.")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal rank fusion constant for hybrid mode.")
    parser.add_argument("--qa-backend", choices=("ollama", "extractive"), default="ollama")
    parser.add_argument("--ollama-endpoint", default=DEFAULT_OLLAMA_GENERATE_ENDPOINT)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--chunk-chars", type=int, default=2200)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def _retrieval_hits(ranked_docs: list[tuple[str, float]], expected: list[str]) -> tuple[bool, bool]:
    doc_ids = [doc_id for doc_id, _ in ranked_docs]
    top1 = bool(doc_ids and doc_ids[0] in expected)
    if len(expected) > 1:
        top5 = set(expected).issubset(set(doc_ids[:5]))
    else:
        top5 = any(doc_id in expected for doc_id in doc_ids[:5])
    return top1, top5


def _rank_baseline_chunks(
    question,
    *,
    args: argparse.Namespace,
    chunks,
    embeddings,
    lexical_index,
):
    retrieval_method = getattr(args, "retrieval_method", "dense")
    device = getattr(args, "device", "cpu")
    reranker_model = getattr(args, "reranker_model", None)
    reranker_device = getattr(args, "reranker_device", None)
    rrf_k = getattr(args, "rrf_k", 60)
    rerank_top_n = getattr(args, "rerank_top_n", 20)
    if retrieval_method == "bm25":
        return rank_bm25_chunks(
            question,
            chunks=chunks,
            lexical_index=lexical_index,
            top_k=args.top_k,
        )
    if retrieval_method == "hybrid":
        return rank_hybrid_chunks(
            question,
            chunks=chunks,
            embeddings=embeddings,
            lexical_index=lexical_index,
            model_name=args.embedding_model,
            device=device,
            top_k=args.top_k,
            rrf_k=rrf_k,
            reranker_model=reranker_model,
            reranker_device=reranker_device,
            rerank_top_n=rerank_top_n,
        )
    return rank_embedding_chunks(
        question,
        chunks=chunks,
        embeddings=embeddings,
        model_name=args.embedding_model,
        device=device,
        top_k=args.top_k,
    )


def _default_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return args.output_path
    retrieval_method = getattr(args, "retrieval_method", "dense")
    reranker_model = getattr(args, "reranker_model", None)
    if retrieval_method == "dense" and args.embedding_model == "all-MiniLM-L6-v2" and not reranker_model:
        return BASELINE_RESULTS_PATH
    return BASELINE_RESULTS_PATH


def run_baseline(args: argparse.Namespace) -> dict[str, object]:
    ensure_directories()
    if not CORPUS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Corpus manifest not found: {CORPUS_MANIFEST_PATH}")
    retrieval_method = getattr(args, "retrieval_method", "dense")
    device = getattr(args, "device", "cpu")
    reranker_model = getattr(args, "reranker_model", None)
    reranker_device = getattr(args, "reranker_device", None)

    questions = load_questions()
    papers = load_manifest()
    chunks = build_chunks(
        papers,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.chunk_overlap,
    )
    embeddings = None
    lexical_index = None
    if retrieval_method in {"dense", "hybrid"}:
        embeddings = encode_embeddings(
            [chunk.text for chunk in chunks],
            model_name=args.embedding_model,
            device=device,
        )
    if retrieval_method in {"bm25", "hybrid"}:
        lexical_index = build_lexical_index(chunks)
    ollama_model = resolve_ollama_model(args.ollama_model) if args.qa_backend == "ollama" else None

    results: list[dict[str, object]] = []
    total_latency = 0.0

    for question in questions:
        started = time.perf_counter()
        retrieval_query = build_retrieval_query(question)
        ranked_chunk_scores = _rank_baseline_chunks(
            question,
            args=args,
            chunks=chunks,
            embeddings=embeddings,
            lexical_index=lexical_index,
        )
        ranked_chunks = [chunks[idx] for idx, _ in ranked_chunk_scores]
        ranked_docs = rank_documents(ranked_chunk_scores, chunks)
        contexts = build_retrieved_contexts(ranked_chunks, max_context_tokens=args.max_context_tokens)
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
                "retrieved_chunks": [chunk.chunk_id for chunk in ranked_chunks[: args.top_k]],
                "retrieval_top1": top1,
                "retrieval_top5": top5,
                "latency_seconds": elapsed,
            }
        )

    question_count = max(1, len(results))
    summary = {
        "system": "baseline_rag",
        "retrieval_method": retrieval_method,
        "qa_backend": args.qa_backend,
        "embedding_model": args.embedding_model if retrieval_method in {"dense", "hybrid"} else None,
        "device": device if retrieval_method in {"dense", "hybrid"} else None,
        "reranker_model": reranker_model,
        "reranker_device": reranker_device if reranker_model else None,
        "questions": len(results),
        "qa_accuracy": sum(1 for row in results if row["correct"]) / question_count,
        "retrieval_top1": sum(1 for row in results if row["retrieval_top1"]) / question_count,
        "retrieval_top5": sum(1 for row in results if row["retrieval_top5"]) / question_count,
        "mean_latency_seconds": total_latency / question_count,
        "max_context_tokens": args.max_context_tokens,
    }
    payload = {"summary": summary, "results": results}
    write_metrics(_default_output_path(args), payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_baseline(args)
    summary = payload["summary"]
    print(
        f"Baseline complete ({summary['retrieval_method']}): QA {summary['qa_accuracy']:.3f} | "
        f"Top-1 {summary['retrieval_top1']:.3f} | "
        f"Top-5 {summary['retrieval_top5']:.3f}"
    )


if __name__ == "__main__":
    main()
