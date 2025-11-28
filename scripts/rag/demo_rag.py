#!/usr/bin/env python3
"""Naive vs manifold-verified RAG demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from manifold.sidecar import build_index, load_index, verify_snippet  # noqa: E402


def load_corpus(path: Path) -> List[Tuple[str, str]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = str(data.get("doc_id", f"doc_{idx:06d}"))
            text = str(data.get("text", ""))
            records.append((doc_id, text))
    if not records:
        raise ValueError(f"No documents found in {path}")
    return records


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for this demo. Install with `pip install sentence-transformers`."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def top_k(query: np.ndarray, passages: np.ndarray, k: int) -> List[int]:
    scores = passages @ query
    order = np.argsort(scores)[::-1]
    return list(order[:k])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate naive vs manifold-verified RAG.")
    parser.add_argument("--index", type=Path, help="Path to a prebuilt manifold index (optional).")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL corpus used to build the index.")
    parser.add_argument("--question", type=str, required=True, help="Question/query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Character chunk size for retrieval.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Character overlap between chunks.")
    parser.add_argument("--window-bytes", type=int, default=512, help="Manifold window bytes.")
    parser.add_argument("--stride-bytes", type=int, default=384, help="Manifold stride bytes.")
    parser.add_argument("--precision", type=int, default=3, help="Manifold signature precision.")
    parser.add_argument("--hazard-threshold", type=float, help="Override hazard gate.")
    parser.add_argument("--coverage-threshold", type=float, default=0.5, help="Coverage needed to keep a chunk.")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model to use for embeddings.",
    )
    return parser.parse_args()


def build_manifold_index_from_corpus(
    corpus: List[Tuple[str, str]], window_bytes: int, stride_bytes: int, precision: int
):
    docs: Dict[str, str] = {doc_id: text for doc_id, text in corpus}
    return build_index(
        docs,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        hazard_percentile=0.8,
    )


def main() -> None:
    args = parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    corpus = load_corpus(dataset)

    passages: List[Tuple[str, str, str]] = []
    for doc_id, text in corpus:
        for idx, chunk in enumerate(chunk_text(text, args.chunk_size, args.chunk_overlap)):
            passages.append((doc_id, f"{doc_id}#chunk={idx}", chunk))

    passage_texts = [p[2] for p in passages]
    passage_embeddings = embed_texts(passage_texts, args.embedding_model)
    question_embedding = embed_texts([args.question], args.embedding_model)[0]

    top_indices = top_k(question_embedding, passage_embeddings, args.top_k)

    if args.index:
        index = load_index(args.index.expanduser().resolve())
    else:
        index = build_manifold_index_from_corpus(
            corpus,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=args.precision,
        )

    print("\n=== Naive RAG ===")
    for rank, idx in enumerate(top_indices, start=1):
        doc_id, chunk_id, text = passages[idx]
        score = float(passage_embeddings[idx] @ question_embedding)
        print(f"[{rank}] {chunk_id} (doc={doc_id}, score={score:.4f})")
        print(text.strip())
        print("---")

    print("\n=== Manifold-verified RAG ===")
    verified_any = False
    for rank, idx in enumerate(top_indices, start=1):
        doc_id, chunk_id, text = passages[idx]
        result = verify_snippet(
            text,
            index,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=args.precision,
            hazard_threshold=args.hazard_threshold,
            coverage_threshold=args.coverage_threshold,
        )
        if not result.verified:
            continue
        verified_any = True
        score = float(passage_embeddings[idx] @ question_embedding)
        print(f"[{rank}] {chunk_id} (doc={doc_id}, score={score:.4f})")
        print(f"coverage={result.coverage:.2f}, hazard<= {result.hazard_threshold:.4f}")
        print(text.strip())
        print("---")

    if not verified_any:
        print("No chunks passed hazard-gated verification at the chosen thresholds.")


if __name__ == "__main__":
    main()
