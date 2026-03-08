from __future__ import annotations

import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from demo.common import ChunkRecord, QuestionRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from manifold.sidecar import encode_text


class HashingEmbedder:
    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                idx = hash(token) % self.dimensions
                matrix[row, idx] += 1.0
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


def get_embedder(model_name: str):
    if model_name == "hash":
        return HashingEmbedder()
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


def encode_embeddings(texts: Sequence[str], *, model_name: str) -> np.ndarray:
    embedder = get_embedder(model_name)
    if isinstance(embedder, HashingEmbedder):
        return embedder.encode(texts)
    embeddings = embedder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return embeddings.astype(np.float32)


def search_embedding_index(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    scores = embeddings @ query_embedding
    order = np.argsort(scores)[::-1]
    return [int(idx) for idx in order[:top_k]]


def aggregate_document_scores(
    chunk_indices: Sequence[int],
    chunk_scores: Sequence[float],
    chunks: Sequence[ChunkRecord],
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for idx, score in zip(chunk_indices, chunk_scores, strict=False):
        scores[chunks[idx].paper_id] = max(scores[chunks[idx].paper_id], float(score))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _idf(document_frequency: int, document_count: int) -> float:
    return math.log((1.0 + document_count) / (1.0 + document_frequency)) + 1.0


def _encode_signature_counter(
    text: str,
    *,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
) -> Counter[str]:
    encoded = encode_text(
        text,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
    )
    return Counter(window.signature for window in encoded.windows)


def build_manifold_payload(
    chunks: Sequence[ChunkRecord],
    *,
    question_hash: str,
    corpus_hash: str,
    corpus_tokens: int,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
) -> tuple[dict[str, object], dict[str, object]]:
    chunk_term_counts: list[Counter[str]] = []
    signature_df: Counter[str] = Counter()
    total_signature_tokens = 0
    unique_signatures = set()

    for chunk in chunks:
        counts = _encode_signature_counter(
            chunk.text,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
        )
        chunk_term_counts.append(counts)
        signature_df.update(counts.keys())
        total_signature_tokens += sum(counts.values())
        unique_signatures.update(counts.keys())

    document_count = len(chunks)
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    chunk_norms: list[float] = []

    chunk_entries: list[dict[str, object]] = []
    for chunk_idx, (chunk, counts) in enumerate(zip(chunks, chunk_term_counts, strict=False)):
        weights: dict[str, float] = {}
        for signature, count in counts.items():
            idf = _idf(signature_df[signature], document_count)
            weight = (1.0 + math.log(float(count))) * idf
            weights[signature] = weight
            postings[signature].append((chunk_idx, weight))
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        chunk_norms.append(norm)
        chunk_entries.append(
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "text_path": chunk.text_path,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "estimated_tokens": chunk.estimated_tokens,
                "signature_count": int(sum(counts.values())),
                "unique_signature_count": len(counts),
            }
        )

    metadata = {
        "format_version": 1,
        "created_at": None,
        "question_hash": question_hash,
        "corpus_hash": corpus_hash,
        "corpus_tokens": corpus_tokens,
        "chunk_count": len(chunks),
        "window_bytes": window_bytes,
        "stride_bytes": stride_bytes,
        "precision": precision,
        "signature_tokens": total_signature_tokens,
        "unique_signatures": len(unique_signatures),
        "compression_ratio": (corpus_tokens / total_signature_tokens) if total_signature_tokens else float("inf"),
        "chunks": chunk_entries,
    }

    binary_payload = {
        "chunk_norms": chunk_norms,
        "postings": dict(postings),
        "signature_df": dict(signature_df),
    }
    return metadata, binary_payload


def save_manifold_index(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_manifold_index(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def rank_manifold_chunks(
    question: QuestionRecord,
    *,
    chunks: Sequence[ChunkRecord],
    index_payload: dict[str, object],
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    top_k: int,
    shuffle_postings: bool = False,
) -> list[tuple[int, float]]:
    signature_df = index_payload["signature_df"]
    chunk_norms = index_payload["chunk_norms"]
    postings = index_payload["postings"]
    chunk_count = len(chunks)

    query_counts = _encode_signature_counter(
        question.question,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
    )
    for evidence in question.evidence_terms:
        query_counts.update(
            _encode_signature_counter(
                evidence,
                window_bytes=window_bytes,
                stride_bytes=stride_bytes,
                precision=precision,
            )
        )

    query_weights: dict[str, float] = {}
    for signature, count in query_counts.items():
        idf = _idf(int(signature_df.get(signature, 0)), chunk_count)
        query_weights[signature] = (1.0 + math.log(float(count))) * idf

    query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values())) or 1.0
    scores: dict[int, float] = defaultdict(float)

    for signature, query_weight in query_weights.items():
        posting_list = list(postings.get(signature, []))
        if shuffle_postings and posting_list:
            rotated = posting_list[1:] + posting_list[:1]
            posting_list = rotated
        for chunk_idx, chunk_weight in posting_list:
            scores[int(chunk_idx)] += float(query_weight) * float(chunk_weight)

    ranked = [
        (chunk_idx, score / (query_norm * float(chunk_norms[chunk_idx])))
        for chunk_idx, score in scores.items()
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def rank_embedding_chunks(
    question: QuestionRecord,
    *,
    chunks: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    model_name: str,
    top_k: int,
) -> list[tuple[int, float]]:
    query_embedding = encode_embeddings([question.question], model_name=model_name)[0]
    indices = search_embedding_index(embeddings, query_embedding, top_k=top_k)
    scores = embeddings[indices] @ query_embedding
    return [(int(idx), float(score)) for idx, score in zip(indices, scores, strict=False)]


def rank_documents(
    ranked_chunks: Sequence[tuple[int, float]],
    chunks: Sequence[ChunkRecord],
) -> list[tuple[str, float]]:
    chunk_indices = [chunk_idx for chunk_idx, _ in ranked_chunks]
    chunk_scores = [score for _, score in ranked_chunks]
    return aggregate_document_scores(chunk_indices, chunk_scores, chunks)


def chunk_lookup(chunks: Iterable[ChunkRecord]) -> dict[str, ChunkRecord]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def serialize_chunks(chunks: Sequence[ChunkRecord]) -> list[dict[str, object]]:
    return [asdict(chunk) for chunk in chunks]
