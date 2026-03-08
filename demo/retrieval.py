from __future__ import annotations

import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from demo.common import (
    ChunkRecord,
    QuestionRecord,
    STOPWORDS,
    WORD_RE,
    build_retrieval_query,
    estimated_token_count,
    iter_retrieval_query_texts,
)
from demo.structure import (
    StructuralNodeRecord,
    build_shuffle_node_indices,
    node_retrieval_text,
    node_structural_text,
    serialize_nodes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from manifold.sidecar import encode_text


@dataclass(slots=True)
class LexicalIndex:
    token_counts: list[Counter[str]]
    document_frequency: Counter[str]
    doc_lengths: list[int]
    avg_doc_length: float


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


@lru_cache(maxsize=4)
def get_embedder(model_name: str, device: str = "cpu"):
    if model_name == "hash":
        return HashingEmbedder()
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name, device=device)


def encode_embeddings(texts: Sequence[str], *, model_name: str, device: str = "cpu") -> np.ndarray:
    embedder = get_embedder(model_name, device)
    if isinstance(embedder, HashingEmbedder):
        return embedder.encode(texts)
    embeddings = embedder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return embeddings.astype(np.float32)


def lexical_tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.lower() for match in WORD_RE.findall(text))
        if token not in STOPWORDS
    ]


def build_lexical_index(chunks: Sequence[ChunkRecord]) -> LexicalIndex:
    token_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    doc_lengths: list[int] = []
    for chunk in chunks:
        tokens = lexical_tokens(chunk.text)
        counts = Counter(tokens)
        token_counts.append(counts)
        doc_lengths.append(sum(counts.values()))
        document_frequency.update(counts.keys())
    avg_doc_length = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    return LexicalIndex(
        token_counts=token_counts,
        document_frequency=document_frequency,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
    )


def search_embedding_index(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    scores = embeddings @ query_embedding
    order = np.argsort(scores)[::-1]
    return [int(idx) for idx in order[:top_k]]


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str, device: str = "cpu"):
    from sentence_transformers import CrossEncoder  # type: ignore

    return CrossEncoder(model_name, device=device)


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
    nodes: Sequence[StructuralNodeRecord],
    *,
    question_hash: str,
    corpus_hash: str,
    corpus_tokens: int,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    embedding_model: str,
) -> tuple[dict[str, object], dict[str, object]]:
    node_signature_counters: list[Counter[str]] = []
    unique_signatures = set()
    sidecar_signature_tokens = 0
    structural_tokens = 0
    retrieval_texts = [node_retrieval_text(node) for node in nodes]
    embeddings = (
        encode_embeddings(retrieval_texts, model_name=embedding_model)
        if nodes
        else np.zeros((0, 0), dtype=np.float32)
    )

    for node in nodes:
        structural_text = node_structural_text(node)
        counts = _encode_signature_counter(
            structural_text,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
        )
        node_signature_counters.append(counts)
        sidecar_signature_tokens += sum(counts.values())
        structural_tokens += estimated_token_count(structural_text)
        unique_signatures.update(counts.keys())

    if embeddings.size:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = (embeddings / norms).astype(np.float32)

    metadata = {
        "format_version": 2,
        "created_at": None,
        "question_hash": question_hash,
        "corpus_hash": corpus_hash,
        "corpus_tokens": corpus_tokens,
        "node_count": len(nodes),
        "window_bytes": window_bytes,
        "stride_bytes": stride_bytes,
        "precision": precision,
        "embedding_model": embedding_model,
        "signature_tokens": sidecar_signature_tokens,
        "structural_tokens": structural_tokens,
        "unique_signatures": len(unique_signatures),
        "compression_ratio": (corpus_tokens / structural_tokens) if structural_tokens else float("inf"),
        "nodes": serialize_nodes(nodes),
    }

    binary_payload = {
        "embeddings": embeddings,
        "signature_counters": node_signature_counters,
        "shuffle_indices": build_shuffle_node_indices(
            nodes,
            seed_text=f"{corpus_hash}:{question_hash}",
        ),
    }
    return metadata, binary_payload


def save_manifold_index(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_manifold_index(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _query_signature_counter(
    question: QuestionRecord,
    *,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
) -> Counter[str]:
    query_stride_bytes = 1 if question.evidence_terms else max(1, stride_bytes // 2)
    query_counts: Counter[str] = Counter()
    for query_text in iter_retrieval_query_texts(question):
        query_counts.update(
            _encode_signature_counter(
                query_text,
                window_bytes=window_bytes,
                stride_bytes=query_stride_bytes,
                precision=precision,
            )
        )
    return query_counts


def _signature_overlap_score(
    query_counts: Counter[str],
    node_counts: Counter[str],
) -> float:
    total = sum(query_counts.values())
    if not total:
        return 0.0
    overlap = sum(min(count, node_counts.get(signature, 0)) for signature, count in query_counts.items())
    return overlap / total


def _phrase_overlap_score(question: QuestionRecord, node: StructuralNodeRecord) -> float:
    query_phrases = [phrase.lower() for phrase in question.evidence_terms if phrase]
    if not query_phrases:
        return 0.0
    searchable = " ".join([node.heading, node.section_type, *node.salient_phrases, node.text_sketch]).lower()
    hits = sum(1 for phrase in query_phrases if phrase in searchable)
    return hits / max(1, len(query_phrases))


def rank_manifold_nodes_detailed(
    question: QuestionRecord,
    *,
    nodes: Sequence[StructuralNodeRecord],
    index_payload: dict[str, object],
    embedding_model: str,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    top_k: int,
    shuffle_nodes: bool = False,
    use_sidecar_rerank: bool = True,
    sidecar_weight: float = 0.25,
    phrase_weight: float = 0.15,
) -> list[dict[str, float | int | bool]]:
    embeddings = index_payload["embeddings"]
    if not len(nodes) or embeddings.size == 0:
        return []

    query_embedding = encode_embeddings([build_retrieval_query(question)], model_name=embedding_model)[0]
    query_counts = _query_signature_counter(
        question,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
    )
    candidate_count = min(len(nodes), max(top_k * 8, 24))
    candidate_indices = search_embedding_index(embeddings, query_embedding, top_k=candidate_count)
    signature_counters = index_payload["signature_counters"]

    ranking: list[dict[str, float | int | bool]] = []
    for node_idx in candidate_indices:
        embedding_score = float(embeddings[node_idx] @ query_embedding)
        sidecar_score = _signature_overlap_score(query_counts, signature_counters[node_idx])
        phrase_score = _phrase_overlap_score(question, nodes[node_idx])
        applied_sidecar_weight = sidecar_weight if use_sidecar_rerank else 0.0
        final_score = embedding_score + (applied_sidecar_weight * sidecar_score) + (phrase_weight * phrase_score)
        ranking.append(
            {
                "node_index": int(node_idx),
                "score": final_score,
                "embedding_score": embedding_score,
                "sidecar_score": sidecar_score,
                "phrase_score": phrase_score,
                "verified": bool((use_sidecar_rerank and sidecar_score >= 0.15) or phrase_score >= 0.5),
                "used_sidecar_rerank": use_sidecar_rerank,
            }
        )

    ranking.sort(key=lambda item: float(item["score"]), reverse=True)

    if shuffle_nodes and ranking:
        shuffled_indices = index_payload["shuffle_indices"]
        remapped: list[dict[str, float | int | bool]] = []
        for item in ranking:
            shuffled_idx = int(shuffled_indices[int(item["node_index"])])
            remapped.append(
                {
                    **item,
                    "node_index": shuffled_idx,
                }
            )
        ranking = remapped

    return ranking[:top_k]


def rank_manifold_chunks(
    question: QuestionRecord,
    *,
    nodes: Sequence[StructuralNodeRecord],
    index_payload: dict[str, object],
    embedding_model: str,
    window_bytes: int,
    stride_bytes: int,
    precision: int,
    top_k: int,
    shuffle_postings: bool = False,
    use_sidecar_rerank: bool = True,
    sidecar_weight: float = 0.25,
    phrase_weight: float = 0.15,
) -> list[tuple[int, float]]:
    ranked = rank_manifold_nodes_detailed(
        question,
        nodes=nodes,
        index_payload=index_payload,
        embedding_model=embedding_model,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        top_k=top_k,
        shuffle_nodes=shuffle_postings,
        use_sidecar_rerank=use_sidecar_rerank,
        sidecar_weight=sidecar_weight,
        phrase_weight=phrase_weight,
    )
    return [(int(item["node_index"]), float(item["score"])) for item in ranked]


def rank_embedding_chunks(
    question: QuestionRecord,
    *,
    chunks: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    model_name: str,
    device: str = "cpu",
    top_k: int,
) -> list[tuple[int, float]]:
    query_embedding = encode_embeddings([build_retrieval_query(question)], model_name=model_name, device=device)[0]
    indices = search_embedding_index(embeddings, query_embedding, top_k=top_k)
    scores = embeddings[indices] @ query_embedding
    return [(int(idx), float(score)) for idx, score in zip(indices, scores, strict=False)]


def rank_bm25_chunks(
    question: QuestionRecord,
    *,
    chunks: Sequence[ChunkRecord],
    lexical_index: LexicalIndex,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    query_counts = Counter(lexical_tokens(build_retrieval_query(question)))
    if not query_counts:
        return []

    scores: list[tuple[int, float]] = []
    doc_count = max(1, len(chunks))
    avg_doc_len = lexical_index.avg_doc_length or 1.0

    for idx, counts in enumerate(lexical_index.token_counts):
        doc_len = lexical_index.doc_lengths[idx] or 1
        score = 0.0
        for token, query_tf in query_counts.items():
            tf = counts.get(token, 0)
            if not tf:
                continue
            df = lexical_index.document_frequency.get(token, 0)
            idf = math.log(1.0 + ((doc_count - df + 0.5) / (df + 0.5)))
            norm = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            score += idf * ((tf * (k1 + 1.0)) / norm) * query_tf
        if score > 0.0:
            scores.append((idx, float(score)))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[int, float]]],
    *,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[int(idx)] += 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def rerank_chunks_with_cross_encoder(
    question: QuestionRecord,
    *,
    chunks: Sequence[ChunkRecord],
    ranked_chunks: Sequence[tuple[int, float]],
    model_name: str,
    device: str = "cpu",
    top_k: int,
) -> list[tuple[int, float]]:
    if not ranked_chunks:
        return []
    reranker = get_cross_encoder(model_name, device)
    pairs = [(build_retrieval_query(question), chunks[idx].text) for idx, _ in ranked_chunks]
    scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=np.float32).reshape(-1)
    reranked = [
        (int(idx), float(score))
        for (idx, _), score in zip(ranked_chunks, scores, strict=False)
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:top_k]


def rank_hybrid_chunks(
    question: QuestionRecord,
    *,
    chunks: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    lexical_index: LexicalIndex,
    model_name: str,
    device: str = "cpu",
    top_k: int,
    dense_candidates: int | None = None,
    lexical_candidates: int | None = None,
    rrf_k: int = 60,
    reranker_model: str | None = None,
    reranker_device: str | None = None,
    rerank_top_n: int = 20,
) -> list[tuple[int, float]]:
    dense_top_k = dense_candidates or max(top_k * 8, 24)
    lexical_top_k = lexical_candidates or max(top_k * 8, 24)
    dense_ranked = rank_embedding_chunks(
        question,
        chunks=chunks,
        embeddings=embeddings,
        model_name=model_name,
        device=device,
        top_k=dense_top_k,
    )
    lexical_ranked = rank_bm25_chunks(
        question,
        chunks=chunks,
        lexical_index=lexical_index,
        top_k=lexical_top_k,
    )
    fused = reciprocal_rank_fusion((dense_ranked, lexical_ranked), rrf_k=rrf_k)
    if reranker_model:
        rerank_pool = fused[: max(top_k, rerank_top_n)]
        reranked = rerank_chunks_with_cross_encoder(
            question,
            chunks=chunks,
            ranked_chunks=rerank_pool,
            model_name=reranker_model,
            device=reranker_device or device,
            top_k=max(top_k, rerank_top_n),
        )
        remaining = {idx for idx, _ in reranked}
        tail = [item for item in fused if item[0] not in remaining]
        fused = reranked + tail
    return fused[:top_k]


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
