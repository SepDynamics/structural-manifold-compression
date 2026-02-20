#!/usr/bin/env python3
"""RAG precision benchmark: FAISS embeddings vs. manifold structural retrieval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", message="to_int32")

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from src.manifold.sidecar import build_index, verify_snippet


@dataclass
class Question:
    query: str
    expected_doc: str


def load_questions(path: Path) -> List[Question]:
    questions: List[Question] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            questions.append(Question(query=record["query"], expected_doc=record["expected_doc"]))
    return questions


def iter_text_files(text_root: Path) -> Iterable[Tuple[str, str]]:
    for path in sorted(text_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        doc_id = path.relative_to(text_root).with_suffix("").as_posix().replace("/", "__")
        yield doc_id, path.read_text(encoding="utf-8")


def build_embedding_index(text_root: Path, model: SentenceTransformer) -> Tuple[faiss.IndexFlatIP, List[str]]:
    docs: List[str] = []
    doc_ids: List[str] = []
    for doc_id, text in iter_text_files(text_root):
        doc_ids.append(doc_id)
        docs.append(text)
    print("Encoding documents for FAISS...")
    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, doc_ids


def evaluate_embedding_recall(index: faiss.IndexFlatIP, doc_ids: List[str], model: SentenceTransformer, questions: List[Question]) -> Dict[str, float]:
    correct = 0
    print("Evaluating FAISS Semantic Embeddings...")
    for q in tqdm(questions, desc="FAISS Recall"):
        query_vec = model.encode([q.query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = index.search(query_vec, 1)
        top_id = doc_ids[int(indices[0][0])]
        if top_id == q.expected_doc:
            correct += 1
    return {"embedding_recall": correct / max(len(questions), 1)}


def evaluate_manifold_recall(index, questions: List[Question], window_bytes: int, stride_bytes: int) -> Dict[str, float]:
    correct = 0
    print("Evaluating AGI-Lite Manifold Structural Engine...")
    for q in tqdm(questions, desc="Manifold Recall"):
        result = verify_snippet(
            text=q.query,
            index=index,
            hazard_threshold=1.0,
            coverage_threshold=0.0,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=3,
            use_native=True,
        )
        matched = result.matched_documents
        if matched and q.expected_doc in str(matched):
            correct += 1
    return {"manifold_recall": correct / max(len(questions), 1)}


def build_manifold_docs(text_root: Path) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for doc_id, text in iter_text_files(text_root):
        docs[doc_id] = text
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG precision benchmark: FAISS vs manifold")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output" / "benchmarks" / "rag_precision.json")
    args = parser.parse_args()

    questions = load_questions(args.questions)

    print(f"Loading SentenceTransformer model: {args.model}...")
    model = SentenceTransformer(args.model)

    faiss_index, doc_ids = build_embedding_index(args.text_root, model)

    docs = build_manifold_docs(args.text_root)
    min_query_bytes = min(len(q.query.encode("utf-8")) for q in questions)
    window_bytes = max(16, min_query_bytes)
    stride_bytes = max(8, window_bytes // 2)
    manifold_index = build_index(
        docs,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=3,
        use_native=True,
    )

    start = time.time()
    embedding_stats = evaluate_embedding_recall(faiss_index, doc_ids, model, questions)
    embedding_stats["embedding_seconds"] = time.time() - start

    start = time.time()
    manifold_stats = evaluate_manifold_recall(manifold_index, questions, window_bytes, stride_bytes)
    manifold_stats["manifold_seconds"] = time.time() - start

    output = {
        "questions": len(questions),
        **embedding_stats,
        **manifold_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== Benchmark Complete ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
