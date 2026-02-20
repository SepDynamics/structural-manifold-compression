#!/usr/bin/env python3
"""RAG precision benchmark: FAISS embeddings vs. manifold structural retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter


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


def build_embedding_index(text_root: Path, model_name: str) -> Tuple[faiss.IndexFlatIP, List[str]]:
    model = SentenceTransformer(model_name)
    docs: List[str] = []
    doc_ids: List[str] = []
    for doc_id, text in iter_text_files(text_root):
        doc_ids.append(doc_id)
        docs.append(text)
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, doc_ids


def evaluate_embedding_recall(index: faiss.IndexFlatIP, doc_ids: List[str], model_name: str, questions: List[Question]) -> Dict[str, float]:
    model = SentenceTransformer(model_name)
    correct = 0
    for q in questions:
        query_vec = model.encode([q.query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = index.search(query_vec, 1)
        top_id = doc_ids[int(indices[0][0])]
        if top_id == q.expected_doc:
            correct += 1
    return {"embedding_recall": correct / max(len(questions), 1)}


def evaluate_manifold_recall(router: TripartiteRouter, questions: List[Question]) -> Dict[str, float]:
    correct = 0
    for q in questions:
        verified, _, _, matched = router.process_query(q.query)
        if verified and matched and matched[0] == q.expected_doc:
            correct += 1
    return {"manifold_recall": correct / max(len(questions), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG precision benchmark: FAISS vs manifold")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output" / "benchmarks" / "rag_precision.json")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    index, doc_ids = build_embedding_index(args.text_root, args.model)

    router = TripartiteRouter()

    start = time.time()
    embedding_stats = evaluate_embedding_recall(index, doc_ids, args.model, questions)
    embedding_stats["embedding_seconds"] = time.time() - start

    start = time.time()
    manifold_stats = evaluate_manifold_recall(router, questions)
    manifold_stats["manifold_seconds"] = time.time() - start

    output = {
        "questions": len(questions),
        **embedding_stats,
        **manifold_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
