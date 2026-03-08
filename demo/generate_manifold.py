#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.common import (
    COMPRESSION_METRICS_PATH,
    CORPUS_MANIFEST_PATH,
    MANIFOLD_INDEX_PATH,
    MANIFOLD_JSON_PATH,
    QUESTIONS_PATH,
    build_chunks,
    corpus_hash,
    ensure_directories,
    estimated_token_count,
    load_manifest,
    utc_now,
    write_metrics,
)
from demo.retrieval import build_manifold_payload, save_manifold_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the structural manifold retrieval index.")
    parser.add_argument("--chunk-chars", type=int, default=2200, help="Chunk size for corpus segmentation.")
    parser.add_argument("--chunk-overlap", type=int, default=250, help="Character overlap between chunks.")
    parser.add_argument("--window-bytes", type=int, default=24, help="Manifold encoder window size in bytes.")
    parser.add_argument("--stride-bytes", type=int, default=12, help="Manifold encoder stride in bytes.")
    parser.add_argument("--precision", type=int, default=2, help="Signature precision.")
    return parser.parse_args()


def generate_manifold(args: argparse.Namespace) -> dict[str, object]:
    ensure_directories()
    if not CORPUS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Corpus manifest not found: {CORPUS_MANIFEST_PATH}")
    if not QUESTIONS_PATH.exists():
        raise RuntimeError("questions.json must exist before manifold generation.")

    papers = load_manifest(CORPUS_MANIFEST_PATH)
    chunks = build_chunks(
        papers,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.chunk_overlap,
    )

    question_hash = hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest()
    corpus_tokens = sum(paper.estimated_tokens for paper in papers)
    metadata, binary_payload = build_manifold_payload(
        chunks,
        question_hash=question_hash,
        corpus_hash=corpus_hash(papers),
        corpus_tokens=corpus_tokens,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
    )
    metadata["created_at"] = utc_now()
    MANIFOLD_JSON_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_manifold_index(MANIFOLD_INDEX_PATH, binary_payload)

    json_bytes = MANIFOLD_JSON_PATH.stat().st_size
    bin_bytes = MANIFOLD_INDEX_PATH.stat().st_size
    metrics = {
        "original_tokens": corpus_tokens,
        "original_bytes": sum(paper.bytes for paper in papers),
        "manifold_signature_tokens": int(metadata["signature_tokens"]),
        "manifold_unique_signatures": int(metadata["unique_signatures"]),
        "manifold_serialized_bytes": json_bytes + bin_bytes,
        "manifold_serialized_tokens_estimate": estimated_token_count(MANIFOLD_JSON_PATH.read_text(encoding="utf-8")),
        "compression_ratio_signature_tokens": float(metadata["compression_ratio"]),
        "compression_ratio_serialized_bytes": (
            sum(paper.bytes for paper in papers) / (json_bytes + bin_bytes)
            if (json_bytes + bin_bytes)
            else float("inf")
        ),
        "chunk_count": len(chunks),
        "question_hash": question_hash,
        "window_bytes": args.window_bytes,
        "stride_bytes": args.stride_bytes,
        "precision": args.precision,
    }
    write_metrics(COMPRESSION_METRICS_PATH, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = generate_manifold(args)
    print(
        f"Manifold ready: {metrics['manifold_signature_tokens']} signature tokens | "
        f"{metrics['compression_ratio_signature_tokens']:.2f}x token compression"
    )


if __name__ == "__main__":
    main()
