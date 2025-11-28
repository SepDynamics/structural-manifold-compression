#!/usr/bin/env python3
"""Build a structural manifold index for downstream RAG verification.

Schema (JSON):
- format_version: str (currently "1")
- hazard_threshold: float (80th percentile by default)
- meta: run configuration + aggregate stats
- signatures: signature -> {prototype, occurrences, hazard stats}
- documents: doc_id -> {bytes, characters, window_count, windows?}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from manifold.sidecar import build_manifold_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifold index over a text corpus.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to text root or JSON/JSONL file.")
    parser.add_argument(
        "--json-text-key",
        type=str,
        default="text",
        help="Field to read when ingesting JSON/JSONL corpora (default: text).",
    )
    parser.add_argument("--window-bytes", type=int, default=512, help="Sliding window size in bytes.")
    parser.add_argument("--stride-bytes", type=int, default=384, help="Sliding window stride in bytes.")
    parser.add_argument("--precision", type=int, default=3, help="Signature precision (decimal places).")
    parser.add_argument(
        "--hazard-percentile",
        type=float,
        default=0.8,
        help="Quantile used to derive the hazard gate (default: 0.8).",
    )
    parser.add_argument("--max-documents", type=int, help="Optional cap on number of documents to process.")
    parser.add_argument("--document-offset", type=int, default=0, help="Skip the first N documents before processing.")
    parser.add_argument("--use-native", action="store_true", help="Prefer the native manifold kernel if available.")
    parser.add_argument(
        "--omit-windows",
        action="store_true",
        help="Do not store per-document window streams (shrinks the index).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "manifold_index" / "index.json",
        help="Destination JSON path (default: output/manifold_index/index.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    index = build_manifold_index(
        text_root=dataset,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
        hazard_percentile=args.hazard_percentile,
        json_text_key=args.json_text_key,
        max_documents=args.max_documents,
        document_offset=args.document_offset,
        use_native=args.use_native,
        store_windows=not args.omit_windows,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2), encoding="utf-8")

    meta = index.get("meta", {})
    print(
        f"Indexed {meta.get('documents', 0)} documents "
        f"-> {meta.get('total_signatures', 0)} signatures | "
        f"hazard gate <= {meta.get('hazard_threshold', 0.0):.4f} | "
        f"saved to {args.output}"
    )


if __name__ == "__main__":
    main()
