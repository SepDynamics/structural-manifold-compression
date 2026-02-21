#!/usr/bin/env python3
"""Bulk Ingest a JSONL text corpus directly into the live Valkey database.

This script aggressively pipelines structural manifold generation through
the C++ `sep_quantum` bindings and into the local `sep-valkey` associative memory.
It acts as the "Highway Stress Test" for the Tripartite Daemon architecture.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from manifold.valkey_client import ValkeyWorkingMemory
from manifold.sidecar import encode_text
from scripts.experiments.manifold_compression_eval import iter_text_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk Ingest Corpus to Valkey")
    parser.add_argument(
        "--input-dir",
        "--dataset",
        type=Path,
        required=True,
        dest="dataset",
        help="Path to text root or JSONL file",
    )
    parser.add_argument(
        "--use-native", action="store_true", help="Use native C++ encoder"
    )
    parser.add_argument(
        "--json-text-key", type=str, default="text", help="JSON key containing text"
    )
    parser.add_argument("--max-documents", type=int, help="Optional cap on documents")
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Wipe all data from Valkey before starting",
    )
    parser.add_argument(
        "--window-bytes", type=int, default=512, help="Manifold Sliding Window Size"
    )
    parser.add_argument(
        "--stride-bytes", type=int, default=384, help="Manifold Sliding Window Stride"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    print(f"Connecting to live Valkey instance on localhost:6379...")
    valkey = ValkeyWorkingMemory()
    if not valkey.ping():
        raise RuntimeError(
            "Could not connect to Valkey! Make sure 'docker run -p 6379:6379 valkey/valkey' is running."
        )

    if args.clear_first:
        print("Wiping existing Valkey working memory...")
        valkey.clear_all()
        print("Valkey wiped.")

    print(f"Commencing Bulk Ingestion of {dataset}")
    print(
        f"Configurations: | Window Size: {args.window_bytes} | Stride: {args.stride_bytes}"
    )

    docs_processed = 0
    total_bytes = 0
    total_windows_encoded = 0
    start_time = time.time()

    # Pre-fetch the cached index to aggressively mutate and push it
    index = valkey.get_cached_index()
    if index is None:
        from manifold.sidecar import ManifoldIndex

        index = ManifoldIndex(
            meta={
                "window_bytes": args.window_bytes,
                "stride_bytes": args.stride_bytes,
                "precision": 3,
                "hazard_percentile": 0.8,
            },
            signatures={},
            documents={},
        )

    meta = index.meta
    signatures = index.signatures
    documents = index.documents

    def log_progress():
        elapsed = time.time() - start_time
        mbps = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
        wps = total_windows_encoded / elapsed if elapsed > 0 else 0
        print(
            f"[{elapsed:.1f}s] Docs: {docs_processed} | MBytes: {total_bytes/1024/1024:.2f} ({mbps:.2f} MB/s) | "
            f"Windows: {total_windows_encoded} ({wps:.0f} win/s) | "
            f"Unique Nodes: {len(signatures)}"
        )

    for doc_id, text in iter_text_documents(dataset, json_text_key=args.json_text_key):
        if args.max_documents is not None and docs_processed >= args.max_documents:
            break

        byte_len = len(text.encode("utf-8"))
        total_bytes += byte_len

        # O(N) Native Signature Generation
        encoded = encode_text(
            text,
            window_bytes=args.window_bytes,
            stride_bytes=args.stride_bytes,
            precision=3,
            use_native=args.use_native,
            hazard_percentile=0.8,
        )

        # We don't need to push the raw text to Valkey if we just want to stress test the structural index graph
        # valkey.add_document(doc_id, text)  # Optional, but consumes more RAM. We will skip for max speed.

        total_windows_encoded += len(encoded.windows)

        # Accumulate the associative graph
        documents[doc_id] = {
            "characters": len(text),
            "bytes": byte_len,
            "window_count": len(encoded.windows),
        }

        for window in encoded.windows:
            sig = window.signature
            entry = signatures.get(sig)
            if entry is None:
                entry = {
                    "prototype": {
                        "text": encoded.prototypes[sig],
                        "doc_id": doc_id,
                        "byte_start": window.byte_start,
                        "byte_end": window.byte_end,
                        "char_start": window.char_start,
                        "char_end": window.char_end,
                    },
                    "occurrences": [],
                    "hazard": {
                        "min": window.hazard,
                        "max": window.hazard,
                        "sum": 0.0,
                        "count": 0,
                    },
                }
                signatures[sig] = entry

            stats = entry["hazard"]
            stats["count"] += 1
            stats["sum"] += window.hazard
            stats["min"] = min(stats["min"], window.hazard)
            stats["max"] = max(stats["max"], window.hazard)

            entry["occurrences"].append(
                {
                    "doc_id": doc_id,
                    "byte_start": window.byte_start,
                    "byte_end": window.byte_end,
                    "char_start": window.char_start,
                    "char_end": window.char_end,
                    "hazard": window.hazard,
                    "window_index": window.window_index,
                }
            )

        docs_processed += 1

        if docs_processed % 100 == 0:
            log_progress()
            # Push periodic state back to Valkey
            meta["total_signatures"] = len(signatures)
            meta["total_windows"] = total_windows_encoded
            meta["documents"] = len(documents)
            index.meta = meta
            index.signatures = signatures
            index.documents = documents
            valkey.store_cached_index(index)

    # Finalize means
    for entry in signatures.values():
        haz = entry["hazard"]
        count = haz["count"]
        haz["mean"] = haz["sum"] / count if count else 0.0

    meta["total_signatures"] = len(signatures)
    meta["total_windows"] = total_windows_encoded
    meta["documents"] = len(documents)

    print("\n--- INGESTION COMPLETE ---")
    log_progress()
    print("Saving final master graph to Valkey...")
    valkey.store_cached_index(index)
    print("Done! Valkey is successfully saturated.")


if __name__ == "__main__":
    main()
