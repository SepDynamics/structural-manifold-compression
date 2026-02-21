#!/usr/bin/env python3
"""Scale Dynamic Codebook

Builds a massive Latent Semantic Adapter mapping from a large corpus by generating
continuous manifold signatures and pairing them with their localized semantic tokens.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure score/src is accessible for the native bindings if required
SCORE_CANDIDATES = [
    REPO_ROOT / "score" / "src",
    REPO_ROOT.parent / "score" / "src",
    REPO_ROOT / "src",  # For python fallback wrappers
]
for candidate in SCORE_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sep_text_manifold import encode, native
from scripts.experiments.manifold_compression_eval import (
    iter_text_documents,
    sliding_windows,
)
from scripts.inference.dynamic_codebook import DynamicCodebook


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric semantic tokenization."""
    # Find all word-like tokens
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if len(t) > 2]  # Keep meaningful words


def main():
    parser = argparse.ArgumentParser(
        description="Scale dynamic codebook on massive corpus"
    )
    parser.add_argument(
        "--text-root",
        type=Path,
        required=True,
        help="Path to text corpus (JSONL or raw txt)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the generated JSON codebook",
    )
    parser.add_argument(
        "--window-bytes", type=int, default=512, help="Sliding window size"
    )
    parser.add_argument(
        "--stride-bytes", type=int, default=384, help="Sliding window stride"
    )
    parser.add_argument("--precision", type=int, default=3, help="Signature precision")
    parser.add_argument(
        "--max-documents", type=int, default=None, help="Optional max docs to process"
    )
    parser.add_argument(
        "--json-text-key", type=str, default="text", help="Key for JSONL text field"
    )
    parser.add_argument(
        "--use-native", action="store_true", help="Use native C++ engine"
    )

    args = parser.parse_args()

    if args.use_native:
        native.set_use_native(True)
        print("Using native C++ manifold encoder.")

    output_file = args.output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    codebook = DynamicCodebook(window_size=args.window_bytes)

    docs_processed = 0
    global_pos = 0
    start_time = time.time()

    print(f"Starting codebook generation from {args.text_root}")

    for doc_id, text in iter_text_documents(
        args.text_root, json_text_key=args.json_text_key
    ):
        if args.max_documents and docs_processed >= args.max_documents:
            break

        docs_processed += 1
        text_bytes = text.encode("utf-8")

        # We need to map bytes -> Signature, and bytes -> Tokens
        doc_signatures = []
        for offset, chunk in sliding_windows(
            text_bytes, args.window_bytes, args.stride_bytes
        ):
            chunk_bytes = bytes(chunk)

            # 1. Compute Manifold Signature
            if args.use_native and hasattr(native, "analyze_window_batch"):
                metrics = native.analyze_window_batch([chunk_bytes])[0]
            else:
                metrics = encode.encode_window(chunk_bytes)

            signature = encode.signature_from_metrics(
                metrics["coherence"],
                metrics["stability"],
                metrics["entropy"],
                precision=args.precision,
            )
            doc_signatures.append(signature)

            # 2. Extract Semantic Tokens for this specific window
            try:
                # Ignore strict decoding errors at window boundaries
                chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
                semantic_tokens = set(tokenize_text(chunk_str))

                # 3. Update the Latent Semantic Adapter mapping
                for token in semantic_tokens:
                    codebook.update(signature, token, global_pos)

            except Exception as e:
                # Failsafe for complete decoding failure
                pass

            global_pos += 1

        # Update spatial context linking
        if len(doc_signatures) > 1:
            codebook.update_spatial_index(doc_signatures)

        if docs_processed % 100 == 0:
            elapsed = time.time() - start_time
            stats = codebook.get_stats()
            print(
                f"[{docs_processed} docs] {stats['unique_signatures']} signatures, {stats['total_token_mappings']} mappings ({elapsed:.1f}s)"
            )

    print(f"\nFinished processing. Saving to {output_file}...")
    codebook.save(output_file)

    stats = codebook.get_stats()
    print("\n=== Final Codebook Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
