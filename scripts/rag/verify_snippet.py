#!/usr/bin/env python3
"""Verify an arbitrary text snippet against a prebuilt manifold index."""

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

from manifold.sidecar import load_index, verify_snippet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hazard-gated manifold verification for RAG snippets.")
    parser.add_argument("--index", type=Path, required=True, help="Path to the manifold index JSON.")
    parser.add_argument("--text", type=str, help="Raw text to verify.")
    parser.add_argument("--text-file", type=Path, help="Path to a UTF-8 text file to verify.")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5,
        help="Minimum low-hazard coverage required to mark a snippet as verified (default: 0.5).",
    )
    parser.add_argument("--hazard-threshold", type=float, help="Override the hazard gate (default: index value).")
    parser.add_argument("--window-bytes", type=int, help="Override window size in bytes (default: index value).")
    parser.add_argument("--stride-bytes", type=int, help="Override stride in bytes (default: index value).")
    parser.add_argument("--precision", type=int, help="Override signature precision (default: index value).")
    parser.add_argument("--use-native", action="store_true", help="Prefer the native manifold kernel if available.")
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Include a reconstruction built only from prototype spans in the response.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination to write the JSON response (prints to stdout regardless).",
    )
    return parser.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        path = args.text_file.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        return path.read_text(encoding="utf-8")
    raise ValueError("Either --text or --text-file must be provided.")


def main() -> None:
    args = parse_args()
    index_path = args.index.expanduser().resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    text = _load_text(args)
    index = load_index(index_path)

    result = verify_snippet(
        text,
        index,
        hazard_threshold=args.hazard_threshold,
        coverage_threshold=args.coverage_threshold,
        window_bytes=args.window_bytes,
        stride_bytes=args.stride_bytes,
        precision=args.precision,
        use_native=args.use_native,
        include_reconstruction=args.reconstruct,
    )

    payload = json.dumps(result.to_dict(), indent=2)
    if args.output:
        out_path = args.output.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
