#!/usr/bin/env python3
"""Stream selected FineMath slices to disk as JSONL shards.

This helper avoids pulling the entire dataset into RAM by iterating over
the Hugging Face dataset in streaming mode and writing shards with a fixed
number of records.  The resulting files can be fed directly into
``prepare_causal_dataset.py`` via ``--text-root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive guard
    raise SystemExit(
        "datasets package not found. Install it with `pip install datasets` inside your venv."
    ) from exc


FINEMATH_DATASET = "HuggingFaceTB/finemath"


def infer_text_field(record: dict[str, object]) -> str:
    """Extract a text payload from a FineMath record."""

    for key in ("text", "content", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Unable to locate a text field in record; checked 'text', 'content', 'body'.")


def iter_records(
    dataset_name: str,
    config: str,
    split: str,
    shuffle: bool,
) -> Iterator[dict[str, object]]:
    dataset = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
        token=True,
    )
    iterator: Iterable[dict[str, object]] = dataset.shuffle(buffer_size=10_000) if shuffle else dataset
    yield from iterator


def open_shard(out_dir: Path, prefix: str, index: int) -> tuple[Path, object]:
    shard_path = out_dir / f"{prefix}_{index:05d}.jsonl"
    handle = shard_path.open("w", encoding="utf-8")
    return shard_path, handle


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a FineMath slice to JSONL shards.")
    parser.add_argument("--config", default="web_0.90_to_1.00", help="FineMath config to pull (default: web_0.90_to_1.00)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for shards")
    parser.add_argument(
        "--records-per-file",
        type=int,
        default=50_000,
        help="Number of records per JSONL shard (default: 50k)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional cap on total records (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        help="Optional cap on total bytes written (approximate, stops after exceeding this many bytes).",
    )
    parser.add_argument(
        "--prefix",
        default="finemath",
        help="Shard filename prefix (default: finemath)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomise record order via streaming shuffle buffer",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    records_in_shard = 0
    total_records = 0
    total_bytes = 0
    shard_path, shard_handle = open_shard(out_dir, args.prefix, shard_index)

    try:
        for idx, record in enumerate(iter_records(FINEMATH_DATASET, args.config, args.split, args.shuffle)):
            if args.max_records is not None and idx >= args.max_records:
                break

            try:
                text = infer_text_field(record)
            except KeyError as exc:
                print(f"[warn] skipping record {idx}: {exc}", file=sys.stderr)
                continue

            payload = {
                "id": record.get("id", idx),
                "source": record.get("source"),
                "quality": record.get("quality_score"),
                "text": text,
            }
            line = json.dumps(payload, ensure_ascii=False)
            shard_handle.write(line + "\n")
            records_in_shard += 1
            total_records += 1
            total_bytes += len(line) + 1

            if args.max_bytes is not None and total_bytes >= args.max_bytes:
                break

            if args.records_per_file > 0 and records_in_shard >= args.records_per_file:
                shard_handle.close()
                shard_index += 1
                records_in_shard = 0
                shard_path, shard_handle = open_shard(out_dir, args.prefix, shard_index)
    finally:
        shard_handle.close()

    print(
        json.dumps(
            {
                "config": args.config,
                "split": args.split,
                "output_dir": str(out_dir),
                "shards_written": shard_index + (1 if records_in_shard else 0),
                "records": total_records,
                "bytes": total_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
