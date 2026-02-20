#!/usr/bin/env python3
"""Generate structural motif queries for the RAG precision benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple


def iter_text_files(text_root: Path) -> Iterable[Tuple[str, str]]:
    for path in sorted(text_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        doc_id = path.relative_to(text_root).with_suffix("").as_posix().replace("/", "__")
        yield doc_id, path.read_text(encoding="utf-8")


def sample_motifs(text: str, motif_count: int, min_len: int, max_len: int) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    motifs: List[str] = []
    for _ in range(motif_count):
        line = random.choice(lines)
        if len(line) < min_len:
            continue
        start = random.randint(0, max(0, len(line) - min_len))
        length = random.randint(min_len, min(max_len, len(line) - start))
        motifs.append(line[start : start + length])
    return motifs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate motif queries for RAG benchmark")
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", type=int, default=50)
    parser.add_argument("--motifs-per-file", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    files = list(iter_text_files(args.text_root))
    random.shuffle(files)
    selected = files[: args.files]

    records = []
    for doc_id, text in selected:
        motifs = sample_motifs(text, args.motifs_per_file, args.min_len, args.max_len)
        for motif in motifs:
            records.append({"query": motif, "expected_doc": doc_id})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
