#!/usr/bin/env python3
"""Convert a directory of txt/md/pdf files into a JSONL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def _iter_text_file(doc_id: str, path: Path) -> Iterable[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield doc_id, text


def _iter_pdf_file(doc_id_prefix: str, path: Path) -> Iterable[Tuple[str, str]]:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in runtime, not unit test
        raise RuntimeError("pdfplumber is required to parse PDF files. Install with `pip install pdfplumber`.") from exc

    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            doc_id = f"{doc_id_prefix}#page={idx + 1}"
            yield doc_id, text


def _iter_docs(input_dir: Path) -> Iterable[Tuple[str, str]]:
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(input_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            yield from _iter_text_file(relative, path)
        elif suffix == ".pdf":
            yield from _iter_pdf_file(relative, path)


def prepare_corpus(input_dir: Path, output_jsonl: Path) -> int:
    """Walk input_dir, extract text, and write JSONL {doc_id, text} to output_jsonl."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    count = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for doc_id, text in _iter_docs(input_dir):
            if not text.strip():
                continue
            record = {"doc_id": doc_id, "text": text}
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a JSONL corpus from txt/md/pdf files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw documents.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Destination JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = prepare_corpus(args.input_dir.expanduser().resolve(), args.output_jsonl.expanduser().resolve())
    print(f"Wrote {written} documents to {args.output_jsonl}")


if __name__ == "__main__":
    main()
