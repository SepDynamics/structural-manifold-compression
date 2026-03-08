#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pdfplumber  # type: ignore
import requests

from demo.common import (
    BUILD_METRICS_PATH,
    CORPUS_DIR,
    CORPUS_FULL_PATH,
    CORPUS_MANIFEST_PATH,
    DEFAULT_ARXIV_CATEGORIES,
    PaperRecord,
    QUESTIONS_PATH,
    combine_corpus,
    corpus_metrics_payload,
    corpus_hash,
    download_arxiv_entries,
    ensure_directories,
    estimated_token_count,
    generate_questions,
    iter_local_documents,
    load_manifest,
    normalize_paper_text,
    save_manifest,
    save_questions,
    sha256_text,
    write_metrics,
)

RAW_PDF_DIR = CORPUS_DIR.parent / "raw_pdfs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper corpus and freeze the evaluation questions.")
    parser.add_argument("--paper-count", type=int, default=60, help="Target number of papers to ingest.")
    parser.add_argument("--question-count", type=int, default=60, help="Target number of frozen questions.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_ARXIV_CATEGORIES),
        help="arXiv categories to sample from.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Optional local directory of .txt/.md files for offline or test builds.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild corpus assets even if they already exist.")
    return parser.parse_args()


def extract_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [(page.extract_text() or "").strip() for page in pdf.pages]
    return "\n\n".join(page for page in pages if page).strip()


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(response.raw, handle)


def build_from_arxiv(*, paper_count: int, categories: list[str]) -> list[PaperRecord]:
    entries = download_arxiv_entries(categories=categories, paper_count=paper_count)
    papers: list[PaperRecord] = []
    for idx, entry in enumerate(entries, start=1):
        paper_id = f"paper_{idx:03d}"
        pdf_path = RAW_PDF_DIR / f"{paper_id}.pdf"
        text_path = CORPUS_DIR / f"{paper_id}.txt"

        if not pdf_path.exists():
            download_file(str(entry["pdf_url"]), pdf_path)

        raw_text = extract_pdf_text(pdf_path)
        normalized = normalize_paper_text(raw_text)
        if not normalized:
            continue
        text_path.write_text(normalized, encoding="utf-8")

        relative_pdf = pdf_path.relative_to(pdf_path.parents[2]).as_posix()
        relative_text = text_path.relative_to(text_path.parents[2]).as_posix()
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=str(entry["title"]),
                source_id=str(entry["source_id"]),
                categories=list(entry.get("categories", [])),
                published=str(entry.get("published", "")),
                pdf_url=str(entry.get("pdf_url", "")),
                pdf_path=relative_pdf,
                text_path=relative_text,
                bytes=len(normalized.encode("utf-8")),
                characters=len(normalized),
                estimated_tokens=estimated_token_count(normalized),
                sha256=sha256_text(normalized),
            )
        )
        if len(papers) >= paper_count:
            break
    return papers


def build_from_local(input_dir: Path) -> list[PaperRecord]:
    papers: list[PaperRecord] = []
    for idx, entry in enumerate(iter_local_documents(input_dir), start=1):
        paper_id = f"paper_{idx:03d}"
        source_path = Path(entry["path"])
        text_path = CORPUS_DIR / f"{paper_id}.txt"
        raw_text = source_path.read_text(encoding="utf-8")
        normalized = normalize_paper_text(raw_text)
        text_path.write_text(normalized, encoding="utf-8")
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=str(entry["title"]),
                source_id=str(entry["source_id"]),
                categories=list(entry.get("categories", [])),
                published=str(entry.get("published", "")),
                pdf_url="",
                pdf_path="",
                text_path=text_path.relative_to(text_path.parents[2]).as_posix(),
                bytes=len(normalized.encode("utf-8")),
                characters=len(normalized),
                estimated_tokens=estimated_token_count(normalized),
                sha256=sha256_text(normalized),
            )
        )
    return papers


def build_corpus(args: argparse.Namespace) -> dict[str, object]:
    ensure_directories()
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    if not args.force and CORPUS_MANIFEST_PATH.exists() and QUESTIONS_PATH.exists() and CORPUS_FULL_PATH.exists():
        papers = load_manifest(CORPUS_MANIFEST_PATH)
        combined = CORPUS_FULL_PATH.read_text(encoding="utf-8")
        metrics = corpus_metrics_payload(papers, combined)
        write_metrics(BUILD_METRICS_PATH, metrics)
        return metrics

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    categories = list(args.categories or DEFAULT_ARXIV_CATEGORIES)

    if input_dir:
        papers = build_from_local(input_dir)
    else:
        papers = build_from_arxiv(paper_count=args.paper_count, categories=categories)

    if not papers:
        raise RuntimeError("No papers were ingested into the corpus.")

    combined = combine_corpus(papers)
    CORPUS_FULL_PATH.write_text(combined, encoding="utf-8")
    save_manifest(papers, corpus_text=combined)

    questions = generate_questions(papers, question_count=args.question_count)
    save_questions(questions, corpus_sha256=corpus_hash(papers))

    metrics = corpus_metrics_payload(papers, combined)
    metrics["question_count"] = len(questions)
    write_metrics(BUILD_METRICS_PATH, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = build_corpus(args)
    print(
        f"Corpus ready: {metrics['document_count']} documents | "
        f"{metrics['estimated_tokens']} estimated tokens | "
        f"questions frozen at {QUESTIONS_PATH}"
    )


if __name__ == "__main__":
    main()
