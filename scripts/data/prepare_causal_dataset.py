#!/usr/bin/env python3
"""Prepare (and resume) manifold signature sequences for causal language modeling."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCORE_CANDIDATES = [
    REPO_ROOT / "score" / "src",
    REPO_ROOT.parent / "score" / "src",
]
for candidate in SCORE_CANDIDATES:
    if candidate.exists():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from datasets import Dataset, Features, Sequence as HFSequence, Value  # type: ignore  # noqa: E402

from sep_text_manifold import encode, native  # type: ignore  # noqa: E402

from scripts.experiments.manifold_compression_eval import (  # noqa: E402
    iter_text_documents,
    sliding_windows,
)


@dataclass
class BuilderPaths:
    output_dir: Path
    samples: Path
    processed_docs: Path
    stats: Path
    metadata: Path
    vocab: Path
    dataset: Path
    signatures_dir: Path


def chunk_signatures(
    signatures: Sequence[str],
    sequence_length: int,
    min_sequence_length: int,
) -> Iterable[Sequence[str]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    for start in range(0, len(signatures), sequence_length):
        chunk = signatures[start : start + sequence_length]
        if len(chunk) >= min_sequence_length:
            yield chunk


def encode_chunk(chunk: Sequence[str], vocab: Dict[str, int], vocab_list: List[str]) -> List[int]:
    encoded: List[int] = []
    for signature in chunk:
        token_id = vocab.get(signature)
        if token_id is None:
            token_id = len(vocab_list)
            vocab[signature] = token_id
            vocab_list.append(signature)
        encoded.append(int(token_id))
    return encoded


def load_processed_docs(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_processed_doc(path: Path, doc_id: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(doc_id + "\n")


def load_stats(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"documents": 0, "samples": 0, "total_signatures": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def write_stats(path: Path, stats: Dict[str, int | str]) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_vocab(path: Path) -> tuple[Dict[str, int], List[str]]:
    if not path.exists():
        return {}, []
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab_list = list(data.get("signatures", []))
    vocab = {signature: idx for idx, signature in enumerate(vocab_list)}
    return vocab, vocab_list


def write_vocab(path: Path, vocab_list: List[str]) -> None:
    payload = {"signatures": vocab_list}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_samples(path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_dataset_from_samples(path: Path, features: Features) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Sample store not found: {path}")
    return Dataset.from_json(str(path), features=features)


def reset_output(paths: BuilderPaths) -> None:
    for target in [paths.samples, paths.processed_docs, paths.stats, paths.metadata, paths.vocab]:
        if target.exists():
            target.unlink()
    if paths.dataset.exists():
        shutil.rmtree(paths.dataset)
    if paths.signatures_dir.exists():
        shutil.rmtree(paths.signatures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable causal LM dataset from manifold signatures.")
    parser.add_argument("--text-root", type=Path, required=True, help="Root directory or file with UTF-8 text/JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for artifacts")
    parser.add_argument("--window-bytes", type=int, default=512, help="Sliding window size in bytes")
    parser.add_argument("--stride-bytes", type=int, default=384, help="Sliding window stride in bytes")
    parser.add_argument("--precision", type=int, default=3, help="Quantization precision for signatures")
    parser.add_argument("--sequence-length", type=int, default=512, help="Max manifold tokens per training sample")
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=8,
        help="Minimum number of manifold tokens required to keep a sample (>=2).",
    )
    parser.add_argument(
        "--json-text-key",
        type=str,
        default="text",
        help="Field name that stores text when ingesting JSON/JSONL corpora",
    )
    parser.add_argument("--max-documents", type=int, help="Optional limit on number of new documents to process")
    parser.add_argument(
        "--export-signatures",
        action="store_true",
        help="Store per-document signatures under <output>/signatures for inspection.",
    )
    parser.add_argument("--use-native", action="store_true", help="Use the native CUDA encoder when available.")
    parser.add_argument(
        "--concat-documents",
        action="store_true",
        help="Concatenate signatures across documents before chunking (maximizes sample count for short docs).",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Skip rebuilding the Hugging Face dataset even after processing documents.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not process documents; rebuild the HF dataset from existing samples.jsonl.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete existing samples/progress and rebuild from scratch before processing.",
    )
    args = parser.parse_args()

    if args.min_sequence_length < 2:
        parser.error("--min-sequence-length must be at least 2 to form input/label pairs.")

    text_root = args.text_root.expanduser().resolve()
    if not text_root.exists():
        raise FileNotFoundError(f"text root not found: {text_root}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = BuilderPaths(
        output_dir=output_dir,
        samples=output_dir / "samples.jsonl",
        processed_docs=output_dir / "processed_docs.txt",
        stats=output_dir / "stats.json",
        metadata=output_dir / "metadata.json",
        vocab=output_dir / "vocab.json",
        dataset=output_dir / "hf_dataset",
        signatures_dir=output_dir / "signatures",
    )

    if args.reset_output:
        reset_output(paths)

    if args.use_native:
        native.set_use_native(True)

    vocab, vocab_list = load_vocab(paths.vocab)
    processed_docs = load_processed_docs(paths.processed_docs)
    stats = load_stats(paths.stats)
    pending_signatures: List[Tuple[str, str]] = []

    if args.finalize_only:
        features = Features(
            {
                "doc_id": Value("string"),
                "input_ids": HFSequence(Value("int32")),
                "labels": HFSequence(Value("int32")),
                "length": Value("int32"),
            }
        )
        dataset = build_dataset_from_samples(paths.samples, features)
        dataset.save_to_disk(paths.dataset)
        merged_metadata = {
            "text_root": str(text_root),
            "documents": len(processed_docs),
            "window_bytes": args.window_bytes,
            "stride_bytes": args.stride_bytes,
            "precision": args.precision,
            "sequence_length": args.sequence_length,
            "min_sequence_length": args.min_sequence_length,
            "json_text_key": args.json_text_key,
            "samples": len(dataset),
            "total_signatures": stats.get("total_signatures", 0),
            "vocab_size": len(vocab_list),
            "samples_path": str(paths.samples),
            "processed_docs_path": str(paths.processed_docs),
        }
        paths.metadata.write_text(json.dumps(merged_metadata, indent=2), encoding="utf-8")
        print(json.dumps(merged_metadata, indent=2))
        return

    new_docs_processed = 0

    if args.export_signatures:
        paths.signatures_dir.mkdir(parents=True, exist_ok=True)

    features = Features(
        {
            "doc_id": Value("string"),
            "input_ids": HFSequence(Value("int32")),
            "labels": HFSequence(Value("int32")),
            "length": Value("int32"),
        }
    )

    def build_record_from_chunk(chunk_signatures: List[str], doc_label: str) -> Optional[Dict[str, object]]:
        encoded = encode_chunk(chunk_signatures, vocab, vocab_list)
        if len(encoded) < 2:
            return None
        return {
            "doc_id": doc_label,
            "input_ids": encoded[:-1],
            "labels": encoded[1:],
            "length": len(encoded) - 1,
        }

    def flush_pending(force: bool = False) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        if not pending_signatures:
            return records
        chunk_size = args.sequence_length
        while len(pending_signatures) >= chunk_size:
            chunk_pairs = pending_signatures[:chunk_size]
            del pending_signatures[:chunk_size]
            chunk_signatures = [sig for sig, _ in chunk_pairs]
            start_doc = chunk_pairs[0][1]
            end_doc = chunk_pairs[-1][1]
            doc_label = start_doc if start_doc == end_doc else f"{start_doc}__to__{end_doc}"
            record = build_record_from_chunk(chunk_signatures, doc_label)
            if record:
                records.append(record)
        if force and len(pending_signatures) >= args.min_sequence_length:
            chunk_pairs = pending_signatures[:]
            pending_signatures.clear()
            chunk_signatures = [sig for sig, _ in chunk_pairs]
            start_doc = chunk_pairs[0][1]
            end_doc = chunk_pairs[-1][1]
            doc_label = start_doc if start_doc == end_doc else f"{start_doc}__to__{end_doc}"
            record = build_record_from_chunk(chunk_signatures, doc_label)
            if record:
                records.append(record)
        return records

    for doc_id, text in iter_text_documents(text_root, json_text_key=args.json_text_key):
        if doc_id in processed_docs:
            continue
        if args.max_documents is not None and new_docs_processed >= args.max_documents:
            break

        new_docs_processed += 1
        text_bytes = text.encode("utf-8")
        if not text_bytes:
            append_processed_doc(paths.processed_docs, doc_id)
            processed_docs.add(doc_id)
            continue

        doc_signatures: List[str] = []
        for _, chunk in sliding_windows(text_bytes, args.window_bytes, args.stride_bytes):
            metrics = encode.encode_window(bytes(chunk))
            signature = encode.signature_from_metrics(
                metrics["coherence"],
                metrics["stability"],
                metrics["entropy"],
                precision=args.precision,
            )
            doc_signatures.append(signature)

        if args.export_signatures:
            sig_path = paths.signatures_dir / f"{doc_id}.json"
            sig_payload = {"doc_id": doc_id, "signatures": doc_signatures}
            sig_path.write_text(json.dumps(sig_payload), encoding="utf-8")

        sample_records: List[Dict[str, object]] = []
        if args.concat_documents:
            pending_signatures.extend((signature, doc_id) for signature in doc_signatures)
            sample_records.extend(flush_pending(force=False))
        else:
            for chunk in chunk_signatures(doc_signatures, args.sequence_length, args.min_sequence_length):
                record = build_record_from_chunk(chunk, doc_id)
                if record:
                    sample_records.append(record)

        append_samples(paths.samples, sample_records)
        append_processed_doc(paths.processed_docs, doc_id)
        processed_docs.add(doc_id)

        stats["documents"] = len(processed_docs)
        stats["samples"] = stats.get("samples", 0) + len(sample_records)
        stats["total_signatures"] = stats.get("total_signatures", 0) + len(doc_signatures)
        stats["last_doc_id"] = doc_id
        stats["last_samples"] = len(sample_records)
        stats["last_signatures"] = len(doc_signatures)
        write_stats(paths.stats, stats)

        print(
            f"[{stats['documents']} docs | {stats.get('samples', 0)} samples | "
            f"{len(vocab_list)} vocab] processed {doc_id}"
        )

    if args.concat_documents:
        final_records = flush_pending(force=True)
        if final_records:
            append_samples(paths.samples, final_records)
            stats["samples"] = stats.get("samples", 0) + len(final_records)
            stats["last_samples"] = len(final_records)
            write_stats(paths.stats, stats)

    write_vocab(paths.vocab, vocab_list)

    if args.skip_finalize:
        print("Skipping dataset finalization (samples.jsonl updated).")
        return

    if not paths.samples.exists():
        raise RuntimeError("No samples were written; nothing to finalize.")

    dataset = build_dataset_from_samples(paths.samples, features)
    dataset.save_to_disk(paths.dataset)

    metadata = {
        "text_root": str(text_root),
        "documents": len(processed_docs),
        "window_bytes": args.window_bytes,
        "stride_bytes": args.stride_bytes,
        "precision": args.precision,
        "sequence_length": args.sequence_length,
        "min_sequence_length": args.min_sequence_length,
        "json_text_key": args.json_text_key,
        "samples": len(dataset),
        "total_signatures": stats.get("total_signatures", 0),
        "vocab_size": len(vocab_list),
        "samples_path": str(paths.samples),
        "processed_docs_path": str(paths.processed_docs),
        "concat_documents": args.concat_documents,
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
