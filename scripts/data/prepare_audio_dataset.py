#!/usr/bin/env python3
"""Prepare audio binary sequences for Mamba trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Append score directory for importing encode if needed
SCORE_CANDIDATES = [
    REPO_ROOT / "score" / "src",
    REPO_ROOT.parent / "score" / "src",
]
for candidate in SCORE_CANDIDATES:
    if candidate.exists():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from datasets import Dataset, Features, Sequence as HFSequence, Value
from sep_text_manifold import encode, native
from scripts.experiments.manifold_compression_eval import sliding_windows


def encode_chunk(chunk, vocab, vocab_list):
    encoded = []
    for signature in chunk:
        token_id = vocab.get(signature)
        if token_id is None:
            token_id = len(vocab_list)
            vocab[signature] = token_id
            vocab_list.append(signature)
        encoded.append(int(token_id))
    return encoded


def main():
    parser = argparse.ArgumentParser(
        description="Build a causal LM dataset from a WAV file."
    )
    parser.add_argument(
        "--wav-path", type=Path, required=True, help="Path to input .wav file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for artifacts",
    )
    parser.add_argument(
        "--window-bytes", type=int, default=512, help="Sliding window size in bytes"
    )
    parser.add_argument(
        "--stride-bytes", type=int, default=384, help="Sliding window stride in bytes"
    )
    parser.add_argument(
        "--precision", type=int, default=3, help="Quantization precision for signatures"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Max manifold tokens per sequence",
    )
    args = parser.parse_args()

    wav_path = args.wav_path.expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"wav file not found: {wav_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Loaded {len(audio_bytes)} bytes from {wav_path}")

    # Use native fast C++ window if available
    native.set_use_native(True)
    can_use_native_batch = (
        hasattr(native, "analyze_window_batch") and native.HAVE_NATIVE
    )

    signatures = []
    pending_windows = []

    def flush_batch():
        if not pending_windows:
            return
        if can_use_native_batch:
            metrics_batch = native.analyze_window_batch(pending_windows)
        else:
            metrics_batch = (encode.encode_window(w) for w in pending_windows)

        for metrics in metrics_batch:
            signatures.append(
                encode.signature_from_metrics(
                    metrics["coherence"],
                    metrics["stability"],
                    metrics["entropy"],
                    precision=args.precision,
                )
            )
        pending_windows.clear()

    print("Encoding byte windows into structural manifolds...")
    for _, chunk in sliding_windows(audio_bytes, args.window_bytes, args.stride_bytes):
        pending_windows.append(bytes(chunk))
        if len(pending_windows) >= 1024:
            flush_batch()
    flush_batch()

    vocab = {}
    vocab_list = []
    encoded_all = encode_chunk(signatures, vocab, vocab_list)

    print(
        f"Generated {len(encoded_all)} local tokens. Vocabulary size: {len(vocab_list)}."
    )

    # Chunk into sequences
    samples = []
    chunk_size = args.sequence_length
    for start in range(0, len(encoded_all), chunk_size):
        chunk = encoded_all[start : start + chunk_size]
        if len(chunk) < 8:  # Min length
            continue
        record = {
            "doc_id": f"audio_chunk_{start}",
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
            "length": len(chunk) - 1,
        }
        samples.append(record)

    features = Features(
        {
            "doc_id": Value("string"),
            "input_ids": HFSequence(Value("int32")),
            "labels": HFSequence(Value("int32")),
            "length": Value("int32"),
        }
    )

    dataset_dict = {
        "doc_id": [s["doc_id"] for s in samples],
        "input_ids": [s["input_ids"] for s in samples],
        "labels": [s["labels"] for s in samples],
        "length": [s["length"] for s in samples],
    }

    dataset = Dataset.from_dict(dataset_dict, features=features)
    dataset_path = output_dir / "hf_dataset"
    dataset.save_to_disk(dataset_path)

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({"signatures": vocab_list}, f, indent=2)

    print(f"Saved {len(samples)} training sequences to {dataset_path}")


if __name__ == "__main__":
    main()
