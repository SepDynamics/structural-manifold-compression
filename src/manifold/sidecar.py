"""Sidecar helpers to build and query structural manifold indexes for RAG verification."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from scripts.experiments.manifold_compression_eval import iter_text_documents, sliding_windows
from sep_text_manifold import encode, native


@dataclass
class EncodedWindow:
    signature: str
    hazard: float
    byte_start: int
    byte_end: int
    char_start: int
    char_end: int
    window_index: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "signature": self.signature,
            "hazard": self.hazard,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "window_index": self.window_index,
        }


def _build_byte_index(text: str) -> List[int]:
    offsets = [0]
    for ch in text:
        offsets.append(offsets[-1] + len(ch.encode("utf-8")))
    return offsets


def _byte_to_char(byte_offset: int, byte_index: Sequence[int]) -> int:
    return max(0, bisect_right(byte_index, byte_offset) - 1)


def encode_text_to_windows(
    text: str,
    window_bytes: int = 512,
    stride_bytes: int = 384,
    precision: int = 3,
    use_native: bool = False,
) -> Tuple[List[EncodedWindow], Dict[str, str], List[float]]:
    if use_native:
        native.set_use_native(True)
    text_bytes = text.encode("utf-8")
    byte_index = _build_byte_index(text)

    windows: List[EncodedWindow] = []
    prototypes: Dict[str, str] = {}
    hazards: List[float] = []

    for window_index, (offset, chunk) in enumerate(sliding_windows(text_bytes, window_bytes, stride_bytes)):
        metrics = encode.encode_window(bytes(chunk))
        signature = encode.signature_from_metrics(
            metrics["coherence"],
            metrics["stability"],
            metrics["entropy"],
            precision=precision,
        )
        hazard = float(metrics["lambda_hazard"])
        byte_start = offset
        byte_end = offset + len(chunk)
        char_start = _byte_to_char(byte_start, byte_index)
        char_end = _byte_to_char(byte_end, byte_index)

        windows.append(
            EncodedWindow(
                signature=signature,
                hazard=hazard,
                byte_start=byte_start,
                byte_end=byte_end,
                char_start=char_start,
                char_end=char_end,
                window_index=window_index,
            )
        )

        if signature not in prototypes:
            prototypes[signature] = text_bytes[byte_start:byte_end].decode("utf-8", errors="replace")
        hazards.append(hazard)

    return windows, prototypes, hazards


def _finalise_hazard_stats(raw: MutableMapping[str, float]) -> Dict[str, float]:
    count = int(raw.get("count", 0))
    total = float(raw.get("sum", 0.0))
    mean = total / count if count else 0.0
    return {
        "count": count,
        "min": float(raw.get("min", 0.0)),
        "max": float(raw.get("max", 0.0)),
        "mean": mean,
    }


def build_manifold_index(
    text_root: Path,
    window_bytes: int = 512,
    stride_bytes: int = 384,
    precision: int = 3,
    hazard_percentile: float = 0.8,
    json_text_key: str = "text",
    max_documents: int | None = None,
    document_offset: int = 0,
    use_native: bool = False,
    store_windows: bool = True,
) -> Dict[str, object]:
    if use_native:
        native.set_use_native(True)

    signatures: Dict[str, Dict[str, object]] = {}
    documents: Dict[str, Dict[str, object]] = {}
    hazards: List[float] = []
    total_windows = 0

    processed = 0
    for doc_index, (doc_id, text) in enumerate(iter_text_documents(text_root, json_text_key=json_text_key)):
        if doc_index < max(document_offset, 0):
            continue
        if max_documents is not None and processed >= max_documents:
            break
        processed += 1

        windows, prototypes, window_hazards = encode_text_to_windows(
            text,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
            use_native=False,
        )
        hazards.extend(window_hazards)
        total_windows += len(windows)

        doc_entry: Dict[str, object] = {
            "characters": len(text),
            "bytes": len(text.encode("utf-8")),
            "window_count": len(windows),
        }
        if store_windows:
            doc_entry["windows"] = [window.to_dict() for window in windows]
        documents[doc_id] = doc_entry

        for window in windows:
            entry = signatures.get(window.signature)
            if entry is None:
                entry = {
                    "prototype": {
                        "text": prototypes[window.signature],
                        "doc_id": doc_id,
                        "byte_start": window.byte_start,
                        "byte_end": window.byte_end,
                        "char_start": window.char_start,
                        "char_end": window.char_end,
                    },
                    "occurrences": [],
                    "hazard": {"min": window.hazard, "max": window.hazard, "sum": 0.0, "count": 0},
                }
                signatures[window.signature] = entry

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

    for entry in signatures.values():
        entry["hazard"] = _finalise_hazard_stats(entry["hazard"])

    hazard_threshold = float(np.quantile(np.array(hazards), hazard_percentile)) if hazards else 0.0

    meta = {
        "text_root": str(text_root),
        "documents": len(documents),
        "window_bytes": window_bytes,
        "stride_bytes": stride_bytes,
        "precision": precision,
        "hazard_percentile": hazard_percentile,
        "hazard_threshold": hazard_threshold,
        "total_signatures": len(signatures),
        "total_windows": total_windows,
        "json_text_key": json_text_key,
        "max_documents": max_documents,
        "document_offset": document_offset,
        "use_native": use_native,
        "store_windows": store_windows,
    }

    return {
        "meta": meta,
        "signatures": signatures,
        "documents": documents,
    }


def reconstruct_from_windows(
    windows: Sequence[Mapping[str, object]],
    prototypes: Mapping[str, Mapping[str, object] | str],
) -> str:
    if not windows:
        return ""

    def _prototype_bytes(signature: str) -> bytes:
        proto = prototypes.get(signature)
        if proto is None:
            return b""
        if isinstance(proto, str):
            return proto.encode("utf-8", errors="replace")
        text = proto.get("text", "")
        return str(text).encode("utf-8", errors="replace")

    sorted_windows = sorted(windows, key=lambda item: int(item.get("byte_start", 0)))
    result = bytearray()
    for window in sorted_windows:
        signature = str(window.get("signature", ""))
        chunk = _prototype_bytes(signature)
        start = int(window.get("byte_start", 0))
        if not chunk and len(result) < start:
            continue
        if len(result) < start:
            gap = start - len(result)
            result.extend(chunk[:gap])
        overlap = len(result) - start
        if overlap < 0:
            overlap = 0
        if overlap >= len(chunk):
            continue
        result.extend(chunk[overlap:])
    return result.decode("utf-8", errors="replace")


def verify_snippet(
    text: str,
    index: Mapping[str, object],
    *,
    hazard_threshold: float | None = None,
    coverage_threshold: float = 0.5,
    window_bytes: int | None = None,
    stride_bytes: int | None = None,
    precision: int | None = None,
    use_native: bool = False,
    include_reconstruction: bool = False,
) -> Dict[str, object]:
    meta = index.get("meta", {}) if isinstance(index, Mapping) else {}
    window_bytes = window_bytes or int(meta.get("window_bytes", 512))
    stride_bytes = stride_bytes or int(meta.get("stride_bytes", 384))
    precision = precision or int(meta.get("precision", 3))
    hazard_threshold = hazard_threshold if hazard_threshold is not None else float(meta.get("hazard_threshold", 0.0))

    windows, _, _ = encode_text_to_windows(
        text,
        window_bytes=window_bytes,
        stride_bytes=stride_bytes,
        precision=precision,
        use_native=use_native,
    )
    signatures = index.get("signatures", {}) if isinstance(index, Mapping) else {}

    matched_windows = 0
    gated_hits = 0
    total = len(windows)
    matched_documents = set()
    window_results: List[Dict[str, object]] = []

    for window in windows:
        entry = signatures.get(window.signature, {})
        occurrences = entry.get("occurrences", []) if isinstance(entry, Mapping) else []
        safe_occurrences = [occ for occ in occurrences if float(occ.get("hazard", 0.0)) <= hazard_threshold]
        hazard_ok = window.hazard <= hazard_threshold
        has_match = bool(occurrences)

        if has_match:
            matched_windows += 1
            for occ in safe_occurrences or occurrences:
                matched_documents.add(str(occ.get("doc_id", "")))
        if hazard_ok and (safe_occurrences or occurrences):
            gated_hits += 1

        window_results.append(
            {
                **window.to_dict(),
                "hazard_ok": hazard_ok,
                "matched": has_match,
                "occurrences": safe_occurrences or occurrences,
            }
        )

    coverage = gated_hits / total if total else 0.0
    match_ratio = matched_windows / total if total else 0.0
    verified = coverage >= coverage_threshold

    prototypes = {sig: sig_entry.get("prototype", {}) for sig, sig_entry in signatures.items()}

    result: Dict[str, object] = {
        "verified": verified,
        "coverage": coverage,
        "match_ratio": match_ratio,
        "coverage_threshold": coverage_threshold,
        "hazard_threshold": hazard_threshold,
        "total_windows": total,
        "gated_hits": gated_hits,
        "matches": window_results,
        "matched_documents": sorted(doc for doc in matched_documents if doc),
    }
    if include_reconstruction:
        result["reconstruction"] = reconstruct_from_windows(window_results, prototypes)
    return result


def load_index(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid index payload in {path}")
    return data
