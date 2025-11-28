from pathlib import Path

import json

from manifold.sidecar import build_index, encode_text, verify_snippet


def _load_docs(corpus: Path) -> dict[str, str]:
    docs = {}
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            docs[str(record.get("doc_id", len(docs)))] = str(record.get("text", ""))
    return docs


def _write_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"doc_id": "doc1", "text": "alpha beta gamma alpha"}\n{"doc_id": "doc2", "text": "beta gamma delta"}\n',
        encoding="utf-8",
    )
    return corpus


def test_encode_text_emits_windows_and_stats() -> None:
    encoded = encode_text("alpha beta gamma", window_bytes=16, stride_bytes=8, precision=2, hazard_percentile=0.5)
    assert encoded.windows
    assert encoded.original_bytes > 0
    assert encoded.hazard_threshold >= 0.0
    assert len(encoded.prototypes) == len(set(win.signature for win in encoded.windows))


def test_build_index_and_verify(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path)
    docs = _load_docs(corpus_path)
    index = build_index(
        docs,
        window_bytes=32,
        stride_bytes=16,
        precision=2,
        hazard_percentile=0.8,
    )

    assert index.meta["documents"] == 2
    assert index.meta["total_signatures"] > 0
    assert index.meta["hazard_threshold"] >= 0.0
    assert index.signatures

    result = verify_snippet(
        "alpha beta gamma alpha",
        index,
        window_bytes=32,
        stride_bytes=16,
        precision=2,
        coverage_threshold=0.0,
        include_reconstruction=True,
    )

    assert result.total_windows >= 1
    assert result.match_ratio > 0
    assert result.reconstruction
    assert result.matched_documents


def test_index_serialization_includes_version(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path)
    docs = _load_docs(corpus_path)
    index = build_index(
        docs,
        window_bytes=32,
        stride_bytes=16,
        precision=2,
        hazard_percentile=0.8,
    )
    payload = index.to_dict()
    assert payload.get("format_version") == "1"
    assert "hazard_threshold" in payload
