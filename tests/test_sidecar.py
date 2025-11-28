from pathlib import Path

from manifold.sidecar import build_manifold_index, verify_snippet


def _write_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"text": "alpha beta gamma alpha"}\n{"text": "beta gamma delta"}\n',
        encoding="utf-8",
    )
    return corpus


def test_builds_index_and_tracks_hazard(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path)
    index = build_manifold_index(
        corpus,
        window_bytes=32,
        stride_bytes=16,
        precision=2,
        hazard_percentile=0.8,
    )
    meta = index["meta"]
    assert meta["documents"] == 2
    assert meta["total_signatures"] > 0
    assert meta["hazard_threshold"] >= 0.0

    signatures = index["signatures"]
    assert signatures
    sample_signature = next(iter(signatures.values()))
    assert "prototype" in sample_signature
    assert sample_signature["hazard"]["count"] >= 1


def test_verify_snippet_reports_matches(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path)
    index = build_manifold_index(
        corpus,
        window_bytes=32,
        stride_bytes=16,
        precision=2,
        hazard_percentile=0.8,
    )
    result = verify_snippet(
        "alpha beta gamma alpha",
        index,
        window_bytes=32,
        stride_bytes=16,
        precision=2,
        coverage_threshold=0.0,
        include_reconstruction=True,
    )

    assert result["total_windows"] >= 1
    assert result["match_ratio"] > 0
    assert result["reconstruction"]
    assert result["matched_documents"]
