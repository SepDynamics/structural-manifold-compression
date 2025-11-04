from pathlib import Path

from scripts.experiments import manifold_compression_eval


def test_small_sample_runs(tmp_path):
    sample_root = Path("docs/whitepaper/sample_data/fox")
    assert sample_root.exists()
    summary = manifold_compression_eval.evaluate_manifold(
        text_root=sample_root,
        window_bytes=128,
        stride_bytes=96,
        precision=2,
        tokenizer_name="gpt2",
        max_documents=2,
    )
    assert summary["documents"] >= 1
    tokens = summary["token_metrics"]
    assert tokens["token_compression_unique"] > 1
    assert 0 <= tokens["token_accuracy"] <= 1
