from pathlib import Path
import json

from scripts.rag.prepare_corpus import prepare_corpus


def test_prepare_corpus_writes_jsonl(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample = docs_dir / "note.txt"
    sample.write_text("hello world", encoding="utf-8")

    output = tmp_path / "corpus.jsonl"
    count = prepare_corpus(docs_dir, output)

    assert count == 1
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["doc_id"] == "note.txt"
    assert record["text"] == "hello world"
