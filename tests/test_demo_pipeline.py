from __future__ import annotations

from pathlib import Path

from demo.common import (
    PaperRecord,
    QuestionRecord,
    answer_from_context,
    build_chunks,
    extract_candidate_phrases,
    generate_questions,
    normalize_paper_text,
    score_prediction,
)
from demo.retrieval import (
    build_manifold_payload,
    encode_embeddings,
    rank_documents,
    rank_embedding_chunks,
    rank_manifold_chunks,
)


def _write_paper(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _paper_record(paper_id: str, title: str, text_path: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        source_id=paper_id,
        categories=["local"],
        published="",
        pdf_url="",
        pdf_path="",
        text_path=text_path,
        bytes=0,
        characters=0,
        estimated_tokens=0,
        sha256=paper_id,
    )


def test_normalize_paper_text_strips_frontmatter() -> None:
    raw = """
    Paper Title
    Author Name
    arXiv:1234.5678

    Abstract
    We introduce the Aurora Lattice Optimizer.

    1 Introduction
    The optimizer is evaluated on plasma oscillations.

    References
    [1] Example
    """
    normalized = normalize_paper_text(raw)
    assert "Paper Title" not in normalized
    assert "Author Name" not in normalized
    assert "References" not in normalized
    assert "Aurora Lattice Optimizer" in normalized


def test_generate_questions_extracts_document_specific_phrases(tmp_path: Path) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora",
        "Abstract\nWe introduce the Aurora Lattice Optimizer for plasma oscillations.\n",
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral",
        "Abstract\nWe study the Spectral River Theorem for adaptive meshes.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    questions = generate_questions(papers, question_count=4)
    assert questions
    assert any("Which paper discusses" in question.question for question in questions)
    assert any(question.source_papers == ["paper_001"] for question in questions)


def test_embedding_and_manifold_retrieval_match_expected_chunk(tmp_path: Path) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora",
        "Abstract\nWe introduce the Aurora Lattice Optimizer. Aurora Lattice Optimizer improves plasma oscillations and magnetic drift.\n",
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral",
        "Abstract\nWe study the Spectral River Theorem for adaptive meshes and convergence bounds.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    chunks = build_chunks(papers, chunk_chars=200, overlap_chars=40)
    question = QuestionRecord(
        question_id="q_0000",
        question='Which paper discusses "Aurora Lattice Optimizer"?',
        answer="Aurora Paper",
        answer_aliases=["Aurora Paper"],
        source_papers=["paper_001"],
        question_type="paper_lookup",
        evidence_terms=["Aurora Lattice Optimizer"],
    )
    embeddings = encode_embeddings([chunk.text for chunk in chunks], model_name="hash")

    ranked_embedding = rank_embedding_chunks(
        question,
        chunks=chunks,
        embeddings=embeddings,
        model_name="hash",
        top_k=3,
    )
    embedding_docs = rank_documents(ranked_embedding, chunks)
    assert embedding_docs[0][0] == "paper_001"

    metadata, binary_payload = build_manifold_payload(
        chunks,
        question_hash="qhash",
        corpus_hash="chash",
        corpus_tokens=100,
        window_bytes=24,
        stride_bytes=12,
        precision=2,
    )
    assert metadata["chunk_count"] >= 2
    ranked_manifold = rank_manifold_chunks(
        question,
        chunks=chunks,
        index_payload=binary_payload,
        window_bytes=24,
        stride_bytes=12,
        precision=2,
        top_k=3,
    )
    manifold_docs = rank_documents(ranked_manifold, chunks)
    assert manifold_docs[0][0] == "paper_001"


def test_extractive_answer_and_scoring_for_multi_paper_lookup() -> None:
    question = QuestionRecord(
        question_id="q_0001",
        question='Which papers discuss both "Adaptive Mesh" and "Convergence Bounds"?',
        answer=["Aurora Paper", "Spectral Paper"],
        answer_aliases=["Aurora Paper", "Spectral Paper"],
        source_papers=["paper_001", "paper_002"],
        question_type="multi_paper_lookup",
        evidence_terms=["Adaptive Mesh", "Convergence Bounds"],
    )
    contexts = [
        {"paper_id": "paper_001", "title": "Aurora Paper", "text": "Adaptive Mesh details"},
        {"paper_id": "paper_002", "title": "Spectral Paper", "text": "Convergence Bounds details"},
    ]
    answer = answer_from_context(question, contexts, qa_backend="extractive")
    assert score_prediction(answer, question)
    assert "Aurora Paper" in answer


def test_extract_candidate_phrases_prefers_salient_terms() -> None:
    text = 'Abstract\nWe introduce the "Aurora Lattice Optimizer" and compare it with the Spectral River Theorem.\n'
    phrases = extract_candidate_phrases(text)
    assert "Aurora Lattice Optimizer" in phrases
