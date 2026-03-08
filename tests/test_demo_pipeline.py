from __future__ import annotations

from pathlib import Path

from demo.common import (
    PaperRecord,
    QuestionRecord,
    answer_from_context,
    build_chunks,
    build_retrieved_contexts,
    canonicalize_model_answer,
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
from demo.structure import build_node_contexts, build_structural_nodes, build_shuffle_node_indices


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

    nodes = build_structural_nodes(papers, node_chars=180, node_overlap=40)
    metadata, binary_payload = build_manifold_payload(
        nodes,
        question_hash="qhash",
        corpus_hash="chash",
        corpus_tokens=100,
        window_bytes=24,
        stride_bytes=4,
        precision=2,
        embedding_model="hash",
    )
    assert metadata["node_count"] >= 2
    ranked_manifold = rank_manifold_chunks(
        question,
        nodes=nodes,
        index_payload=binary_payload,
        embedding_model="hash",
        window_bytes=24,
        stride_bytes=4,
        precision=2,
        top_k=3,
    )
    manifold_docs = rank_documents(ranked_manifold, nodes)
    assert manifold_docs[0][0] == "paper_001"


def test_shuffle_node_indices_break_document_assignment(tmp_path: Path) -> None:
    doc1 = _write_paper(tmp_path, "aurora", "Abstract\nAurora Lattice Optimizer for plasma oscillations.\n")
    doc2 = _write_paper(tmp_path, "spectral", "Abstract\nSpectral River Theorem for adaptive meshes.\n")
    doc3 = _write_paper(tmp_path, "quantum", "Abstract\nQuantum Drift Solver for tempering trajectories.\n")
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
        _paper_record("paper_003", "Quantum Paper", doc3),
    ]
    nodes = build_structural_nodes(papers, node_chars=180, node_overlap=20)
    shuffle_indices = build_shuffle_node_indices(nodes, seed_text="demo")
    assert len(shuffle_indices) == len(nodes)
    for idx, shuffled_idx in enumerate(shuffle_indices):
        assert nodes[idx].paper_id != nodes[shuffled_idx].paper_id


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


def test_score_prediction_accepts_distinctive_title_alias() -> None:
    title = "POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation"
    question = QuestionRecord(
        question_id="q_0002",
        question='Which paper discusses "POET-X"?',
        answer=title,
        answer_aliases=[title],
        source_papers=["paper_003"],
        question_type="paper_lookup",
        evidence_terms=["POET-X"],
    )
    assert score_prediction("POET-X", question)
    assert score_prediction("Answer: POET X", question)


def test_canonicalize_model_answer_maps_context_reference_to_title() -> None:
    question = QuestionRecord(
        question_id="q_0003",
        question='Which paper discusses "Aurora Lattice Optimizer"?',
        answer="Aurora Paper",
        answer_aliases=["Aurora Paper"],
        source_papers=["paper_001"],
        question_type="paper_lookup",
        evidence_terms=["Aurora Lattice Optimizer"],
    )
    contexts = [
        {"paper_id": "paper_002", "title": "Spectral Paper", "text": "Other evidence"},
        {"paper_id": "paper_001", "title": "Aurora Paper", "text": "Target evidence"},
    ]
    answer = canonicalize_model_answer("[Context 2]", question, contexts)
    assert answer == "Aurora Paper"


def test_build_retrieved_contexts_groups_chunks_by_paper(tmp_path: Path) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora",
        (
            "Abstract\nAurora Lattice Optimizer for plasma oscillations.\n\n"
            "Methods\nAurora Lattice Optimizer stabilizes magnetic drift in long horizons.\n"
        ),
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral",
        "Abstract\nSpectral River Theorem for adaptive meshes and convergence bounds.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    chunks = build_chunks(papers, chunk_chars=70, overlap_chars=10)
    aurora_chunks = [chunk for chunk in chunks if chunk.paper_id == "paper_001"][:2]
    spectral_chunk = [chunk for chunk in chunks if chunk.paper_id == "paper_002"][:1]
    contexts = build_retrieved_contexts(
        [*aurora_chunks, *spectral_chunk],
        max_context_tokens=400,
    )
    assert [context["paper_id"] for context in contexts] == ["paper_001", "paper_002"]
    assert len(contexts) == 2
    assert "Evidence 1" in contexts[0]["text"]


def test_build_node_contexts_groups_nodes_by_paper(tmp_path: Path) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora_nodes",
        (
            "Abstract\nAurora Lattice Optimizer for plasma oscillations.\n\n"
            "1 Introduction\nAurora dynamics are stable.\n\n"
            "2 Methods\nWe evaluate magnetic drift suppression.\n"
        ),
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral_nodes",
        "Abstract\nSpectral River Theorem for adaptive meshes.\n\n1 Results\nConvergence bounds hold.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    nodes = build_structural_nodes(papers, node_chars=90, node_overlap=20)
    ranked_nodes = [node for node in nodes if node.paper_id == "paper_001"][:2]
    ranked_nodes += [node for node in nodes if node.paper_id == "paper_002"][:1]
    contexts = build_node_contexts(ranked_nodes, max_context_tokens=400)
    assert [context["paper_id"] for context in contexts] == ["paper_001", "paper_002"]
    assert len(contexts) == 2
    assert "Aurora" in contexts[0]["text"]
