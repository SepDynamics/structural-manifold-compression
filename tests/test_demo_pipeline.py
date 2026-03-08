from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

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
    build_lexical_index,
    build_manifold_payload,
    encode_embeddings,
    rank_bm25_chunks,
    save_manifold_index,
    rank_hybrid_chunks,
    rank_documents,
    rank_embedding_chunks,
    rank_manifold_chunks,
    rank_manifold_nodes_detailed,
)
from demo.evaluate import _select_manifold_results_path
from demo.run_manifold_system import run_manifold
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


def test_bm25_and_hybrid_retrieval_match_expected_chunk(tmp_path: Path) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora_bm25",
        "Abstract\nWe introduce the Aurora Lattice Optimizer for plasma oscillations and magnetic drift.\n",
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral_bm25",
        "Abstract\nWe study the Spectral River Theorem for adaptive meshes and convergence bounds.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    chunks = build_chunks(papers, chunk_chars=200, overlap_chars=40)
    question = QuestionRecord(
        question_id="q_bm25",
        question='Which paper discusses "Aurora Lattice Optimizer"?',
        answer="Aurora Paper",
        answer_aliases=["Aurora Paper"],
        source_papers=["paper_001"],
        question_type="paper_lookup",
        evidence_terms=["Aurora Lattice Optimizer"],
    )

    lexical_index = build_lexical_index(chunks)
    ranked_bm25 = rank_bm25_chunks(
        question,
        chunks=chunks,
        lexical_index=lexical_index,
        top_k=3,
    )
    bm25_docs = rank_documents(ranked_bm25, chunks)
    assert bm25_docs[0][0] == "paper_001"

    embeddings = encode_embeddings([chunk.text for chunk in chunks], model_name="hash")
    ranked_hybrid = rank_hybrid_chunks(
        question,
        chunks=chunks,
        embeddings=embeddings,
        lexical_index=lexical_index,
        model_name="hash",
        top_k=3,
    )
    hybrid_docs = rank_documents(ranked_hybrid, chunks)
    assert hybrid_docs[0][0] == "paper_001"


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


def test_build_node_contexts_respects_snippets_per_paper(tmp_path: Path) -> None:
    doc = _write_paper(
        tmp_path,
        "aurora_many",
        (
            "Abstract\nAurora Lattice Optimizer for plasma oscillations.\n\n"
            "1 Introduction\nAurora dynamics are stable and easy to identify.\n\n"
            "2 Methods\nWe evaluate magnetic drift suppression with long horizon rollouts.\n\n"
            "3 Results\nThe optimizer improves downstream convergence.\n"
        ),
    )
    papers = [_paper_record("paper_001", "Aurora Paper", doc)]
    nodes = build_structural_nodes(papers, node_chars=90, node_overlap=20)
    ranked_nodes = [node for node in nodes if node.paper_id == "paper_001"][:3]

    limited = build_node_contexts(ranked_nodes, max_context_tokens=400, snippets_per_paper=1)
    expanded = build_node_contexts(ranked_nodes, max_context_tokens=400, snippets_per_paper=3)

    assert len(limited) == 1
    assert len(expanded) == 1
    assert expanded[0]["text"].count("\n\n") >= limited[0]["text"].count("\n\n")


def test_rank_manifold_nodes_can_disable_sidecar_rerank(tmp_path: Path) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora_sidecar",
        "Abstract\nWe introduce the Aurora Lattice Optimizer for plasma oscillations and magnetic drift.\n",
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral_sidecar",
        "Abstract\nWe study the Spectral River Theorem for adaptive meshes and convergence bounds.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    nodes = build_structural_nodes(papers, node_chars=180, node_overlap=20)
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
    question = QuestionRecord(
        question_id="q_0004",
        question='Which paper discusses "Aurora Lattice Optimizer"?',
        answer="Aurora Paper",
        answer_aliases=["Aurora Paper"],
        source_papers=["paper_001"],
        question_type="paper_lookup",
        evidence_terms=["Aurora Lattice Optimizer"],
    )

    no_sidecar = rank_manifold_nodes_detailed(
        question,
        nodes=nodes,
        index_payload=binary_payload,
        embedding_model="hash",
        window_bytes=int(metadata["window_bytes"]),
        stride_bytes=int(metadata["stride_bytes"]),
        precision=int(metadata["precision"]),
        top_k=1,
        use_sidecar_rerank=False,
        phrase_weight=0.0,
    )
    with_sidecar = rank_manifold_nodes_detailed(
        question,
        nodes=nodes,
        index_payload=binary_payload,
        embedding_model="hash",
        window_bytes=int(metadata["window_bytes"]),
        stride_bytes=int(metadata["stride_bytes"]),
        precision=int(metadata["precision"]),
        top_k=1,
        use_sidecar_rerank=True,
        phrase_weight=0.0,
    )

    assert no_sidecar[0]["used_sidecar_rerank"] is False
    assert with_sidecar[0]["used_sidecar_rerank"] is True
    assert no_sidecar[0]["score"] == pytest.approx(no_sidecar[0]["embedding_score"])
    assert with_sidecar[0]["score"] >= no_sidecar[0]["score"]


def test_run_manifold_accepts_legacy_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc1 = _write_paper(
        tmp_path,
        "aurora_legacy",
        "Abstract\nWe introduce the Aurora Lattice Optimizer for plasma oscillations.\n",
    )
    doc2 = _write_paper(
        tmp_path,
        "spectral_legacy",
        "Abstract\nWe study the Spectral River Theorem for adaptive meshes.\n",
    )
    papers = [
        _paper_record("paper_001", "Aurora Paper", doc1),
        _paper_record("paper_002", "Spectral Paper", doc2),
    ]
    nodes = build_structural_nodes(papers, node_chars=180, node_overlap=20)
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

    manifold_json = tmp_path / "manifold.json"
    manifold_json.write_text(json.dumps(metadata), encoding="utf-8")
    manifold_index = tmp_path / "manifold_index.bin"
    save_manifold_index(manifold_index, binary_payload)

    monkeypatch.setattr("demo.run_manifold_system.MANIFOLD_JSON_PATH", manifold_json)
    monkeypatch.setattr("demo.run_manifold_system.MANIFOLD_INDEX_PATH", manifold_index)
    monkeypatch.setattr("demo.run_manifold_system.MANIFOLD_RESULTS_PATH", tmp_path / "manifold_results.json")
    monkeypatch.setattr(
        "demo.run_manifold_system.load_questions",
        lambda: [
            QuestionRecord(
                question_id="q_legacy",
                question='Which paper discusses "Aurora Lattice Optimizer"?',
                answer="Aurora Paper",
                answer_aliases=["Aurora Paper"],
                source_papers=["paper_001"],
                question_type="paper_lookup",
                evidence_terms=["Aurora Lattice Optimizer"],
            )
        ],
    )

    payload = run_manifold(
        SimpleNamespace(
            qa_backend="extractive",
            ollama_endpoint="http://127.0.0.1:11434/api/generate",
            ollama_model=None,
            top_k=3,
            max_context_tokens=400,
            shuffle_index=False,
        )
    )

    assert payload["summary"]["qa_accuracy"] == 1.0
    assert payload["summary"]["sidecar_rerank"] is True


def test_evaluate_prefers_matching_no_sidecar_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "summary": {
                    "qa_backend": "ollama",
                    "questions": 60,
                }
            }
        ),
        encoding="utf-8",
    )
    stale_manifold = tmp_path / "manifold_results.json"
    stale_manifold.write_text(
        json.dumps(
            {
                "summary": {
                    "qa_backend": "extractive",
                    "questions": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    no_sidecar = tmp_path / "manifold_results_no_sidecar.json"
    no_sidecar.write_text(
        json.dumps(
            {
                "summary": {
                    "qa_backend": "ollama",
                    "questions": 60,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("demo.evaluate.MANIFOLD_RESULTS_PATH", stale_manifold)
    monkeypatch.setattr("demo.evaluate.MANIFOLD_NO_SIDECAR_RESULTS_PATH", no_sidecar)

    selected = _select_manifold_results_path(
        preferred_path=None,
        baseline=json.loads(baseline_path.read_text(encoding="utf-8")),
    )
    assert selected == no_sidecar
