#!/usr/bin/env python3
"""Gradio app for structural manifold sidecar: compression, reconstruction, and verification."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import os
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

WINDOW_BYTES = 128
STRIDE_BYTES = 96
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ENABLE_RETRIEVE = os.getenv("ENABLE_RETRIEVE", "0").lower() in {"1", "true", "yes"}

from manifold.sidecar import (
    EncodeResult,
    ManifoldIndex,
    build_index,
    encode_text,
    reconstruct_from_windows,
    verify_snippet,
)

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency handled by requirements.txt
    pdfplumber = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - lazy load handled later
    SentenceTransformer = None


docs_store: Dict[str, str] = {}
encodings_store: Dict[str, EncodeResult] = {}
doc_counter = 0
_embedding_model = None


def _next_doc_id() -> str:
    global doc_counter
    doc_counter += 1
    return f"doc-{doc_counter}"


def _extract_text_from_file(file_obj) -> Tuple[Optional[str], Optional[str]]:
    if file_obj is None:
        return None, None
    path = Path(file_obj.name)
    suffix = path.suffix.lower()
    raw_bytes = file_obj.read()
    file_obj.seek(0)
    if suffix in {".txt", ".md"}:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return path.name, text
    if suffix == ".pdf":
        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is required for PDF ingestion. Install with `pip install pdfplumber`."
            )
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        return path.name, text
    raise ValueError(f"Unsupported file type: {suffix}")


def _make_hazard_plot(hazards):
    fig, ax = plt.subplots(figsize=(5, 3))
    if hazards:
        ax.hist(hazards, bins=20, color="#2f6fff", alpha=0.8)
    ax.set_title("Window hazards")
    ax.set_xlabel("Hazard λ")
    ax.set_ylabel("Window count")
    fig.tight_layout()
    return fig


def _preview(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n… [truncated {len(text) - limit} chars]"


def _chunk_text(
    text: str, chunk_size: int = 512, overlap: int = 128
) -> list[tuple[str, str]]:
    chunks = []
    start = 0
    text_len = len(text)
    idx = 0
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = text[start:end]
        chunks.append((f"chunk-{idx}", chunk))
        if end == text_len:
            break
        start = end - overlap
        idx += 1
    return chunks


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for retrieval demo. Install with `pip install sentence-transformers`."
            )
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def handle_compress(file, raw_text):
    try:
        text_source = None
        text_content = ""
        if file is not None:
            name, content = _extract_text_from_file(file)
            text_source = name or "upload"
            text_content = content or ""
        elif raw_text and raw_text.strip():
            text_source = "pasted"
            text_content = raw_text
        if not text_content or not text_content.strip():
            return (
                "No text provided.",
                "",
                "",
                "",
                None,
                gr.update(choices=list(docs_store.keys()), value=None),
            )

        doc_id = _next_doc_id()
        docs_store[doc_id] = text_content
        encoded = encode_text(
            text_content,
            window_bytes=WINDOW_BYTES,
            stride_bytes=STRIDE_BYTES,
        )
        encodings_store[doc_id] = encoded

        reconstruction = reconstruct_from_windows(encoded.windows, encoded.prototypes)

        unique_sigs = len(encoded.prototypes)
        bytes_before = encoded.original_bytes
        # approximate storage using 9 bytes/signature (matches default precision)
        bytes_after = unique_sigs * 9
        compression_ratio = (bytes_before / bytes_after) if bytes_after else 0.0
        stats = textwrap.dedent(
            f"""
            **doc_id**: {doc_id} ({text_source})
            - windows: {len(encoded.windows)}
            - unique signatures: {unique_sigs}
            - hazard gate: ≤ {encoded.hazard_threshold:.4f}
            - original bytes: {bytes_before}
            - manifold payload bytes (~signatures): {bytes_after}
            - compression ratio (approx): {compression_ratio:.2f}×
            """
        ).strip()

        hazards = encoded.hazards
        fig = _make_hazard_plot(hazards)
        if encoded.hazard_threshold and hazards:
            ax = fig.axes[0]
            ax.axvline(
                encoded.hazard_threshold,
                color="red",
                linestyle="--",
                label="hazard gate",
            )
            ax.legend()

        dropdown_update = gr.update(choices=list(docs_store.keys()), value=doc_id)
        return (
            f"Stored {doc_id}",
            _preview(text_content),
            _preview(reconstruction),
            stats,
            fig,
            dropdown_update,
        )
    except Exception as exc:  # pragma: no cover - UI surface
        return (
            f"Error: {exc}",
            "",
            "",
            "",
            None,
            gr.update(choices=list(docs_store.keys()), value=None),
        )


def _ensure_index() -> Optional[ManifoldIndex]:
    if not docs_store:
        return None
    return build_index(
        docs_store,
        window_bytes=WINDOW_BYTES,
        stride_bytes=STRIDE_BYTES,
    )


def handle_verify(selected_doc, snippet, coverage_threshold):
    if not snippet or not snippet.strip():
        return "Provide a snippet to verify.", ""
    index = _ensure_index()
    if index is None:
        return "No documents ingested yet.", ""

    meta = getattr(index, "meta", {}) if hasattr(index, "meta") else {}
    window_bytes = int(meta.get("window_bytes", WINDOW_BYTES))
    default_hazard_threshold = float(meta.get("hazard_threshold", 0.8))
    # hazard threshold slider is passed via bound partial; fallback to meta value
    hazard_threshold = handle_verify.hazard_threshold  # type: ignore[attr-defined]
    if hazard_threshold is None:
        hazard_threshold = default_hazard_threshold

    snippet_bytes = len(snippet.encode("utf-8"))
    too_short = snippet_bytes < window_bytes

    result = verify_snippet(
        snippet,
        index,
        coverage_threshold=coverage_threshold,
        hazard_threshold=hazard_threshold,
        window_bytes=WINDOW_BYTES,
        stride_bytes=STRIDE_BYTES,
        include_reconstruction=False,
    )
    total = max(result.total_windows, 1)
    raw_hits = sum(1 for m in result.matches if m.get("matched"))
    hazard_hits = sum(1 for m in result.matches if m.get("hazard_ok"))
    raw_coverage = raw_hits / total
    safe_coverage = hazard_hits / total
    verified = safe_coverage >= coverage_threshold
    status = "✅ Verified" if verified else "❌ Not verified"
    status_color = "green" if verified else "red"
    status_line = (
        f"<span style='color:{status_color}; font-weight:700;'>{status}</span> "
        f"(raw={raw_coverage*100:.2f}%, safe={safe_coverage*100:.2f}%, hazard_gate ≤ {hazard_threshold:.3f})"
    )

    lines = []
    matched = [m for m in result.matches if m.get("occurrences")]
    for match in matched[:20]:
        sig = str(match.get("signature", ""))[:12]
        hz = float(match.get("hazard", 0.0))
        occ = match.get("occurrences", []) or []
        first_doc = occ[0].get("doc_id") if occ else ""
        lines.append(
            f"- `{sig}` hazard={hz:.3f} occurrences={len(occ)} doc={first_doc}"
        )
    matches_md = "\n".join(lines) if lines else "_No matches_"
    if too_short and not lines:
        matches_md = (
            matches_md
            + f"\n\n_Note: snippet is {snippet_bytes} bytes; index windows are {window_bytes} bytes. "
            "Use a longer snippet or build the index with a smaller window to improve coverage._"
        )
    if raw_hits and not hazard_hits:
        matches_md = (
            matches_md
            + "\n\n_Note: matching signatures exist but were filtered out by the hazard gate. "
            "Raise the hazard threshold slider to test without gating._"
        )
    return status_line, matches_md


def handle_retrieve(question, top_k, coverage_threshold, hazard_threshold):
    if not question or not question.strip():
        return "Provide a question.", "", ""
    if not docs_store:
        return "No documents ingested yet.", "", ""

    index = _ensure_index()
    if index is None:
        return "No documents ingested yet.", "", ""

    # Build chunks
    chunks = []
    for doc_id, text in docs_store.items():
        for chunk_id, chunk_text in _chunk_text(text):
            chunks.append((doc_id, chunk_id, chunk_text))
    if not chunks:
        return "No chunks available to retrieve.", "", ""

    chunk_texts = [c[2] for c in chunks]
    chunk_embeddings = _embed_texts(chunk_texts)
    question_embedding = _embed_texts([question])[0]
    scores = np.dot(chunk_embeddings, question_embedding)
    order = np.argsort(scores)[::-1]
    top_indices = order[: int(top_k)]

    naive_lines = []
    verified_lines = []
    for rank, idx in enumerate(top_indices, start=1):
        doc_id, chunk_id, chunk_text = chunks[int(idx)]
        score = float(scores[int(idx)])
        naive_lines.append(
            f"- [{rank}] {doc_id}::{chunk_id} score={score:.3f}\n  {chunk_text[:200]}..."
        )

        result = verify_snippet(
            chunk_text,
            index,
            coverage_threshold=coverage_threshold,
            hazard_threshold=hazard_threshold,
            window_bytes=WINDOW_BYTES,
            stride_bytes=STRIDE_BYTES,
            include_reconstruction=False,
        )
        total = max(result.total_windows, 1)
        raw_hits = sum(1 for m in result.matches if m.get("matched"))
        hazard_hits = sum(1 for m in result.matches if m.get("hazard_ok"))
        raw_coverage = raw_hits / total
        safe_coverage = hazard_hits / total
        status = "✅" if safe_coverage >= coverage_threshold else "❌"
        verified_lines.append(
            f"- [{rank}] {doc_id}::{chunk_id} {status} score={score:.3f} "
            f"raw={raw_coverage*100:.2f}%, safe={safe_coverage*100:.2f}% "
            f"(hazard_gate ≤ {hazard_threshold:.3f})"
        )
    naive_md = "\n".join(naive_lines) if naive_lines else "_No chunks_"
    verified_md = (
        "\n".join(verified_lines) if verified_lines else "_No verified chunks_"
    )
    return "Retrieved top-k chunks:", naive_md, verified_md


with gr.Blocks(title="Structural Manifold Sidecar") as demo:
    gr.Markdown(
        "# Structural Manifold Sidecar\nCompression + verification for RAG provenance."
    )

    with gr.Tab("Compress & Reconstruct (Structural Manifolds)"):
        gr.Markdown(
            "Upload a document or paste text. We encode it into structural manifolds, reconstruct an approximate "
            "version, and show compression + hazard stats."
        )
        file_input = gr.File(
            label="Upload (.pdf, .txt, .md)", file_types=[".pdf", ".txt", ".md"]
        )
        text_input = gr.Textbox(label="Or paste text", lines=6)
        run_btn = gr.Button("Run structural manifold")
        doc_msg = gr.Markdown()
        original_box = gr.Textbox(label="Original (preview)", lines=10)
        recon_box = gr.Textbox(label="Reconstruction (preview)", lines=10)
        stats_box = gr.Markdown(label="Stats")
        hazard_plot = gr.Plot(label="Hazard histogram")

    with gr.Tab("Verify snippet"):
        gr.Markdown(
            "Paste any snippet. We re-encode it, look for matching manifold signatures in your ingested docs, "
            "and compute hazard-gated coverage."
        )
        doc_dropdown = gr.Dropdown(
            label="Docs ingested this session",
            choices=list(docs_store.keys()),
            interactive=True,
        )
        snippet_box = gr.Textbox(label="Snippet to verify", lines=6)
        coverage_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Coverage threshold",
        )
        hazard_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            step=0.01,
            label="Hazard gate (raise to be more permissive)",
        )
        verify_btn = gr.Button("Verify")
        verify_status = gr.Markdown()
        verify_matches = gr.Markdown()

    if ENABLE_RETRIEVE:
        with gr.Tab("Retrieve & Verify"):
            gr.Markdown(
                "Chunk-level RAG demo: retrieve top-k chunks via embeddings, then hazard-gate them with manifold verification."
            )
            question_box = gr.Textbox(label="Question / query", lines=3)
            topk_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1, label="Top-k chunks"
            )
            rag_coverage = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Coverage threshold (verification)",
            )
            rag_hazard = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.8,
                step=0.01,
                label="Hazard gate (verification)",
            )
            retrieve_btn = gr.Button("Retrieve & verify")
        retrieve_status = gr.Markdown()
        naive_rag = gr.Markdown(label="Naive retrieval")
        verified_rag = gr.Markdown(label="Hazard-gated retrieval (secondary demo)")

    with gr.Tab("Cortex Overview (Valkey)"):
        gr.Markdown(
            "Live Structural Heatmap visualizing the Active Working Memory. "
            "Maps absolute topological positions (Coherence vs Stability) of 'Grid Cell' signatures across all ingested data."
        )
        refresh_map_btn = gr.Button("Refresh Grid Cell Heatmap")
        heatmap_plot = gr.Plot()
        heatmap_stats = gr.Markdown()

        def generate_heatmap():
            from matplotlib.colors import LogNorm
            from manifold.valkey_client import ValkeyWorkingMemory

            vwm = ValkeyWorkingMemory()
            index = vwm.get_cached_index()

            if not index or not index.signatures:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(
                    0.5, 0.5, "Valkey Working Memory is Empty", ha="center", va="center"
                )
                ax.set_title("Manifold Topology")
                plt.close(fig)
                return fig, "No signatures found in Valkey."

            coherence_vals = []
            stability_vals = []
            counts = []

            total_hits = 0
            for sig, docs in index.signatures.items():
                try:
                    parts = sig.split("_")
                    c = float(parts[0][1:])
                    s = float(parts[1][1:])
                    # sum instances across all documents
                    count = sum(len(windows) for windows in docs.values())

                    coherence_vals.append(c)
                    stability_vals.append(s)
                    counts.append(count)
                    total_hits += count
                except Exception:
                    continue

            fig, ax = plt.subplots(figsize=(8, 6))
            if coherence_vals:
                # 2D Histogram heatmap
                h = ax.hist2d(
                    coherence_vals,
                    stability_vals,
                    bins=20,
                    range=[[0, 1], [0, 1]],
                    weights=counts,
                    cmap="magma",
                    norm=LogNorm(),
                )
                fig.colorbar(h[3], ax=ax, label="Hit Density (Log Scale)")

            ax.set_title("Valkey Grid Cell Phase Space")
            ax.set_xlabel("Coherence (QFH Phase Lock)")
            ax.set_ylabel("Stability (Lyapunov Divergence)")
            ax.grid(alpha=0.2)
            fig.tight_layout()

            stats_msg = f"**Total Unique Grid Cells:** {len(coherence_vals)} | **Total Window Hits:** {total_hits}"
            plt.close(fig)
            return fig, stats_msg

        refresh_map_btn.click(
            generate_heatmap, inputs=[], outputs=[heatmap_plot, heatmap_stats]
        )

    with gr.Tab("Architectural Alignment Score"):
        gr.Markdown(
            "Real-time visualization of codebase structural integrity. Watch the inference state "
            "evaluate against established continuous geometries. High Structural Tension "
            "indicates the code deviates from established motifs, requiring heuristic intervention."
        )
        orbit_plot = gr.Plot()
        orbit_warning = gr.Markdown()

        def render_orbital_map():
            # In a live environment, this would read the Lyapunov exponent / Chaos Proxy from Valkey
            # Here, we simulate a particle that drifts based on a mocked reading, but let's look for a real key
            from manifold.valkey_client import ValkeyWorkingMemory
            import math
            import random
            import matplotlib.patches as patches

            vwm = ValkeyWorkingMemory()
            raw_lyapunov = vwm.r.get("manifold:chaos_proxy")
            chaos_val = (
                float(raw_lyapunov) if raw_lyapunov else 0.0
            )  # baseline zero chaos unless active

            # Triangle Vertices
            # SSM (Predictive State): Top, Valkey (Spatial Memory): Bottom Left, Transformer (Heuristic): Bottom Right
            points = {"SSM": (0, 1), "Valkey": (-1, -1), "Transformer": (1, -1)}

            # Position the particle based on chaos.
            # Low chaos (0.0) -> Particle orbits exactly between SSM and Valkey
            # High chaos (1.0) -> Particle is pulled violently toward the Transformer Vertex
            stability_centroid_x = (points["SSM"][0] + points["Valkey"][0]) / 2
            stability_centroid_y = (points["SSM"][1] + points["Valkey"][1]) / 2

            # Linear interpolation toward the Transformer
            particle_x = (
                stability_centroid_x
                + (points["Transformer"][0] - stability_centroid_x) * chaos_val
            )
            particle_y = (
                stability_centroid_y
                + (points["Transformer"][1] - stability_centroid_y) * chaos_val
            )

            # Add some orbit jitter
            particle_x += random.uniform(-0.1, 0.1)
            particle_y += random.uniform(-0.1, 0.1)

            fig, ax = plt.subplots(figsize=(6, 6))

            # Draw Attractor Triangle
            polygon = patches.Polygon(
                [points["SSM"], points["Valkey"], points["Transformer"]],
                closed=True,
                fill=False,
                edgecolor="black",
                linestyle="--",
                alpha=0.3,
            )
            ax.add_patch(polygon)

            # Draw Vertices
            ax.scatter(
                *points["SSM"], color="blue", s=200, label="Structural Baseline (SSM)"
            )
            ax.scatter(
                *points["Valkey"], color="green", s=200, label="Fast Memory (Valkey)"
            )
            ax.scatter(
                *points["Transformer"],
                color="orange",
                s=200,
                label="Heuristic Fallback (Transformer)",
            )

            # Draw Particle
            is_high_chaos = chaos_val > 0.8
            particle_color = "red" if is_high_chaos else "purple"
            ax.scatter(
                particle_x,
                particle_y,
                color=particle_color,
                s=150,
                marker="*",
                label="Current Inference State",
            )

            ax.set_title(f"Architectural Alignment (Tension Proxy: {chaos_val:.2f})")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.axis("off")
            ax.legend(loc="upper right", fontsize="small")
            fig.tight_layout()

            msg = "🟢 **Structurally Aligned**: Code geometry matches the established repository baseline."
            if is_high_chaos:
                msg = "🛑 **High Structural Tension — Code deviates significantly from codebase motifs.**"

            plt.close(fig)
            return fig, msg

        orbit_timer = gr.Timer(1)
        orbit_timer.tick(
            render_orbital_map, inputs=[], outputs=[orbit_plot, orbit_warning]
        )

    with gr.Tab("Prompt Binding (Bi-Directional Steering)"):
        gr.Markdown(
            "Freeze the Semantic Adapter's Recency List. This overrides the physical Codebook, "
            "forcing the Proactive Agent to conditionally generate only within this exact topological vocabulary."
        )

        bind_input = gr.Textbox(
            label="Frozen Vocabulary (comma-separated tokens)", lines=2
        )
        with gr.Row():
            bind_btn = gr.Button("Bind to Cortex", variant="primary")
            clear_bind_btn = gr.Button("Release Binding", variant="secondary")
        bind_status = gr.Markdown()

        def handle_bind(tokens):
            if not tokens or not tokens.strip():
                return "⚠️ Please provide tokens to bind."
            from manifold.valkey_client import ValkeyWorkingMemory

            vwm = ValkeyWorkingMemory()
            if not vwm.ping():
                return "❌ Valkey is offline."
            vwm.r.set("manifold:prompt_binding", tokens.strip())
            return f"✅ **Cortex Recency List Frozen** to: `{tokens.strip()}`"

        def handle_clear():
            from manifold.valkey_client import ValkeyWorkingMemory

            vwm = ValkeyWorkingMemory()
            if not vwm.ping():
                return "❌ Valkey is offline."
            vwm.r.delete("manifold:prompt_binding")
            return "✅ **Cortex Binding Released**. Agent returning to dynamic structural tracking."

        bind_btn.click(handle_bind, inputs=[bind_input], outputs=[bind_status])
        clear_bind_btn.click(handle_clear, inputs=[], outputs=[bind_status])

    with gr.Tab("Proactive Agent (Hazards)"):
        gr.Markdown(
            "Live predictive hazards detected autonomously by the Pair Programmer daemon. "
            "Polled every 2 seconds via `watcher_output.txt`."
        )
        hazard_box = gr.Textbox(label="Predictive Hazards", lines=15, interactive=False)

        def read_hazards():
            import os

            hazard_file = os.path.join(os.path.dirname(__file__), "watcher_output.txt")
            if os.path.exists(hazard_file):
                try:
                    with open(hazard_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        return content if content else "No hazards detected..."
                except Exception:
                    return "Waiting for agent..."
            return "File watcher_output.txt not found. Is pair_programmer_agent.py running?"

        timer = gr.Timer(2)
        timer.tick(read_hazards, inputs=[], outputs=[hazard_box])

    run_btn.click(
        handle_compress,
        inputs=[file_input, text_input],
        outputs=[
            doc_msg,
            original_box,
            recon_box,
            stats_box,
            hazard_plot,
            doc_dropdown,
        ],
    )

    def bound_verify(snippet, coverage, hazard):
        # stash hazard threshold on the function object so handle_verify can read it without changing signature
        handle_verify.hazard_threshold = hazard  # type: ignore[attr-defined]
        return handle_verify(None, snippet, coverage)

    verify_btn.click(
        bound_verify,
        inputs=[snippet_box, coverage_slider, hazard_slider],
        outputs=[verify_status, verify_matches],
    )
    if ENABLE_RETRIEVE:
        retrieve_btn.click(
            handle_retrieve,
            inputs=[question_box, topk_slider, rag_coverage, rag_hazard],
            outputs=[retrieve_status, naive_rag, verified_rag],
        )


if __name__ == "__main__":
    demo.launch()
