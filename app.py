#!/usr/bin/env python3
"""Gradio app for structural manifold sidecar: compression, reconstruction, and verification."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt

import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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


docs_store: Dict[str, str] = {}
encodings_store: Dict[str, EncodeResult] = {}
global_index: Optional[ManifoldIndex] = None
index_dirty: bool = False
doc_counter = 0


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
            raise RuntimeError("pdfplumber is required for PDF ingestion. Install with `pip install pdfplumber`.")
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


def handle_compress(file, raw_text):
    global global_index, index_dirty
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
            return "No text provided.", "", "", "", None, gr.update(choices=list(docs_store.keys()), value=None)

        doc_id = _next_doc_id()
        docs_store[doc_id] = text_content
        encoded = encode_text(text_content)
        encodings_store[doc_id] = encoded
        index_dirty = True
        global_index = None

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
            ax.axvline(encoded.hazard_threshold, color="red", linestyle="--", label="hazard gate")
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
        return f"Error: {exc}", "", "", "", None, gr.update(choices=list(docs_store.keys()), value=None)


def _ensure_index() -> Optional[ManifoldIndex]:
    global global_index, index_dirty
    if global_index is None or index_dirty:
        if not docs_store:
            return None
        global_index = build_index(docs_store)
        index_dirty = False
    return global_index


def handle_verify(selected_doc, snippet, coverage_threshold):
    if not snippet or not snippet.strip():
        return "Provide a snippet to verify.", ""
    index = _ensure_index()
    if index is None:
        return "No documents ingested yet.", ""

    meta = getattr(index, "meta", {}) if hasattr(index, "meta") else {}
    window_bytes = int(meta.get("window_bytes", 512))
    snippet_bytes = len(snippet.encode("utf-8"))
    too_short = snippet_bytes < window_bytes

    result = verify_snippet(
        snippet,
        index,
        coverage_threshold=coverage_threshold,
        include_reconstruction=False,
    )
    coverage_pct = result.coverage * 100.0
    status = "✅ Verified" if result.verified else "❌ Not verified"
    status_line = f"{status} (coverage = {coverage_pct:.2f}%)"

    lines = []
    matched = [m for m in result.matches if m.get("occurrences")]
    for match in matched[:20]:
        sig = str(match.get("signature", ""))[:12]
        hz = float(match.get("hazard", 0.0))
        occ = match.get("occurrences", []) or []
        first_doc = occ[0].get("doc_id") if occ else ""
        lines.append(f"- `{sig}` hazard={hz:.3f} occurrences={len(occ)} doc={first_doc}")
    matches_md = "\n".join(lines) if lines else "_No matches_"
    if too_short and not lines:
        matches_md = (
            matches_md
            + f"\n\n_Note: snippet is {snippet_bytes} bytes; index windows are {window_bytes} bytes. "
            "Use a longer snippet or build the index with a smaller window to improve coverage._"
        )
    return status_line, matches_md


with gr.Blocks(title="Structural Manifold Sidecar") as demo:
    gr.Markdown("# Structural Manifold Sidecar\nCompression + verification for RAG provenance.")

    with gr.Tab("Compress & Reconstruct (Structural Manifolds)"):
        gr.Markdown(
            "Upload a document or paste text. We encode it into structural manifolds, reconstruct an approximate "
            "version, and show compression + hazard stats."
        )
        file_input = gr.File(label="Upload (.pdf, .txt, .md)", file_types=[".pdf", ".txt", ".md"])
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
        verify_btn = gr.Button("Verify")
        verify_status = gr.Markdown()
        verify_matches = gr.Markdown()

    run_btn.click(
        handle_compress,
        inputs=[file_input, text_input],
        outputs=[doc_msg, original_box, recon_box, stats_box, hazard_plot, doc_dropdown],
    )
    verify_btn.click(
        handle_verify,
        inputs=[doc_dropdown, snippet_box, coverage_slider],
        outputs=[verify_status, verify_matches],
    )


if __name__ == "__main__":
    demo.launch()
