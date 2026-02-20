#!/usr/bin/env python3
"""Run the Tripartite Architecture Manifold Engine as an MCP Server."""

import sys
from pathlib import Path

# Provide access to local project modules
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# We must import `mcp` to expose the FastMCP wrapper
from mcp.server.fastmcp import FastMCP

# Import from the internal Manifold engine
from src.manifold.sidecar import (
    encode_text,
    verify_snippet,
    build_index,
    ManifoldIndex,
)

# Initialize the MCP Server
mcp = FastMCP("ManifoldEngine")

# We will store the injected facts in an in-memory index for this session.
# (This represents the Dynamic Semantic Codebook.)
# Key: document ID, Value: plain text fact
IN_MEMORY_DOCS = {}
_ACTIVE_INDEX = None


def _get_or_build_index() -> ManifoldIndex | None:
    """Helper to maintain the dynamic semantic codebook."""
    global _ACTIVE_INDEX
    if not IN_MEMORY_DOCS:
        return None
    # Rebuild index dynamically when accessed (mimics < 12ms injection)
    # Optimization: In a more complex architecture, we'd incrementally update it.
    _ACTIVE_INDEX = build_index(
        IN_MEMORY_DOCS,
        window_bytes=512,
        stride_bytes=384,
        precision=3,
        use_native=True,  # leverage the C++ bindings for speed
    )
    return _ACTIVE_INDEX


@mcp.tool()
def compress_to_manifold(text: str) -> str:
    """
    Compresses a large block of text into Structural Manifold signatures.

    Args:
        text: The raw text string to compress.

    Returns:
        A report detailing the compression ratio and unique signatures generated.
    """
    encoded = encode_text(
        text,
        window_bytes=512,
        stride_bytes=384,
        precision=3,
        use_native=True,
    )

    unique_sigs = len(encoded.prototypes)
    bytes_before = encoded.original_bytes
    bytes_after = unique_sigs * 9  # Approx 9 bytes per signature
    compression_ratio = (bytes_before / bytes_after) if bytes_after else 0.0

    return (
        f"✅ Compressed {bytes_before} bytes into {bytes_after} manifold bytes.\n"
        f"Compression Ratio: {compression_ratio:.2f}x\n"
        f"Unique Signatures Generated: {unique_sigs}\n"
        f"Total Windows: {len(encoded.windows)}\n"
        f"Hazard Gate Threshold (80th pctl): {encoded.hazard_threshold:.4f}"
    )


@mcp.tool()
def inject_semantic_fact(fact_id: str, fact_text: str) -> str:
    """
    Injects a new fact dynamically into the Working Memory Codebook (Zero-Shot Learning),
    bypassing the need for catastrophic base-network fine-tuning.

    Args:
        fact_id: A unique identifier for this fact (e.g., 'doc-1', 'api-key-reset').
        fact_text: The complete factual text to assimilate.

    Returns:
        A success message indicating the fact was injected.
    """
    global _ACTIVE_INDEX
    IN_MEMORY_DOCS[fact_id] = fact_text

    # Invalidate the cache; it will rebuild on next verify
    _ACTIVE_INDEX = None

    return f"🚀 Fact '{fact_id}' assimilated into the Dynamic Semantic Codebook successfully."


@mcp.tool()
def verify_manifold_snippet(snippet: str, coverage_threshold: float = 0.5) -> str:
    """
    Verifies if a specific snippet of text (e.g. hallucinated code or facts)
    actually exists within the active Working Memory Codebook by checking structural tension.

    Args:
        snippet: The text snippet to verify against injected facts.
        coverage_threshold: The acceptable ratio of safe matches (default: 0.5).

    Returns:
        A verification report spanning the hit ratio and whether the snippet passed the hazard gate.
    """
    index = _get_or_build_index()
    if index is None:
        return "❌ Error: The Working Memory Codebook is empty. Inject facts first using `inject_semantic_fact`."

    snippet_bytes = len(snippet.encode("utf-8"))
    if snippet_bytes < 512:
        return f"❌ Error: Snippet is too short ({snippet_bytes} bytes). Must be >= 512 bytes (the window limit)."

    result = verify_snippet(
        text=snippet,
        index=index,
        coverage_threshold=coverage_threshold,
        window_bytes=512,
        stride_bytes=384,
        precision=3,
        use_native=True,
    )

    status = "✅ VERIFIED" if result.verified else "❌ FAILED VERIFICATION"
    return (
        f"Status: {status}\n"
        f"Safe Coverage Ratio: {result.coverage * 100:.2f}%\n"
        f"Raw Match Ratio: {result.match_ratio * 100:.2f}%\n"
        f"Gated Hits: {result.gated_hits} / {result.total_windows}\n"
        f"Matched Source Documents: {', '.join(result.matched_documents) if result.matched_documents else 'None'}"
    )


@mcp.tool()
def evaluate_structural_perplexity(
    manifold_model_path: str,
    manifold_dataset_path: str,
    manifold_vocab_path: str,
    raw_text_path: str,
    gpt2_model_id: str = "gpt2-medium",
) -> str:
    """
    Runs a side-by-side A/B Perplexity test of the Manifold State Space LM vs GPT-2.

    Args:
        manifold_model_path: Path to the trained Manifold LM HF model.
        manifold_dataset_path: Path to the encoded HF dataset.
        manifold_vocab_path: Path to the vocab.json file.
        raw_text_path: Path to the raw text document (.txt or .json) used for the baseline.
        gpt2_model_id: HuggingFace ID for the baseline model (default 'gpt2-medium').

    Returns:
        A formatted report detailing perplexity scores, inference costs, and structural compression ratios.
    """
    import torch
    from scripts.experiments.perplexity_compare import evaluate_manifold, evaluate_gpt2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        manifold_stats = evaluate_manifold(
            model_path=Path(manifold_model_path),
            dataset_path=Path(manifold_dataset_path),
            vocab_path=Path(manifold_vocab_path),
            eval_fraction=1.0,  # Evaluate all passed in samples for the tool
            max_samples=100,
            batch_size=4,
            device=device,
            seed=13,
        )

        gpt2_stats = evaluate_gpt2(
            model_name=gpt2_model_id,
            text_path=Path(raw_text_path),
            json_text_key="text",
            max_documents=1,
            block_size=1024,
            device=device,
        )

        compression_ratio = 0.0
        if manifold_stats["tokens_evaluated"] and gpt2_stats["raw_tokens_total"]:
            compression_ratio = (
                gpt2_stats["raw_tokens_total"] / manifold_stats["tokens_evaluated"]
            )

        return (
            f"📊 A/B Perplexity Benchmark Results\n"
            f"-----------------------------------\n"
            f"Manifold LM ({manifold_model_path}):\n"
            f"  - Perplexity: {manifold_stats['perplexity']:.2f}\n"
            f"  - Manifold Tokens Evaluated: {manifold_stats['tokens_evaluated']}\n"
            f"\n"
            f"Baseline GPT-2 ({gpt2_model_id}):\n"
            f"  - Perplexity: {gpt2_stats['perplexity']:.2f}\n"
            f"  - Standard Tokens Evaluated: {gpt2_stats['raw_tokens_total']}\n"
            f"\n"
            f"Structural Compression Gain: {compression_ratio:.2f}x\n"
        )

    except Exception as e:
        return f"❌ Evaluation failed: {str(e)}"


if __name__ == "__main__":
    # Start the stdio server
    mcp.run()
