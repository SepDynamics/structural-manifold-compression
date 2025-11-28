"""Convenience helpers to expose structural manifold functionality as a library."""

from .encoder import build_signature_index
from .sidecar import (
    build_manifold_index,
    encode_text_to_windows,
    load_index,
    reconstruct_from_windows,
    verify_snippet,
)
from .verifier import score_documents

__all__ = [
    "build_signature_index",
    "build_manifold_index",
    "encode_text_to_windows",
    "load_index",
    "reconstruct_from_windows",
    "score_documents",
    "verify_snippet",
]
