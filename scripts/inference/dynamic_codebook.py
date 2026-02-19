#!/usr/bin/env python3
"""Dynamic Codebook: Lightweight deterministic lookup for manifold-to-token routing.

This is Phase 2 of the dual-stream architecture: the Router.
Instead of heavy embedding models, this uses a localized dictionary that maps
structural motifs to the tokens currently occupying that coordinate in context.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CodebookEntry:
    """Single entry in the dynamic codebook."""

    signature: str
    tokens: List[str]
    positions: List[int]
    frequency: int
    last_seen: int

    def to_dict(self):
        return {
            "signature": self.signature,
            "tokens": self.tokens,
            "positions": self.positions,
            "frequency": self.frequency,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            signature=data["signature"],
            tokens=data["tokens"],
            positions=data["positions"],
            frequency=data["frequency"],
            last_seen=data["last_seen"],
        )


class DynamicCodebook:
    """Localized, deterministic codebook for manifold-to-token mapping.

    Key properties:
    - No neural embeddings required
    - O(1) lookup per signature
    - Context-aware: tracks which tokens appear in which topological regions
    - Dynamic: updates as new context is processed
    - Zero-shot compatible: can add new terms without retraining
    """

    def __init__(self, window_size: int = 512, decay_factor: float = 0.95):
        """Initialize dynamic codebook.

        Args:
            window_size: Size of the context window for localization
            decay_factor: Exponential decay for frequency tracking
        """
        self.window_size = window_size
        self.decay_factor = decay_factor

        # Map: signature -> CodebookEntry
        self.entries: Dict[str, CodebookEntry] = {}

        # Spatial index: track which signatures appear near each other
        self.spatial_index: Dict[str, Set[str]] = defaultdict(set)

        # Track global position for recency
        self.global_position = 0

    def update(self, signature: str, token: str, position: int):
        """Update codebook with a new signature-token observation.

        Args:
            signature: Manifold signature (e.g., "c0.9_s0.1_e0.5")
            token: The token observed at this signature
            position: Position in the sequence
        """
        if signature not in self.entries:
            self.entries[signature] = CodebookEntry(
                signature=signature,
                tokens=[token],
                positions=[position],
                frequency=1,
                last_seen=position,
            )
        else:
            entry = self.entries[signature]

            # Add token if not already in list (keep unique)
            if token not in entry.tokens:
                entry.tokens.append(token)

            # Add position
            entry.positions.append(position)

            # Update frequency with decay
            entry.frequency = entry.frequency * self.decay_factor + 1
            entry.last_seen = position

        self.global_position = max(self.global_position, position)

    def update_spatial_index(self, signatures: List[str]):
        """Update spatial index with co-occurring signatures.

        Args:
            signatures: List of signatures in a local window
        """
        # Each signature co-occurs with its neighbors
        for i, sig in enumerate(signatures):
            for j in range(max(0, i - 5), min(len(signatures), i + 6)):
                if i != j:
                    self.spatial_index[sig].add(signatures[j])

    def lookup(
        self,
        signature: str,
        context_signatures: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Lookup tokens for a signature with optional context.

        Args:
            signature: Manifold signature to lookup
            context_signatures: Recent signatures for disambiguation
            top_k: Number of top candidates to return

        Returns:
            List of (token, confidence) tuples
        """
        if signature not in self.entries:
            return []

        entry = self.entries[signature]

        # Calculate confidence scores for each token
        token_scores: Dict[str, float] = {}

        # Base score: frequency and recency
        recency_weight = np.exp(
            -(self.global_position - entry.last_seen) / self.window_size
        )

        for token in entry.tokens:
            # Base score from frequency
            base_score = entry.frequency * recency_weight

            # Context boost: if context signatures share tokens with this entry
            context_boost = 0.0
            if context_signatures:
                context_boost = self._compute_context_boost(
                    signature, token, context_signatures
                )

            token_scores[token] = base_score + context_boost

        # Sort by score and return top-k
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_tokens[:top_k]

    def _compute_context_boost(
        self, signature: str, token: str, context_signatures: List[str]
    ) -> float:
        """Compute context-based confidence boost.

        Args:
            signature: Current signature
            token: Candidate token
            context_signatures: Recent context signatures

        Returns:
            Boost score based on spatial coherence
        """
        boost = 0.0

        # Check if context signatures are spatially related
        related_sigs = self.spatial_index.get(signature, set())

        for ctx_sig in context_signatures:
            if ctx_sig in related_sigs:
                # This context signature is spatially related
                if ctx_sig in self.entries:
                    ctx_entry = self.entries[ctx_sig]
                    # Boost if the token appears in the related signature
                    if token in ctx_entry.tokens:
                        boost += 1.0

        return boost

    def add_novel_term(self, signature: str, token: str, position: int):
        """Add a completely novel term to the codebook.

        This enables zero-shot injection: new vocabulary can be added
        without retraining the underlying manifold model.

        Args:
            signature: Manifold signature where this term appears
            token: The novel token/term
            position: Position in sequence
        """
        # Same as update, but explicitly labeled for novel terms
        self.update(signature, token, position)

    def prune(self, min_frequency: float = 0.1, max_age: int = 10000):
        """Prune old/infrequent entries to manage memory.

        Args:
            min_frequency: Minimum frequency threshold
            max_age: Maximum age (in positions) to keep
        """
        to_remove = []

        for sig, entry in self.entries.items():
            age = self.global_position - entry.last_seen
            if entry.frequency < min_frequency or age > max_age:
                to_remove.append(sig)

        for sig in to_remove:
            del self.entries[sig]
            if sig in self.spatial_index:
                del self.spatial_index[sig]

    def save(self, path: Path):
        """Save codebook to disk."""
        data = {
            "window_size": self.window_size,
            "decay_factor": self.decay_factor,
            "global_position": self.global_position,
            "entries": {sig: entry.to_dict() for sig, entry in self.entries.items()},
            "spatial_index": {
                sig: list(neighbors) for sig, neighbors in self.spatial_index.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DynamicCodebook":
        """Load codebook from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        codebook = cls(
            window_size=data["window_size"],
            decay_factor=data["decay_factor"],
        )

        codebook.global_position = data["global_position"]
        codebook.entries = {
            sig: CodebookEntry.from_dict(entry_data)
            for sig, entry_data in data["entries"].items()
        }
        codebook.spatial_index = defaultdict(
            set,
            {sig: set(neighbors) for sig, neighbors in data["spatial_index"].items()},
        )

        return codebook

    def get_stats(self) -> dict:
        """Get codebook statistics."""
        unique_signatures = len(self.entries)
        total_tokens = sum(len(entry.tokens) for entry in self.entries.values())
        avg_tokens_per_sig = total_tokens / unique_signatures if unique_signatures > 0 else 0

        return {
            "unique_signatures": unique_signatures,
            "total_token_mappings": total_tokens,
            "avg_tokens_per_signature": avg_tokens_per_sig,
            "global_position": self.global_position,
            "spatial_index_size": len(self.spatial_index),
        }


def build_codebook_from_corpus(
    corpus_path: Path,
    manifold_signatures_path: Path,
    output_path: Path,
    window_size: int = 512,
):
    """Build a codebook from a corpus and its manifold signatures.

    Args:
        corpus_path: Path to tokenized corpus (JSONL with 'tokens' field)
        manifold_signatures_path: Path to manifold signatures (JSONL with 'signatures')
        output_path: Where to save the codebook
        window_size: Context window size
    """
    codebook = DynamicCodebook(window_size=window_size)

    # Load corpus and signatures
    with open(corpus_path, "r", encoding="utf-8") as corpus_file, \
         open(manifold_signatures_path, "r", encoding="utf-8") as sig_file:

        for position, (corpus_line, sig_line) in enumerate(zip(corpus_file, sig_file)):
            corpus_data = json.loads(corpus_line)
            sig_data = json.loads(sig_line)

            tokens = corpus_data.get("tokens", [])
            signatures = sig_data.get("signatures", [])

            # Update codebook for each token-signature pair
            for token, signature in zip(tokens, signatures):
                codebook.update(signature, token, position)

            # Update spatial index with this window
            if len(signatures) > 1:
                codebook.update_spatial_index(signatures)

    # Save codebook
    codebook.save(output_path)

    # Print stats
    stats = codebook.get_stats()
    print("\n=== Codebook Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return codebook


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dynamic codebook from corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="Tokenized corpus (JSONL)")
    parser.add_argument("--signatures", type=Path, required=True, help="Manifold signatures (JSONL)")
    parser.add_argument("--output", type=Path, required=True, help="Output codebook path")
    parser.add_argument("--window-size", type=int, default=512, help="Context window size")

    args = parser.parse_args()

    build_codebook_from_corpus(
        args.corpus,
        args.signatures,
        args.output,
        args.window_size,
    )
