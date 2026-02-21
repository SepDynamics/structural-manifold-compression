#!/usr/bin/env python3
"""
Tripartite Router Module
Shared logic for processing queries through the structural manifold engine.
"""
import re
import sys
import requests
from src.manifold.valkey_client import ValkeyWorkingMemory
from src.manifold.sidecar import verify_snippet
from pathlib import Path
from typing import Tuple, List

# Provide access to local project modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.inference.dynamic_codebook import DynamicCodebook


class FreeEnergyCalculator:
    def __init__(self, target_coverage: float = 50.0):
        self.target_coverage = target_coverage

    def compute(self, observed_coverage: float) -> float:
        epsilon = self.target_coverage - observed_coverage
        return 0.5 * (epsilon**2)

    def normalized(self, observed_coverage: float) -> float:
        free_energy = self.compute(observed_coverage)
        return min(1.0, free_energy / 10000.0)


class TripartiteRouter:
    def __init__(self):
        self.wm = ValkeyWorkingMemory()
        self.free_energy = FreeEnergyCalculator()
        self.codebook = DynamicCodebook(window_size=512)
        self.codebook_position = 0

    def generate_llm_response(self, query: str, context: str, endpoint: str) -> str:
        """Call the local LLM (e.g. Ollama) with the structurally retrieved context."""
        prompt = f"""You are the ADHD Heuristic Resolver for the Tripartite Architecture.
The deterministic manifold engine encountered High Hazard Tension (predictive error) for the following query.
Instead of raw text, you are receiving the Latent Semantic Adapter's 'Recency Buffer'—the highly active
vocabulary tokens mapped mathematically to the physical geometry of this collision.

Use ONLY these active semantic tokens to guess the underlying structural relationship and answer the user.

{context}

User Query: {query}
"""
        return "MOCK LLM GENERATION TO SAVE COMPUTE"

    def process_query(
        self,
        query: str,
        hazard_threshold: float = 0.8,
        coverage_threshold: float = 0.5,
        llm_endpoint: str = "http://localhost:11434/api/generate",
        window_bytes: int = 512,
        stride_bytes: int = 384,
        precision: int = 3,
        use_native: bool = True,
    ) -> Tuple[bool, str, float, List[str]]:
        """
        Process a query through the tripartite engine.

        Returns:
            verified: bool - Whether the query passed the hazard gate
            response: str - The final response (deterministic or LLM)
            coverage: float - The coverage percentage
            matched_documents: List[str] - List of matched document IDs
        """
        index = self.wm.get_or_build_index()
        if index is None:
            return (
                False,
                "Working Memory is currently empty. The Sensory Watcher needs to ingest data first.",
                0.0,
                [],
            )

        # 1. Deterministic Reflex
        result = verify_snippet(
            text=query,
            index=index,
            coverage_threshold=coverage_threshold,
            hazard_threshold=hazard_threshold,
            window_bytes=window_bytes,
            stride_bytes=stride_bytes,
            precision=precision,
            use_native=use_native,
        )

        coverage = result.coverage * 100
        self.free_energy.target_coverage = coverage_threshold * 100
        free_energy_norm = self.free_energy.normalized(coverage)

        # Real Tripartite tension telemetry sent to Valkey Phase Space Map
        try:
            self.wm.r.set("manifold:chaos_proxy", str(free_energy_norm))
        except Exception:
            pass

        # Dynamically populate the Activation Buffer based on physical collision neighborhood
        index_cache = self.wm.get_cached_index()
        prototypes = index_cache.signatures if index_cache else {}

        recent_signatures = []
        for match in result.matches:
            sig = (
                match.get("signature")
                if isinstance(match, dict)
                else getattr(match, "signature", None)
            )
            if not sig:
                continue
            recent_signatures.append(sig)

            # Tokenize Valkey prototypes to seed the Codebook's Semantic Recency Buffer
            entry = prototypes.get(sig, {})
            proto_obj = entry.get("prototype", {}) if isinstance(entry, dict) else {}
            proto_text = (
                proto_obj.get("text", "")
                if isinstance(proto_obj, dict)
                else str(proto_obj)
            )

            if proto_text:
                words = re.findall(r"\b[A-Za-z_]{3,}\b", str(proto_text))
                for w in words:
                    self.codebook.update(sig, w.lower(), self.codebook_position)
                    self.codebook_position += 1

        if recent_signatures:
            self.codebook.update_spatial_index(recent_signatures)

        # Extract the highest-resonance Semantic Context Vector (Soft Prompts)
        activation_buffer = self.codebook.get_activation_buffer(
            recent_signatures, top_n=50
        )

        # Grab raw context strings ONLY for deterministic deterministic playback
        context_blocks = []
        for doc_id in result.matched_documents:
            doc_text = self.wm.r.get(f"{self.wm.doc_prefix}{doc_id}")
            if doc_text:
                context_blocks.append(
                    f"--- Document: {doc_id} ---\n{doc_text[:2000]}"
                )  # Truncate to save context window

        context_str = "\n".join(context_blocks)

        # 2. Heuristic Resolution (Tension Gate)
        if result.verified and free_energy_norm <= hazard_threshold:
            response = (
                "\n".join(context_blocks)
                if context_blocks
                else "(No documents securely resolved, but coverage math passed the threshold.)"
            )
        else:
            # The Latent Semantic Adapter Intercept
            if activation_buffer:
                soft_prompt_list = ", ".join(activation_buffer)
                context_str = f"[Latent Semantic Adapter | Top 50 Active Vocabulary Tokens]: {soft_prompt_list}"
            elif not context_blocks:
                context_str = "(No relevant structural matches found in Working Memory. Answer based on general knowledge if necessary.)"
            else:
                context_str = "\n".join(context_blocks)

            llm_answer = self.generate_llm_response(query, context_str, llm_endpoint)
            response = llm_answer

            # Feedback loop: write the heuristic resolution back to Valkey
            resolution_id = f"resolution-{hash(query)}"
            self.wm.add_document(
                resolution_id, f"Query: {query}\nResolution: {llm_answer}"
            )

        return result.verified, response, coverage, result.matched_documents
