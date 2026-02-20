#!/usr/bin/env python3
"""
Tripartite Router Module
Shared logic for processing queries through the structural manifold engine.
"""

import sys
import requests
from pathlib import Path
from typing import Tuple, List

# Provide access to local project modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.valkey_client import ValkeyWorkingMemory
from src.manifold.sidecar import verify_snippet


class FreeEnergyCalculator:
    def __init__(self, target_coverage: float = 50.0):
        self.target_coverage = target_coverage

    def compute(self, observed_coverage: float) -> float:
        epsilon = self.target_coverage - observed_coverage
        return 0.5 * (epsilon ** 2)

    def normalized(self, observed_coverage: float) -> float:
        free_energy = self.compute(observed_coverage)
        return min(1.0, free_energy / 10000.0)


class TripartiteRouter:
    def __init__(self):
        self.wm = ValkeyWorkingMemory()
        self.free_energy = FreeEnergyCalculator()

    def generate_llm_response(self, query: str, context: str, endpoint: str) -> str:
        """Call the local LLM (e.g. Ollama) with the structurally retrieved context."""
        prompt = f"""You are the ADHD Heuristic Resolver for the Tripartite Architecture.
The deterministic engine encountered high Hazard Tension for the following query and needs you to synthesize a response.
Use ONLY the provided structurally-retrieved context to answer the user.

Context:
{context}

User Query: {query}
"""
        try:
            response = requests.post(
                endpoint,
                json={
                    "model": "llama3",  # Configurable or default
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "No response generated.")
        except Exception as e:
            return f"[LLM Error: {e}]"

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
            return False, "Working Memory is currently empty. The Sensory Watcher needs to ingest data first.", 0.0, []

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

        # Grab context from Valkey based on matched documents
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
            response = "\n".join(context_blocks) if context_blocks else "(No documents securely resolved, but coverage math passed the threshold.)"
        else:
            if not context_blocks:
                context_str = "(No relevant structural matches found in Working Memory. Answer based on general knowledge if necessary.)"

            llm_answer = self.generate_llm_response(query, context_str, llm_endpoint)
            response = llm_answer

            # Feedback loop: write the heuristic resolution back to Valkey
            resolution_id = f"resolution-{hash(query)}"
            self.wm.add_document(
                resolution_id, f"Query: {query}\nResolution: {llm_answer}"
            )

        return result.verified, response, coverage, result.matched_documents