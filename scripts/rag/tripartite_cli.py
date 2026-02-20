#!/usr/bin/env python3
"""
The Tripartite Daemon CLI
An interactive loop that prioritizes deterministic structural ANN retrieval,
only falling back to an LLM ("The ADHD Heuristic") when the Hazard Tension spikes.
"""

import sys
import json
import requests
import argparse
from pathlib import Path

# Provide access to local project modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.valkey_client import ValkeyWorkingMemory
from src.manifold.sidecar import verify_snippet

wm = ValkeyWorkingMemory()


def generate_llm_response(query: str, context: str, endpoint: str) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="Tripartite Daemon CLI")
    parser.add_argument(
        "--llm-endpoint",
        default="http://localhost:11434/api/generate",
        help="Ollama or OpenAI compatible generate endpoint",
    )
    parser.add_argument(
        "--hazard-threshold",
        type=float,
        default=0.8,
        help="Strictness of the tension gate",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5,
        help="Required verified coverage to bypass LLM",
    )
    args = parser.parse_args()

    print(f"🚀 Tripartite Daemon CLI Initialized")
    print(
        f"Hazard Gate: <= {args.hazard_threshold} | LLM Endpoint: {args.llm_endpoint}"
    )
    print("Type 'exit' or 'quit' to close.\n")

    if not wm.ping():
        print(
            "❌ CRITICAL: Valkey Work Memory is offline (port 6379). Please start it."
        )
        sys.exit(1)

    while True:
        try:
            query = input("Tripartite> ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            index = wm.get_or_build_index()
            if index is None:
                print(
                    "⚠️ Working Memory is currently empty. The Sensory Watcher needs to ingest data first."
                )
                continue

            # 1. Deterministic Reflex
            result = verify_snippet(
                text=query,
                index=index,
                coverage_threshold=args.coverage_threshold,
                hazard_threshold=args.hazard_threshold,
                window_bytes=512,
                stride_bytes=384,
                precision=3,
                use_native=True,
            )

            print(
                f"\n[Sensory Engine] Tension/Hazard Gate: {result.coverage*100:.1f}% safe matches."
            )

            # Grab context from Valkey based on matched documents
            context_blocks = []
            for doc_id in result.matched_documents:
                doc_text = wm.r.get(f"{wm.doc_prefix}{doc_id}")
                if doc_text:
                    context_blocks.append(
                        f"--- Document: {doc_id} ---\n{doc_text[:2000]}"
                    )  # Truncate to save context window

            context_str = "\n".join(context_blocks)

            # 2. Heuristic Resolution (Tension Gate)
            if result.verified:
                print(
                    "✅ ---------------------------------- [DETERMINISTIC REFLEX] ---------------------------------- ✅"
                )
                print(
                    "Hazard Tension is LOW. Bypass LLM and inject direct structural match:\n"
                )
                if not context_blocks:
                    print(
                        "(No documents securely resolved, but coverage math passed the threshold.)"
                    )
                else:
                    for block in context_blocks:
                        print(block)
                print(
                    "------------------------------------------------------------------------------------------------\n"
                )
            else:
                print(
                    "⚠️ -------------------------------- [HEURISTIC RESOLUTION] --------------------------------- ⚠️"
                )
                print(
                    "Hazard Tension is HIGH. Motif collision detected or abstract query. Engaging LLM Fallback...\n"
                )

                if not context_blocks:
                    context_str = "(No relevant structural matches found in Working Memory. Answer based on general knowledge if necessary.)"

                llm_answer = generate_llm_response(
                    query, context_str, args.llm_endpoint
                )
                print(f"🤖 LLM Insight:\n{llm_answer}")

                # Feedback loop: write the heuristic resolution back to Valkey
                resolution_id = f"resolution-{hash(query)}"
                wm.add_document(
                    resolution_id, f"Query: {query}\nResolution: {llm_answer}"
                )
                print(
                    f"\n[Daemon] Synthesized resolution assimilated into Working Memory as '{resolution_id}'."
                )
                print(
                    "------------------------------------------------------------------------------------------------\n"
                )

        except KeyboardInterrupt:
            print("\nExiting Tripartite Daemon.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
