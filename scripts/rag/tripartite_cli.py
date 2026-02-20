#!/usr/bin/env python3
"""
The Tripartite Daemon CLI
An interactive loop that prioritizes deterministic structural ANN retrieval,
only falling back to an LLM ("The ADHD Heuristic") when the Hazard Tension spikes.
"""

import sys
import argparse
from pathlib import Path

# Provide access to local project modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter

router = TripartiteRouter()


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

    if not router.wm.ping():
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

            verified, response, coverage, matched_documents = router.process_query(
                query=query,
                hazard_threshold=args.hazard_threshold,
                coverage_threshold=args.coverage_threshold,
                llm_endpoint=args.llm_endpoint,
            )

            print(
                f"\n[Sensory Engine] Tension/Hazard Gate: {coverage:.1f}% safe matches."
            )

            # 2. Heuristic Resolution (Tension Gate)
            if verified:
                print(
                    "✅ ---------------------------------- [DETERMINISTIC REFLEX] ---------------------------------- ✅"
                )
                print(
                    "Hazard Tension is LOW. Bypass LLM and inject direct structural match:\n"
                )
                print(response)
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

                print(f"🤖 LLM Insight:\n{response}")
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
