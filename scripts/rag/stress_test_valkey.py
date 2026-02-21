#!/usr/bin/env python3
"""
Synthetic Valkey Stress Test
Injects 5,000,000 synthetic manifold signatures into Valkey Working Memory
to verify O(1) Tripartite Router latency at scale.
"""

import sys
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.valkey_client import ValkeyWorkingMemory
from src.manifold.sidecar import ManifoldIndex


def generate_synthetic_index(num_signatures: int = 5_000_000) -> ManifoldIndex:
    print(f"Generating {num_signatures} synthetic modular signatures in RAM...")
    start = time.time()

    signatures = {}
    for i in range(num_signatures):
        sig = f"c{0.5 + (i%5)/10:.1f}_s{0.1 + (i%3)/10:.1f}_e{0.5 + (i%4)/10:.1f}__synth_{i}"

        signatures[sig] = {"prototype": {"text": f"synth_{i}"}}

        if (i + 1) % 1_000_000 == 0:
            print(f"  ...generated {i + 1} signatures")

    print(f"Generation complete in {time.time() - start:.2f}s")

    meta = {
        "total_signatures": num_signatures,
        "total_windows": num_signatures,
        "documents": 1,
        "window_bytes": 512,
        "hazard_threshold": 0.8,
    }

    return ManifoldIndex(meta=meta, signatures=signatures, documents={"doc_synth": 500})


def main():
    wm = ValkeyWorkingMemory()
    if not wm.ping():
        print("❌ CRITICAL: Valkey is offline.")
        sys.exit(1)

    print("WARNING: Clearing existing Valkey index...")
    wm.clear_all()

    TARGET = 5_000_000
    index = generate_synthetic_index(TARGET)

    print("Serializing and pushing to Valkey...")
    start = time.time()
    try:
        wm.store_cached_index(index)
        print(
            f"✅ Successfully injected {TARGET} signatures in {time.time() - start:.2f}s"
        )
    except Exception as e:
        print(f"❌ Failed to store full {TARGET} signatures: {e}")


if __name__ == "__main__":
    main()
