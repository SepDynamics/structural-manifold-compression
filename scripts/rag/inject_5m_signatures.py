#!/usr/bin/env python3
import sys
import time
from pathlib import Path

from src.manifold.valkey_client import ValkeyWorkingMemory
from src.manifold.sidecar import ManifoldIndex

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    print("Connecting to live Valkey instance...")
    valkey = ValkeyWorkingMemory()
    if not valkey.ping():
        print("Valkey is offline.")
        return

    print("Fetching/Building initial index...")
    index = valkey.get_cached_index()
    if index is None:
        index = ManifoldIndex(
            meta={
                "window_bytes": 512,
                "stride_bytes": 384,
                "precision": 3,
                "hazard_percentile": 0.8,
            },
            signatures={},
            documents={},
        )

    print("Injecting 5,000,000 unique dummy signatures...")
    signatures = index.signatures

    start_time = time.time()
    batch_size = 500000
    for i in range(5000000):
        sig_val = f"{i*0.001:.3f}_0.100_0.500"
        signatures[sig_val] = {
            "prototype": {
                "text": f"dummy{i}",
                "doc_id": "dummy",
                "byte_start": 0,
                "byte_end": 10,
            },
            "occurrences": [],
            "hazard": {"min": 0.5, "max": 0.5, "sum": 0.5, "count": 1, "mean": 0.5},
        }
        if i > 0 and i % batch_size == 0:
            print(f"Injected {i} signatures. Elapsed: {time.time() - start_time:.2f}s")

    print(f"Total signatures in dict: {len(signatures)}")
    index.meta["total_signatures"] = len(signatures)
    index.meta["total_windows"] = len(signatures)

    print("Storing back to Valkey...")
    valkey.store_cached_index(index)

    print("Done! Valkey is saturated.")


if __name__ == "__main__":
    main()
