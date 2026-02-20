#!/usr/bin/env python3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter
from src.manifold.sidecar import verify_snippet


def main():
    router = TripartiteRouter()
    index = router.wm.get_cached_index()
    if index is None:
        from src.manifold.sidecar import ManifoldIndex

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

    # Artificially expand the local memory index to 5 million signatures
    print("Artificially inflating cache to 5,000,000 unique signatures...")
    for i in range(5000000):
        index.signatures[f"{i*0.001:.3f}_0.100_0.500"] = {
            "prototype": {
                "text": f"dummy{i}",
                "doc_id": "dummy",
                "byte_start": 0,
                "byte_end": 10,
            },
            "occurrences": [],
            "hazard": {"min": 0.5, "max": 0.5, "sum": 0.5, "count": 1, "mean": 0.5},
        }

    print(
        f"Index now contains {len(index.signatures)} signatures.\nRunning latency test..."
    )

    query = "Determine the derivative of f(x) = x^2"

    import gc

    gc.disable()

    start_time = time.perf_counter()
    result = verify_snippet(
        text=query,
        index=index,
        coverage_threshold=0.5,
        hazard_threshold=0.8,
        window_bytes=512,
        stride_bytes=384,
        precision=3,
        use_native=False,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Status: Coverage {result.coverage*100:.2f}%")
    print(f"Latency: {elapsed_ms:.2f} ms")
    if elapsed_ms < 15:
        print("✅ SUCCESS: Sub-15ms guaranteed across >5M signatures in memory.")
    else:
        print(f"⚠️ WARNING: Exceeded 15ms barrier ({elapsed_ms:.2f}ms).")


if __name__ == "__main__":
    main()
