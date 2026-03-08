# Unit Economics: Scenario Notes

This file is a scenario analysis, not a benchmark result.

## Measured Ceiling From The Current Repo

Level 1: measured result

The strongest current corpus benchmark in this repository shows:

- about `1.81x` token compression on the locked 25-paper arXiv pilot
- serialized manifold artifacts larger than the source corpus bytes
- strong retrieval signal, but weak storage reduction

These measurements set the current ceiling for any cost discussion in this repo.

## What This Means For Cost Claims

Level 2: observed behavior

- The current implementation is not yet ready for strong storage-savings claims.
- The current pilot does support discussion of retrieval behavior under bounded reconstruction.

## Scenario Analysis

Level 4: research direction

If future versions of the system achieve materially stronger compression while preserving retrieval quality, a smaller retrieval representation could reduce:

- index storage
- query-time retrieval footprint
- hardware requirements for bounded reconstruction

Those outcomes are not established by the current benchmark artifacts.

## Safe Pitch Line

"If future benchmarks show stronger compression without losing retrieval quality, structural retrieval may reduce storage and retrieval overhead for large document collections."
