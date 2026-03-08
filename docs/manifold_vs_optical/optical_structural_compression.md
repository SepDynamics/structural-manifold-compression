# Structural vs. Optical Compression Evaluation Plan

This document is an evaluation plan and hypothesis note. It does not report the current corpus benchmark result.

## Measured Results Available Elsewhere

Level 1: measured result

For the current corpus benchmark, use the committed artifacts in:

- `results/qa_results.json`
- `results/manifold_ablation.json`
- `results/manifold_reconstruction_sweep.json`

Those artifacts support a retrieval claim. They do not yet support a strong compression claim.

## Evaluation Goal

Level 4: research direction

Evaluate whether structural signatures can become a useful alternative to larger text or optical intermediate representations for long-context retrieval workloads.

## Working Hypotheses

Level 3: hypothesis

1. Deduplicated structural signatures may reduce representation size relative to raw text or optical intermediates.
2. Hazard and repetition summaries may support useful verification signals.
3. Structural representations may be operationally simpler than optical pipelines in some settings.

These are hypotheses to test, not established repo-level conclusions.

## Evaluation Checklist

1. Use a fixed manifest and shared sample order across compared systems.
2. Record compression ratio, retrieval quality, verification quality, and storage footprint separately.
3. Keep optical and structural pipelines on the same subset when comparing them.
4. Treat storage reduction and retrieval retention as separate claims.
5. Reject any comparison that does not preserve identical sample boundaries.

## Interpretation Rules

Level 2: observed behavior

- If retrieval remains strong but storage does not shrink, the result is a retrieval win, not a compression win.
- If storage shrinks but retrieval collapses, the representation is not yet useful for LLM retrieval workflows.

## Output Artifacts To Produce

- compression summary JSON
- retrieval summary JSON
- verification summary JSON
- plot-ready sweep outputs for precision, stride, and hazard thresholds
