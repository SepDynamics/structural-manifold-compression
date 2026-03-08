# AGI-Lite: Predictive Coding and Topological Manifolds for Non-von Neumann Cognitive Architecture

## Evidence Status

This document should be read as a hypothesis and prior-results note, not as a settled research conclusion.

### Established in the repo
- Structural manifold encoding, indexing, and verification primitives exist.
- The repository now contains a leakage-aware corpus benchmark harness with frozen questions, neutral document ids, bounded reconstruction, and a shuffled-manifold control.

### Prior reported results
- The benchmarks described below were reported on earlier internal experiments and narrower datasets.
- They are useful as background measurements, but they do not establish the new large-corpus compression claim.

### Not yet established
- 200-paper arXiv compression-retention performance
- QA parity with a baseline RAG system at corpus scale
- Any claim of general replacement for transformer context handling

## 1. Introduction: Beyond the Transformer Plateau

### The Quadratic Collapse
The contemporary trajectory of artificial intelligence has been overwhelmingly dominated by the scaling laws of transformer-based Large Language Models (LLMs). However, this paradigm faces a real computational wall: the cost of self-attention grows quadratically with sequence length. Our empirical "Needle-in-a-Haystack" measurements illustrate this in one tested setup: for 81,967 bytes (81KB) of context, a baseline GPT-2 transformer consumed 652.88 MB of VRAM.

### The AGI-Lite Thesis
To explore alternatives to quadratic attention costs, this document outlines the **Tripartite Architecture**, a design inspired by Thousand Brains Theory (Hawkins, 2021) and the Free-Energy Principle (Friston). The working hypothesis is that structural topological manifolds, State Space Models, and Valkey grid cells may support lower-overhead continuous inference than standard full-context transformer pipelines in the tested settings.

---

## 2. Structural Manifold Physics (The C++ Layer)

### Waveform Digestion
This project investigates whether byte-level structure can preserve useful information that semantic subword tokenization discards. Instead of starting from subword tokens, the C++ engine (`sep_quantum.so`) scans byte-stream waveforms via a sliding window and measures continuous topological folds.

### Quantum Failure Hazard (QFH)
As the byte stream is processed, the engine calculates the structural geometry of the data. We formalize this physics via the mathematics of Structural Tension ($I_t$). Phase shifts (Coherence), sequence repetition decay (Stability), and chaotic permutations (Entropy) are tracked continuously. When a topological manifold breaks structural constraints—such as an abrupt syntax violation or architectural inconsistency—it generates a "Reflex Spike" (High Variational Free Energy). This Quantum Failure Hazard (QFH) dictates exactly *when* the system needs to pay attention to novel reality.

---

## 3. The Tripartite Cognitive Engine

AGI-Lite deconstructs the monolithic Transformer into three specialized cortical layers:

### 3.1 Long-Term Memory (SSM)
The foundational sensory prediction engine is a Mamba State Space Model (SSM). Unlike Transformers, the SSM maintains a fixed-size recurrent state ($h_t$). In model design terms this keeps recurrent state size constant with respect to sequence length; end-to-end system scaling still requires empirical validation.

### 3.2 Working Memory (Valkey Grid Cells)
Acting as the system's spatial reference frame, the architecture routes structural signatures into Valkey memory. In earlier internal scale tests, the system indexed over 5,000,000 manifold signatures with reported sub-10ms lookup speed in the tested setup.

### 3.3 The Thalamic Adapter
The Latent Semantic Thalamic Adapter bridges the gap between pure mathematical topology and human-readable semantics. By freezing a "Recency List" of 50 active grid-cell tokens (e.g., `[matrix, function, matrix, log]`), the adapter queries a local, high-variance top-down generator (like Llama3) *only* when the structural tension spikes. The intended effect is to reduce LLM context bloat and constrain the generator to a narrower evidence window than standard raw-passage fallback.

### 3.4 Thousand Brains Voting
We implemented a `VotingProcessor` inside the SSM proxy to mimic cortical-column-style consensus. Instead of a single monolithic forward pass, the data traverses 3 parallel columns. Earlier internal measurements reported roughly **22% faster** FEP error resolution than the single-stream baseline used in that experiment.

---

## 4. Empirical Convergence (Escaping Backpropagation)

### 4.1 The Hebbian Descent Curve
One internal experiment replaced global weight backpropagation (`loss.backward()`) with a Local Hebbian Descent rule. In that 10-epoch run, perplexity decreased from **3376** to **892** through local, asynchronous updates.

### 4.2 Biological Stabilization
To prevent the Hebbian process from diverging into logit explosions, we mathematically define our optimization via Prediction Error Normalization (binding the update steps bounds) alongside Oja's Weight Decay implementation. \
The update logic: $\Delta \theta_l \propto \epsilon_l - \gamma \theta_l$ \
This strict regularization is intended to reduce drift and instability in continuous updates. Broader claims about eliminating catastrophic forgetting remain unvalidated at large scale.

---

## 5. Generality and Agency

### 5.1 The Multi-Modal Boundary
To probe whether the engine computes structural tension on raw physical signals rather than only on text-like inputs, we applied the architecture to acoustic waves. We analyzed **22,052 audio windows** from a raw 440Hz binary `.wav` input stream without using audio-specific tokenizers.

### 5.2 Proactive Architectural Autonomy
The `pair_programmer_agent.py` script demonstrates an interactive anomaly-monitoring workflow. Running as a backend daemon, it tracks live code edits, maps them against manifold memory, and surfaces high-tension events in real time.
