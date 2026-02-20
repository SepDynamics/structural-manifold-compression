# Predictive Coding Integration Plan for AGI-Lite

## 1. Overview and Objective
Transition AGI-Lite from global backprop training to a continuous, locally updating Predictive Coding (PC) network. This integrates the Free-Energy Principle by replacing global error signals with local, asynchronous Hebbian-style updates and by fusing Structural Tension (I_t) with Variational Free Energy (F).

## 2. Tripartite Mapping to Predictive Coding Hierarchy
- **Level 0 (Bottom-Up Evidence / Sensory Input):** Native C++ manifold encoder (`sep_quantum.so`) emitting structural motifs.
- **Level 1 (Local Prediction & State Tracking):** Mamba SSM (`scripts/training/mamba_ssm_trainer.py`) maintaining local state and predicting next motifs.
- **Level N (Top-Down Priors / High-Level Cortex):** LLM/Transformer (`scripts/rag/tripartite_cli.py`, `src/manifold/router.py`) sending top-down predictions on failure.
- **Synaptic Junction (Error Resolution Buffer):** Valkey Working Memory, reconciling bottom-up motifs with top-down priors.

## 3. Structural Tension as Variational Free Energy
**Goal:** Redefine the hazard/tension trigger as free-energy (prediction error) accumulation.

- **Prediction Error at level l:**
  - ε_l = x_l - f(x_{l+1}, θ_l)
- **Variational Free Energy:**
  - F = 1/2 Σ ε_l^T Σ_l^{-1} ε_l

**Implementation direction:**
- Extend `src/core/qfh.cpp` and `src/manifold/router.py` so the hazard threshold is computed from ε, not just phase shift.
- When F spikes, trigger top-down inference and contextual reconciliation in Valkey.

## 4. Implementation Milestones

### Milestone 1: Free Energy Calculation in Router
**Target:** `src/manifold/router.py`
- Add a `FreeEnergyCalculator` to compute ε between expected and retrieved motifs.
- Use F as the primary gating signal.

### Milestone 2: Local Hebbian Update Loops in SSM
**Target:** `scripts/training/mamba_ssm_trainer.py`
- Replace global `loss.backward()` with local weight updates:
  - Δθ_l ∝ ε_l · ∂f/∂θ_l
- Enable on-the-fly updates for streaming data.

### Milestone 3: Bi-Directional Predictive Routing
**Targets:** `src/manifold/router.py`, `scripts/rag/tripartite_cli.py`
- Enable continuous top-down priors into Valkey.
- Persist bottom-up motif evidence alongside priors for reconciliation.

### Milestone 4: Latent Semantic Adapter (Recency Buffer)
**Targets:** `scripts/inference/dynamic_codebook.py`, `src/manifold/router.py`
**Goal:** Replace raw text RAG contexts with constrained biological vocabulary lists during FEP spikes.
- **Step 1:** Modify `dynamic_codebook.py` to maintain a rolling "Activation Buffer" of the top 50 highly active tokens corresponding to the most recent spatial signatures.
- **Step 2:** Refactor `router.py`'s LLM generation call (`generate_llm_response`). Instead of pasting 5,000 raw text tokens into the prompt, inject the active tokens as a semantic context vector or soft-prompt, drastically reducing compute overhead and mimicking biological cortical states (Thalamus/Global Workspace).

## 5. Immediate Next Step
Begin Milestone 1: implement free-energy computation and gating in `src/manifold/router.py` with minimal changes to existing query flow.
