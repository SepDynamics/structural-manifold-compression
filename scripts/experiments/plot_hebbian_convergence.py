#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path


def extract_loss(run_dir):
    state_path = Path(run_dir) / "checkpoint-final" / "trainer_state.json"
    if not state_path.exists():
        # Fallback to checking logs if trainer_state isn't formatted as expected
        return []
    with open(state_path) as f:
        data = json.load(f)
    return [log["loss"] for log in data.get("log_history", []) if "loss" in log]


backprop_loss = extract_loss("output/benchmarks/hebbian_run_backprop")
hebbian_loss = extract_loss("output/benchmarks/hebbian_run_local")

plt.figure(figsize=(10, 6))
plt.plot(backprop_loss, label="Global Backprop (O(N))", color="red", alpha=0.7)
plt.plot(hebbian_loss, label="Local Hebbian FEP (O(1))", color="blue", alpha=0.9)
plt.title("Training Convergence: Backpropagation vs. Local Hebbian Plasticity")
plt.xlabel("Training Steps")
plt.ylabel("Loss (Variational Free Energy Proxy)")
plt.legend()
plt.grid(True, alpha=0.3)

out_path = Path("output/benchmarks/figures/hebbian_convergence.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path)
print(f"Plot saved to {out_path}")
