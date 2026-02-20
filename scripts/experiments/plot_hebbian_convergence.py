#!/usr/bin/env python3
import matplotlib.pyplot as plt
from pathlib import Path
import re


def extract_loss_from_log(run_dir):
    """Parses the custom mamba trainer output text file for loss metrics."""
    log_path = Path(run_dir) / "training.log"
    losses = []

    if not log_path.exists():
        print(f"Warning: Could not find {log_path}")
        return losses

    content = log_path.read_text()

    # Regex to match "Epoch 1 | Loss: 218.4642"
    pattern = re.compile(r"Epoch \d+ \| Loss: ([\d\.]+)")

    for match in pattern.finditer(content):
        losses.append(float(match.group(1)))

    return losses


# Assuming you piped the output to a file. If not, we will need to re-run the training
# commands and pipe them to training.log like this: `... > output/benchmarks/hebbian_run_backprop/training.log`
backprop_loss = extract_loss_from_log("output/benchmarks/hebbian_run_backprop")
hebbian_loss = extract_loss_from_log("output/benchmarks/hebbian_run_local")

# If the lists are empty, it means we didn't save the logs to disk.
# We'll hardcode the values you just pasted in the console to generate the graph immediately.
if not backprop_loss:
    print("Logs not found on disk, using the values from the console trace...")
    backprop_loss = [218.4642, 219.3156, 219.1870]
    hebbian_loss = [206.9172, 218.6924, 220.3913]

plt.figure(figsize=(10, 6))
plt.plot(
    backprop_loss, label="Global Backprop (O(N))", color="red", alpha=0.7, marker="o"
)
plt.plot(
    hebbian_loss, label="Local Hebbian FEP (O(1))", color="blue", alpha=0.9, marker="x"
)

plt.title("Training Convergence: Backpropagation vs. Local Hebbian Plasticity")
plt.xlabel("Training Epochs")
plt.ylabel("Loss (Variational Free Energy Proxy)")
plt.xticks([0, 1, 2], ["Epoch 1", "Epoch 2", "Epoch 3"])
plt.legend()
plt.grid(True, alpha=0.3)

out_path = Path("output/benchmarks/figures/hebbian_convergence.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path)
print(f"Plot successfully saved to {out_path}")
