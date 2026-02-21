#!/usr/bin/env python3
"""
Plots the Thousand-Brains Consensus vs Single-Stream Baseline.
Reads the training logs to extract Perplexity (Structural Tension) per Epoch.
"""

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def parse_log_file(filepath: Path) -> list[float]:
    perplexities = []
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return perplexities

    content = filepath.read_text(encoding="utf-8")
    for line in content.splitlines():
        if "Perplexity:" in line:
            # Example line: Eval Loss: 8.1247 | Perplexity: 3376.81
            match = re.search(r"Perplexity:\s+([0-9\.]+)", line)
            if match:
                perplexities.append(float(match.group(1)))
    return perplexities


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = repo_root / "output" / "production_scale"

    accel_log = logs_dir / "training_accelerated.log"
    single_log = logs_dir / "training_single_stream.log"

    accel_ppl = parse_log_file(accel_log)
    single_ppl = parse_log_file(single_log)

    if not accel_ppl or not single_ppl:
        print("Missing log data. Assure both logs are present.")
        sys.exit(1)

    epochs = list(range(1, 11))

    # Ensure they have the same length for plotting
    min_len = min(len(accel_ppl), len(single_ppl), 10)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs[:min_len],
        single_ppl[:min_len],
        marker="x",
        linestyle="--",
        color="#ff7f50",
        label="1-Stream Baseline (No Consensus)",
        linewidth=2,
    )
    plt.plot(
        epochs[:min_len],
        accel_ppl[:min_len],
        marker="o",
        linestyle="-",
        color="#2f6fff",
        label="3-Stream 'Thousand Brains' Consensus",
        linewidth=3,
    )

    plt.title(
        "Structural Tension vs Hebbian Epochs\nThousand Brains Consensus Acceleration",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Hebbian Epochs", fontsize=12)
    plt.ylabel("Structural Tension / Prediction Error ($I_t$)", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    # Annotate final drops
    plt.annotate(
        f"Final Tension: {single_ppl[-1]:.1f}",
        xy=(10, single_ppl[-1]),
        xytext=(7, single_ppl[-1] + 200),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
    )

    plt.annotate(
        f"Accelerated Tension: {accel_ppl[-1]:.1f}",
        xy=(10, accel_ppl[-1]),
        xytext=(7, accel_ppl[-1] - 300),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
    )

    # Save the plot
    out_path = repo_root / "docs" / "images" / "consensus_tension_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to: {out_path}")


if __name__ == "__main__":
    main()
