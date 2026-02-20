#!/usr/bin/env python3
"""Plot FEP and needle benchmark curves for the Phase 1 manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_fep(fep_path: Path, output: Path) -> None:
    data = load_json(fep_path)
    baseline = data.get("baseline_seconds", 0.0)
    spike = data.get("spike_to_assimilate_seconds", 0.0)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.bar(["baseline", "spike->assimilate"], [baseline, spike], color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Seconds")
    ax.set_title("FEP Learning Latency")
    for idx, value in enumerate([baseline, spike]):
        ax.text(idx, value, f"{value:.4f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_needle(needle_path: Path, output: Path) -> None:
    data = load_json(needle_path)
    ttft = data.get("ttft_seconds", 0.0)
    vram = data.get("vram_bytes", 0.0) / (1024 * 1024)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.bar(["TTFT (s)", "VRAM (MB)"], [ttft, vram], color=["#54a24b", "#e45756"])
    ax.set_title("Needle-in-Haystack Baseline Cost")
    for idx, value in enumerate([ttft, vram]):
        ax.text(idx, value, f"{value:.3f}" if idx == 0 else f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FEP + needle benchmark curves")
    parser.add_argument("--fep", type=Path, required=True, help="Path to fep_learning_test.json")
    parser.add_argument("--needle", type=Path, required=True, help="Path to needle_haystack.json")
    parser.add_argument("--out-dir", type=Path, default=Path("output/benchmarks/figures"))
    args = parser.parse_args()

    plot_fep(args.fep, args.out_dir / "fep_latency.png")
    plot_needle(args.needle, args.out_dir / "needle_cost.png")


if __name__ == "__main__":
    main()
