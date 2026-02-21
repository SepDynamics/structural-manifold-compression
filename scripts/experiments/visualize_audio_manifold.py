#!/usr/bin/env python3
"""Objective Audio Topology Visualizer

Generates 5 distinct 1-second audio waveforms (Silence, White Noise, Sine, Square, Impulse),
processes them blindly through the O(1) C++ Manifold Engine (no audio encoders), and plots
the resulting topological topologies (Entropy, Coherence, Stability) using Matplotlib.
"""

import sys
import math
import struct
import wave
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure local modules are accessible
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure score/src is accessible for the native bindings
SCORE_CANDIDATES = [
    REPO_ROOT / "score" / "src",
    REPO_ROOT.parent / "score" / "src",
    REPO_ROOT / "src",
]
for candidate in SCORE_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sep_text_manifold import native
from scripts.experiments.manifold_compression_eval import sliding_windows


def generate_test_signals(
    out_dir: Path, sample_rate: int = 44100, duration: float = 1.0
) -> dict:
    """Generate 5 objective edge-case audio signals as raw 16-bit PCM WAVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration)
    freq = 440.0  # Standard A4 tone

    # 1. Silence (Flatline 0)
    silence = np.zeros(num_samples, dtype=np.int16)

    # 2. White Noise (Maximum entropy scatter)
    noise = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)

    # 3. Sine Wave (Perfect continuous geometry)
    t = np.arange(num_samples) / sample_rate
    sine = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    # 4. Square Wave (Sharp mathematical discontinuities)
    square = (np.sign(np.sin(2 * np.pi * freq * t)) * 32767).astype(np.int16)

    # 5. Impulse (Silence -> single max spike -> silence)
    impulse = np.zeros(num_samples, dtype=np.int16)
    impulse[num_samples // 2] = 32767

    signals = {
        "Silence": silence,
        "White Noise": noise,
        "Sine Wave": sine,
        "Square Wave": square,
        "Impulse": impulse,
    }

    paths = {}
    for name, data in signals.items():
        filename = name.lower().replace(" ", "_") + ".wav"
        filepath = out_dir / filename
        with wave.open(str(filepath), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(data.tobytes())
        paths[name] = filepath
        print(f"Generated {name}: {filepath.name}")

    return paths


def extract_manifold_topology(
    file_path: Path, window_bytes: int = 2048, stride_bytes: int = 1024
) -> dict:
    """Analyze the audio file byte-by-byte through the native spatial engine."""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    native.set_use_native(True)

    entropy_series = []
    coherence_series = []
    stability_series = []

    # Ensure native engine is ready
    if not hasattr(native, "analyze_window_batch"):
        raise RuntimeError(
            "Native C++ engine 'analyze_window_batch' required but unavailable."
        )

    pending_chunks = []
    for _, chunk in sliding_windows(
        audio_bytes, window_bytes=window_bytes, stride_bytes=stride_bytes
    ):
        chunk_b = bytes(chunk)
        # Pad with 0s if window is slightly short at the tail
        if len(chunk_b) < window_bytes:
            chunk_b += b"\x00" * (window_bytes - len(chunk_b))
        pending_chunks.append(chunk_b)

        # Batch execution for speed
        if len(pending_chunks) >= 256:
            metrics_batch = native.analyze_window_batch(pending_chunks)
            for m in metrics_batch:
                entropy_series.append(float(m.get("entropy", 0.0)))
                coherence_series.append(float(m.get("coherence", 0.0)))
                stability_series.append(float(m.get("stability", 0.0)))
            pending_chunks.clear()

    if pending_chunks:
        metrics_batch = native.analyze_window_batch(pending_chunks)
        for m in metrics_batch:
            entropy_series.append(float(m.get("entropy", 0.0)))
            coherence_series.append(float(m.get("coherence", 0.0)))
            stability_series.append(float(m.get("stability", 0.0)))

    return {
        "entropy": entropy_series,
        "coherence": coherence_series,
        "stability": stability_series,
    }


def plot_topologies(results: dict, output_path: Path):
    """Generate a multi-panel plot comparing the topological dimensions across signals."""
    metrics = ["entropy", "coherence", "stability"]
    colors = {
        "Silence": "gray",
        "White Noise": "red",
        "Sine Wave": "blue",
        "Square Wave": "purple",
        "Impulse": "orange",
    }

    # 3 subplots: one for each metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "O(1) Topological Extractor: Objective Audio Manifold Analysis",
        fontsize=16,
        fontweight="bold",
    )

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for signal_name, data in results.items():
            series = data[metric]
            ax.plot(
                series,
                label=signal_name,
                color=colors[signal_name],
                alpha=0.8,
                linewidth=1.5,
            )

        ax.set_title(f"Dimension: {metric.capitalize()}", fontsize=12)
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, linestyle="--", alpha=0.6)

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Time (Sliding Window Index)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ High-res plot saved to: {output_path}")


def main():
    print("=== Objective Audio Topology Visualizer ===")

    out_dir = REPO_ROOT / "data" / "raw_audio_testbed"
    plot_path = REPO_ROOT / "docs" / "assets" / "audio_topology.png"

    print("\n[1/3] Synthesizing edge-case waveforms...")
    # Generate signals
    audio_paths = generate_test_signals(out_dir)

    print("\n[2/3] Extracting native structural geometries (Bypassing Encoders)...")
    results = {}
    for name, path in audio_paths.items():
        topology = extract_manifold_topology(path, window_bytes=2048, stride_bytes=1024)
        results[name] = topology

        # Display averages for sanity check
        avg_ent = sum(topology["entropy"]) / max(1, len(topology["entropy"]))
        avg_coh = sum(topology["coherence"]) / max(1, len(topology["coherence"]))
        print(f"  {name:12s} -> Entropy: {avg_ent:>5.2f} | Coherence: {avg_coh:>5.2f}")

    print("\n[3/3] Rendering Matplotlib Topologies...")
    plot_topologies(results, plot_path)


if __name__ == "__main__":
    main()
