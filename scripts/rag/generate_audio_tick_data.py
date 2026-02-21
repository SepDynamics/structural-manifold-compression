#!/usr/bin/env python3
import sys
import wave
import struct
import math
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

sys.path.append("/sep/score/src")

from sep_text_manifold import native
from scripts.experiments.manifold_compression_eval import sliding_windows


def generate_wav(file_path: Path, is_chaotic: bool, duration: float = 2.0):
    sample_rate = 44100
    frequency = 4400.0  # High freq for tight repeating motifs

    with wave.open(str(file_path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(int(sample_rate * duration)):
            if is_chaotic:
                # White noise
                value = random.randint(-32768, 32767)
            else:
                # Stable Square Wave
                value = 32767 if (i // 10) % 2 == 0 else -32768

            data = struct.pack("<h", value)
            wav.writeframesraw(data)


def analyze_audio_tension(file_path: Path) -> float:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    native.set_use_native(True)
    ruptures = []

    pending = []
    for _, chunk in sliding_windows(audio_bytes, window_bytes=4096, stride_bytes=2048):
        pending.append(bytes(chunk))
        if len(pending) >= 256:
            if hasattr(native, "analyze_window_batch"):
                metrics_batch = native.analyze_window_batch(pending)
                for m in metrics_batch:
                    ruptures.append(float(m["rupture"]))
            pending.clear()

    if pending and hasattr(native, "analyze_window_batch"):
        metrics_batch = native.analyze_window_batch(pending)
        for m in metrics_batch:
            ruptures.append(float(m["rupture"]))

    return sum(ruptures) / len(ruptures) if ruptures else 0.0


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    stable_path = out_dir / "stable_audio.wav"
    chaotic_path = out_dir / "chaotic_audio.wav"

    print("Synthesizing audio streams...")
    generate_wav(stable_path, is_chaotic=False)
    generate_wav(chaotic_path, is_chaotic=True)

    print("\n[Multimodal Structural Classification Proof]")
    print(
        f"Analyzing raw bytes through O(1) C++ Manifold Engine (No Audio Encoders Used)...\n"
    )

    stable_rupture = analyze_audio_tension(stable_path)
    chaotic_rupture = analyze_audio_tension(chaotic_path)

    print(f"🎵 Stable Audio (Square Wave) Topological Rupture: {stable_rupture:.4f}")
    print(f"🔊 Chaotic Audio (White Noise) Topological Rupture: {chaotic_rupture:.4f}")

    if stable_rupture > chaotic_rupture * 1.5:
        print(
            "\n✅ SUCCESS: Engine mathematically recognized the unnatural square-wave motif as a massive Topological Rupture purely from byte geometry!"
        )
    else:
        print("\n⚠️ WARNING: Rupture delta lacked significance.")
