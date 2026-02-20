#!/usr/bin/env python3
import wave
import struct
import math
from pathlib import Path

# Provide a sample continuous byte stream masquerading as a .wav file
# to prove the structural manifold natively encodes any byte topology.
file_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw_audio"
file_path.mkdir(parents=True, exist_ok=True)
wav_file = file_path / "ambient_motifs.wav"

sample_rate = 44100
duration = 2.0  # seconds
frequency = 440.0  # Hz

# We create a simple sine wave which has a highly deterministic, repeating topology
# The structural tension (QFH) will recognize this as highly stable (high Coherence)
with wave.open(str(wav_file), "w") as wav:
    wav.setnchannels(1)  # mono
    wav.setsampwidth(2)  # 2 bytes per sample
    wav.setframerate(sample_rate)

    for i in range(int(sample_rate * duration)):
        value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
        data = struct.pack("<h", value)
        wav.writeframesraw(data)

print(f"Generated raw audio byte payload at {wav_file}")
