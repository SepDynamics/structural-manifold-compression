#!/usr/bin/env python3
"""
System Sensor (O(1) Kernel Telemetry Pathologist)
Phase 10: Dynamic State Stabilization in High-Entropy Streams

Streams live /proc/[pid]/stat or dmesg output into the C++ continuous
spatial manifold encoder. Monitors the Structural Tension (Lyapunov Proxy)
of the operating system's raw execution state. Only queries the LLM when
a topological rupture is detected (e.g., memory leak, syntax bomb, or
kernel panic trajectory).

Usage:
    python scripts/rag/system_sensor.py --pid 1234
    python scripts/rag/system_sensor.py --dmesg
"""

import sys
import time
import argparse
import subprocess
import requests
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from src.manifold.valkey_client import ValkeyWorkingMemory


class SystemsPathologist:
    def __init__(self, target_pid=None, use_dmesg=False):
        self.target_pid = target_pid
        self.use_dmesg = use_dmesg
        self.wm = ValkeyWorkingMemory()

        if not self.wm.ping():
            print(
                "Warning: Valkey Working Memory offline. High-speed caching disabled."
            )

        self.telemetry_buffer = []
        # Keep track of history to detect topological ruptures
        self.orbit_history = []

        print(f"🔬 AGI-Lite Systems Pathologist Initialized.")
        if target_pid:
            print(
                f"📡 Attaching O(1) Manifold Encoder to PID: {target_pid} (/proc/{target_pid}/stat)"
            )
        elif use_dmesg:
            print(f"📡 Attaching O(1) Manifold Encoder to Kernel Ring Buffer (dmesg)")

    def _get_proc_stat(self):
        """Read continuous telemetry from /proc/[pid]/stat"""
        try:
            with open(f"/proc/{self.target_pid}/stat", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"❌ Process {self.target_pid} not found or terminated.")
            sys.exit(1)
        except Exception as e:
            return str(e)

    def _get_dmesg_tail(self):
        """Read the tail of the kernel ring buffer"""
        try:
            result = subprocess.run(["dmesg", "-t"], capture_output=True, text=True)
            lines = result.stdout.strip().split("\n")
            return "\n".join(lines[-10:])
        except Exception as e:
            return str(e)

    def extract_spatial_topology(self, raw_telemetry: str):
        """
        Pass telemetry through the real C++ continuous spatial manifold encoder.
        Extracts physical bounding metrics directly from the temporal byte sequence.
        """
        from src.manifold.sidecar import encode_text

        # Ensure we have enough data for the engine window
        if len(raw_telemetry.encode("utf-8")) < 256:
            raw_telemetry = (raw_telemetry + " " * 256)[:256]

        try:
            # O(1) mathematical execution against the stream
            encoded = encode_text(
                raw_telemetry,
                window_bytes=256,
                stride_bytes=128,
                precision=3,
                use_native=True,
            )

            if not encoded.windows:
                return "c0.000_s0.000_e0.000", 0.0, 0.0, 0.0

            window = encoded.windows[-1]
            sig = window.signature

            # The format is c<coherence>_s<stability>_e<entropy>
            parts = sig.split("_")

            c = float(parts[0][1:]) if len(parts) > 0 else 0.0
            s = float(parts[1][1:]) if len(parts) > 1 else 0.0
            e = float(parts[2][1:]) if len(parts) > 2 else 0.0

            return sig, c, s, e

        except Exception as err:
            return "c0.000_s0.000_e0.000", 0.0, 0.0, 0.0

    def run_inference_loop(self):
        """Indefinite low-overhead polling loop"""
        poll_interval = 0.5 if self.target_pid else 2.0

        while True:
            # 1. Gather Telemetry
            data = self._get_proc_stat() if self.target_pid else self._get_dmesg_tail()
            if not data:
                time.sleep(poll_interval)
                continue

            self.telemetry_buffer.append(data)
            if len(self.telemetry_buffer) > 20:
                self.telemetry_buffer.pop(0)

            # 2. Extract Topological Manifold (O(1) execution)
            current_context = "\n".join(self.telemetry_buffer[-5:])
            sig, c, s, e = self.extract_spatial_topology(current_context)

            # Record orbit
            self.orbit_history.append((c, s, e))
            if len(self.orbit_history) > 10:
                self.orbit_history.pop(0)

            # 3. Calculate Structural Tension (Variational Free Energy)
            # Tension is high if the current stability deviates widely from the recent orbit mean
            if len(self.orbit_history) == 10:
                mean_stability = sum(orb[1] for orb in self.orbit_history) / 10.0
                structural_tension = abs(s - mean_stability)

                # Write to Valkey for the UI Orbital Stability map
                try:
                    self.wm.r.set("manifold:chaos_proxy", str(structural_tension))
                except:
                    pass

                sys.stdout.write(
                    f"\r[O(1) Sensor] Target: {self.target_pid or 'dmesg'} | Tensor: {sig} | Tension: {structural_tension:.4f}  "
                )
                sys.stdout.flush()

                # 4. The Pathologist Wake-Up Call
                if structural_tension > 0.6:
                    print(
                        f"\n\n🛑 [Three-Body Rupture] Structural Tension Spiked: {structural_tension:.4f}"
                    )
                    print(
                        f"⚠️ Operating System trajectory deviating from stable orbit. Waking LLM adapter..."
                    )

                    self.diagnose_anomaly(current_context, sig)

                    # Cool down to prevent API spam
                    print(
                        "\n[Sensor] Cooling down for 10 seconds before resuming O(1) stream..."
                    )
                    time.sleep(10)
                    self.orbit_history.clear()

            time.sleep(poll_interval)

    def diagnose_anomaly(self, telemetry: str, signature: str):
        """Invoke the Latent Semantic Adapter to diagnose physical OS ruptures."""
        prompt = f"""You are the AGI-Lite Systems Pathologist. 
The O(1) physics engine just detected a Topological Rupture (High Variational Free Energy) in the OS telemetry stream.

The current continuous manifold signature is: {signature}
The raw OS telemetry at the point of rupture is:

{telemetry}

Diagnose the trajectory of this system. Is it a memory leak, a runaway thread, an I/O block, or a kernel panic vector?
Reply with EXACTLY a 2-sentence diagnostic warning. Do not add fluff.
"""
        print("\n[Pathologist] Querying Llama3-70b via Latent Semantic Adapter...")
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:70b",
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            elapsed = time.time() - start
            if response.status_code == 200:
                ans = response.json().get("response", "").strip()
                msg = f"\n🚨 [Pathologist Diagnosis - {elapsed:.2f}s]\n{ans}\n"
                print(msg)

                # Log to the active watcher so it displays in the Gradio UI
                try:
                    with open(
                        REPO_ROOT / "watcher_output.txt", "a", encoding="utf-8"
                    ) as f:
                        f.write(msg + "---\n")
                except Exception:
                    pass
            else:
                print(f"[Error] LLM Adapter returned {response.status_code}")
        except Exception as e:
            print(f"[Error] LLM Adapter offline: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGI-Lite Systems Pathologist Sensor")
    parser.add_argument(
        "--pid", type=int, help="Target process ID to monitor (/proc/[pid]/stat)"
    )
    parser.add_argument(
        "--dmesg", action="store_true", help="Monitor the kernel ring buffer (dmesg)"
    )
    args = parser.parse_args()

    if not args.pid and not args.dmesg:
        print("Please specify a target to stream: --pid [ID] or --dmesg")
        sys.exit(1)

    sensor = SystemsPathologist(target_pid=args.pid, use_dmesg=args.dmesg)
    try:
        sensor.run_inference_loop()
    except KeyboardInterrupt:
        print("\n\n[Sensor] O(1) Streaming Terminated.")
        sys.exit(0)
