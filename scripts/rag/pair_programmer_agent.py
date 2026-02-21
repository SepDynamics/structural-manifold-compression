#!/usr/bin/env python3
"""
Autonomous Pair Programmer (Phase 6 Agentic Loop)
Watches the codebase as you type. Upon detecting an FEP spike (structural tension),
it triggers the Latent Semantic Adapter to query the local LLM (Ollama) and pre-generates
code completions based entirely on spatial motif mappings.
"""

import sys
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(1)

from src.manifold.router import TripartiteRouter

OLLAMA_URL = "http://localhost:11434/api/generate"


class PairProgrammerHandler(FileSystemEventHandler):
    def __init__(self, router: TripartiteRouter):
        super().__init__()
        self.router = router
        self.last_trigger = time.time()
        self.active_window_content = ""

    def _process_event(self, event):
        if event.is_directory or not event.src_path.endswith(".py"):
            return

        # Debounce
        now = time.time()
        if now - self.last_trigger < 3.0:
            return

        filepath = Path(event.src_path)
        try:
            time.sleep(0.1)  # Wait for flush
            content = filepath.read_text(encoding="utf-8")
            if len(content) < 50:
                return

            print(
                f"\n[Pair Programmer] Detected structure change in {filepath.name}... Computing FEP Spike."
            )

            # Take the last 512 bytes as the active typing window
            self.active_window_content = content[-512:]
            active_window = self.active_window_content

            self.last_trigger = now
            verified, response, coverage, _ = self.router.process_query(
                query=active_window,
                hazard_threshold=0.8,
                coverage_threshold=0.5,
                llm_endpoint=OLLAMA_URL,
                use_native=True,
            )

            if not verified:
                msg = f"⚡ [FEP Spike | Coverage: {coverage:.2f}%] Pre-generating contextual completion...\n🤖 LLM Suggestion:\n{response}\n"
                print(msg)
                try:
                    with open(
                        REPO_ROOT / "watcher_output.txt", "a", encoding="utf-8"
                    ) as f:
                        f.write(msg + "\n---\n")
                except Exception:
                    pass
            else:
                print(
                    f"✅ Structural tension low (Coverage: {coverage:.2f}%). No completion needed."
                )

        except Exception as e:
            print(f"ERROR: {e}")

    def on_modified(self, event):
        self._process_event(event)

    def on_created(self, event):
        self._process_event(event)


def recursive_self_correction_loop(handler: PairProgrammerHandler):
    """
    Background thread that autonomously probes the Recency Buffer when the user
    stops typing. It hunts for architectural inconsistencies using the heuristic LLM.
    """
    while True:
        time.sleep(5)
        now = time.time()
        # If the user hasn't typed in 5 seconds and we have an active buffer
        if now - handler.last_trigger > 5.0 and handler.active_window_content:
            # Sort entries by last_seen to get the true Recency Buffer
            sorted_entries = sorted(
                handler.router.codebook.entries.values(),
                key=lambda e: e.last_seen,
                reverse=True,
            )
            recent_sigs = [e.signature for e in sorted_entries][:50]
            if len(recent_sigs) < 1:
                continue

            print(
                "\n[Self-Correction] User idle. Probing Semantic Recency Buffer for Predictive Hazards..."
            )

            # Check Valkey for Bi-Directional Prompt Binding
            bound_tokens = handler.router.wm.r.get("manifold:prompt_binding")
            if bound_tokens:
                print(
                    "\n[Bi-Directional Action] Cortex Binding detected! Freezing Semantic Adapter Recency List."
                )
                active_tokens = bound_tokens.split(",")
            else:
                # Use the Dynamic Codebook to find semantic tokens
                active_tokens = handler.router.codebook.get_activation_buffer(
                    recent_sigs, 20
                )

            # Phase 9: Recursive Error Correction (Future Orbit Simulation)
            import random

            # Use the trailing ~512 bytes of the active window for context
            context_tail = (
                handler.active_window_content[-512:]
                if len(handler.active_window_content) >= 512
                else handler.active_window_content
            )

            try:
                # Simulate 10 Future Orbits to test manifold stability
                simulated_orbits = []
                # To simulate perturbative noise without a temperature parameter on the C++ side,
                # we inject minor whitespace variations into the tail context.
                for _ in range(10):
                    jitter = " " * random.randint(0, 3)
                    test_context = context_tail + jitter

                    # Extract spatial signature proxy (Simulated Manifold extraction for the background daemon)
                    import hashlib

                    val = int(hashlib.md5(test_context.encode()).hexdigest(), 16)
                    predicted_signature = (
                        f"c0.{val % 99}_s0.{(val // 100) % 99}_e0.{(val // 10000) % 99}"
                    )
                    simulated_orbits.append(predicted_signature)

                features = set(simulated_orbits)
                # If we get >7 unique signatures from 10 minor perturbations, the manifold is shattered
                if len(features) > 7:
                    rupture_msg = f"🛑 **[Three-Body Rupture]** Predictive Chaos detected (Lyapunov proxy: {len(features)/10.0}). Trajectory unstable. Aborting LLM generation to prevent semantic hallucination.\n"
                    print(rupture_msg)
                    try:
                        with open(
                            REPO_ROOT / "watcher_output.txt", "a", encoding="utf-8"
                        ) as f:
                            f.write(rupture_msg + "---\n")
                    except Exception:
                        pass

                    # Abort the LLM query as it will likely hallucinate given the high physical entropy
                    handler.active_window_content = ""
                    continue

            except Exception as e:
                print(f"[Orbit Simulation Failed]: {e}")
                pass

            prompt = f"""You are the Recursive Self-Correction agent. 
The user represents a biological cortical column currently paused on this local state:

{handler.active_window_content}

The current active spatial vocabulary bounded to this region is:
{active_tokens}

Detect any architectural inconsistencies, type violations, or structural logic errors 
that the user is about to make or just made. If none, reply EXACTLY with 'SAFE'.
If you find a hazard, reply with a 'Predictive Hazard Warning:' followed by a 1-sentence warning.
"""
            try:
                import requests

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3:70b",
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    ans = response.json().get("response", "").strip()
                    if "SAFE" not in ans and len(ans) > 5:
                        msg = f"⚠️ [Predictive Hazard Warning]\n{ans}\n"
                        print(msg)
                        try:
                            with open(
                                REPO_ROOT / "watcher_output.txt", "a", encoding="utf-8"
                            ) as f:
                                f.write(msg + "\n---\n")
                        except Exception:
                            pass
            except Exception:
                pass

            # Clear window so we don't spam the same hazard over and over
            handler.active_window_content = ""


def main():
    print("Initializing Autonomous Pair Programmer Daemon...")
    router = TripartiteRouter()
    if not router.wm.ping():
        print("❌ CRITICAL: Valkey Work Memory is offline.")
        sys.exit(1)

    watch_path = REPO_ROOT / "scripts"  # Monitor just scripts to avoid noise
    print(f"👀 Watching codebase for typing patterns in {watch_path}...")

    event_handler = PairProgrammerHandler(router)
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=True)
    observer.start()

    # Start the true autonomous self-correction loop
    correction_thread = threading.Thread(
        target=recursive_self_correction_loop, args=(event_handler,), daemon=True
    )
    correction_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping Pair Programmer...")
    observer.join()


if __name__ == "__main__":
    main()
