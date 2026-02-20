#!/usr/bin/env python3
"""
Autonomous Pair Programmer (Phase 6 Agentic Loop)
Watches the codebase as you type. Upon detecting an FEP spike (structural tension),
it triggers the Latent Semantic Adapter to query the local LLM (Ollama) and pre-generates
code completions based entirely on spatial motif mappings.
"""

import sys
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter


class PairProgrammerHandler(FileSystemEventHandler):
    def __init__(self, router: TripartiteRouter):
        super().__init__()
        self.router = router
        self.last_trigger = 0.0

    def on_modified(self, event):
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
            active_window = content[-512:]

            self.last_trigger = now
            verified, response, coverage, _ = self.router.process_query(
                query=active_window,
                hazard_threshold=0.8,
                coverage_threshold=0.5,
                llm_endpoint="http://localhost:11434/api/generate",
                use_native=True,
            )

            if not verified:
                print(
                    f"⚡ [FEP Spike | Coverage: {coverage:.2f}%] Pre-generating contextual completion..."
                )
                print(f"🤖 LLM Suggestion:\n{response}\n")
            else:
                print(
                    f"✅ Structural tension low (Coverage: {coverage:.2f}%). No completion needed."
                )

        except Exception as e:
            pass


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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping Pair Programmer...")
    observer.join()


if __name__ == "__main__":
    main()
