#!/usr/bin/env python3
"""
Autonomous Sensory Layer (The Tripartite Daemon Watcher)
Continuously monitors a directory for file changes, encodes them into structural
manifold signatures, and streams the updates to the active Valkey memory.
"""

import sys
import time
import argparse

from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.manifold.valkey_client import ValkeyWorkingMemory

# Provide access to local project modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VALID_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".wav",
    ".dat",
}
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "node_modules", "output", "build"}

wm = ValkeyWorkingMemory()


def should_process(file_path: Path) -> bool:
    """Check if the file has a valid extension and is not in an excluded directory."""
    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    for part in file_path.parts:
        if part in EXCLUDE_DIRS:
            return False
    return True


class TripartiteSensoryHandler(FileSystemEventHandler):

    def on_modified(self, event):
        if not event.is_directory:
            self._process_file(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self._process_file(Path(event.src_path))

    def _process_file(self, file_path: Path):
        if not should_process(file_path):
            return

        try:
            # We add a slight delay to ensure file writing is complete before reading
            time.sleep(0.1)

            # Multi-Modal Boundary: Read raw bytes if audio/data, else text
            if file_path.suffix.lower() in [".wav", ".dat"]:
                raw_bytes = file_path.read_bytes()
                if not raw_bytes:
                    return
                # Convert raw bytes to a storable hex string representation for Valkey
                text = raw_bytes.hex()
            else:
                text = file_path.read_text(encoding="utf-8")
                if not text.strip():
                    return

            doc_id = str(file_path.relative_to(REPO_ROOT))
            print(f"👁️ [SENSORY INGESTION] Encoding changes in {doc_id}...")

            # The encoding and hazard logic is implicitly handled inside the index builder later
            # For the autonomous layer, we simply push the raw environment bytes to Valkey.
            # The Tripartite CLI will compute the structural signatures dynamically upon querying,
            # or the MCP server will rebuild the index. By passing it to Valkey, the engine "knows" it.
            if wm.ping():
                wm.add_document(doc_id, text)
                print(f"✅ Assimalated {doc_id} into Working Memory.")
            else:
                print("❌ Valkey connection failed. Is sep-valkey running?")

        except Exception as e:
            print(f"⚠️ Error ingesting {file_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Tripartite Sensory Daemon")
    parser.add_argument(
        "--watch-dir", type=Path, default=REPO_ROOT, help="Directory to monitor"
    )
    args = parser.parse_args()

    watch_path = args.watch_dir.resolve()
    print(f"🚀 Starting Tripartite Daemon Watcher on {watch_path}")
    if not wm.ping():
        print("❌ Warning: Valkey instance is not responding on port 6379!")

    event_handler = TripartiteSensoryHandler()
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping Watcher...")

    observer.join()


if __name__ == "__main__":
    main()
