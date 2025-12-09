import subprocess
import json

from pathlib import Path

# Paths to binaries
BIN_DIR = Path("/sep/structural-manifold-compression/src/bin")
MANIFOLD_GENERATOR = BIN_DIR / "manifold_generator"
BYTE_STREAM_MANIFOLD = BIN_DIR / "byte_stream_manifold"


def run_market_manifold(csv_path: str):
    """
    Runs manifold_generator on a CSV file.
    Expects CSV with columns: time,open,high,low,close,volume
    """
    if not MANIFOLD_GENERATOR.exists():
        raise FileNotFoundError(f"Binary not found: {MANIFOLD_GENERATOR}")

    cmd = [str(MANIFOLD_GENERATOR), "--input", csv_path, "--output", ""]  # stdout

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Market manifold failed: {result.stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse market JSON: {result.stdout[:500]}...")


def run_trace_manifold(text_content: str, window_size: int = 256):
    """
    Runs byte_stream_manifold on a string (treated as bytes).
    """
    if not BYTE_STREAM_MANIFOLD.exists():
        raise FileNotFoundError(f"Binary not found: {BYTE_STREAM_MANIFOLD}")

    cmd = [
        str(BYTE_STREAM_MANIFOLD),
        "--window-bits",
        str(window_size),
        "--format",
        "json",
    ]

    # Pass text via stdin
    result = subprocess.run(
        cmd,
        input=text_content,
        capture_output=True,
        text=True,
        encoding="utf-8",  # Ensure we pass utf-8 string
    )

    if result.returncode != 0:
        raise RuntimeError(f"Trace manifold failed: {result.stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse trace JSON: {result.stdout[:500]}...")


if __name__ == "__main__":
    # Smoke test
    print("Testing binaries...")
    if MANIFOLD_GENERATOR.exists():
        print(f"Found {MANIFOLD_GENERATOR}")
    else:
        print(f"MISSING {MANIFOLD_GENERATOR}")

    if BYTE_STREAM_MANIFOLD.exists():
        print(f"Found {BYTE_STREAM_MANIFOLD}")
    else:
        print(f"MISSING {BYTE_STREAM_MANIFOLD}")
