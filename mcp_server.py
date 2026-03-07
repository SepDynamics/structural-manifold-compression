#!/usr/bin/env python3
"""Structural Manifold Compression – MCP Codebase Index Server.

Exposes the full repository as an O(1) interactive codebase brain
for AI assistants via the Model Context Protocol (stdio transport).

Design principles (aligned with the chaos-proxy signal-first pipeline):
  1. **Instant startup** – no heavy imports at module level.
  2. **Signal-first ingest** – store raw text *and* structural byte-window
     signatures in Valkey so queries can be answered at sub-second latency.
  3. **Bounded streaming** – per-file byte cap + exclusion rules prevent OOM.
  4. **Hazard-gated retrieval** – structural tension thresholds gate which
     matches are surfaced (mirrors the chaos proxy collapse detection).

Tools exposed:
  • ingest_repo          – stream-ingest repo into Valkey (text + signatures)
  • search_code          – keyword / regex search across indexed files
  • get_file             – read a single indexed file
  • list_indexed_files   – list indexed paths (glob filter)
  • get_index_stats      – Valkey db health + doc count
  • compute_signature    – compress text to manifold signatures
  • verify_snippet       – hazard-gated snippet verification
  • start_watcher        – live filesystem watcher for IDE saves
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap – ensure local project modules are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# sep_text_manifold lives in the companion `score` repo
_SCORE_SRC = Path("/workspace/sep/score/src")
if _SCORE_SRC.exists() and str(_SCORE_SRC) not in sys.path:
    sys.path.insert(0, str(_SCORE_SRC))

# The C++ extension (sep_quantum) lives in src/
_SRC_DIR = REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# MCP server init (lightweight – no torch / numpy yet)
# ---------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ManifoldCodebaseIndex")

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_valkey_wm = None


def _get_valkey():
    global _valkey_wm
    if _valkey_wm is None:
        from src.manifold.valkey_client import ValkeyWorkingMemory

        _valkey_wm = ValkeyWorkingMemory()
    return _valkey_wm


# ---------------------------------------------------------------------------
# Ingestion config
# ---------------------------------------------------------------------------
DOC_PREFIX = "manifold:docs:"
SIG_PREFIX = "manifold:sig:"
META_KEY = "manifold:meta:ingest"
FILE_LIST_KEY = "manifold:file_list"

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    ".venv-py313-bak",
    ".venv_mamba",
    "node_modules",
    "output",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "egg-info",
    ".eggs",
    "src/build",
}

EXCLUDE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.o",
    "*.a",
    "*.dylib",
    "*.egg-info",
    ".DS_Store",
    "*.mp4",
    "*.webm",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.ico",
    "*.wav",
    "*.mp3",
    "*.arrow",
    "*.safetensors",
    "*.pdf",
    "*.whl",
    "*.tar.gz",
    "*.zip",
}

# Extensions we treat as text (everything else is binary-sig only)
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".rst",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cu",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".makefile",
    ".cmake",
    ".tex",
    ".bib",
    ".bst",
    ".cls",
    ".sty",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".xml",
    ".csv",
    ".tsv",
    ".sql",
    ".r",
    ".R",
    ".go",
    ".rs",
    ".java",
    ".scala",
    ".kt",
    ".swift",
    ".rb",
    ".pl",
    ".pm",
    ".lua",
    ".zig",
    ".nim",
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
    ".jsonl",
    ".ndjson",
    ".env",
    ".cff",
}

# Also treat files without extension that look like config
TEXT_FILENAMES = {
    "Makefile",
    "Dockerfile",
    "LICENSE",
    "Procfile",
    "Gemfile",
    "Rakefile",
    "Vagrantfile",
    ".gitignore",
    ".dockerignore",
}

DEFAULT_MAX_BYTES = 512_000  # 512 KB per file cap


def _should_skip(path: Path) -> bool:
    """Return True if *path* should be excluded from ingest."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    name = path.name
    for pat in EXCLUDE_PATTERNS:
        if fnmatch(name, pat):
            return True
    return False


def _is_text(path: Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def _read_capped(path: Path, cap: int) -> bytes:
    with path.open("rb") as f:
        return f.read(cap)


# ---------------------------------------------------------------------------
# Signature helpers (lazy-loaded)
# ---------------------------------------------------------------------------
_encoder_ready = None


def _ensure_encoder():
    """Lazy-import the structural encoder; return True if available."""
    global _encoder_ready
    if _encoder_ready is not None:
        return _encoder_ready
    try:
        from sep_text_manifold import encode, native  # noqa: F401

        native.set_use_native(True)
        _encoder_ready = True
    except Exception:
        _encoder_ready = False
    return _encoder_ready


def _compute_sig(data: bytes, precision: int = 3) -> Optional[str]:
    """Compute a single structural signature for a ≥512-byte chunk."""
    if not _ensure_encoder():
        return None
    if len(data) < 512:
        return None
    from sep_text_manifold import encode

    metrics = encode.encode_window(data[:512])
    return encode.signature_from_metrics(
        metrics["coherence"],
        metrics["stability"],
        metrics["entropy"],
        precision=precision,
    )


# ===================================================================
# TOOL: ingest_repo
# ===================================================================
@mcp.tool()
def ingest_repo(
    root_dir: str = ".",
    max_bytes_per_file: int = DEFAULT_MAX_BYTES,
    clear_first: bool = False,
    compute_signatures: bool = True,
) -> str:
    """Stream-ingest the repository into Valkey with bounded memory.

    Stores raw file text for instant retrieval and (optionally) the leading
    structural manifold signature for each file.  Excludes virtual-envs,
    caches, build artifacts, and binary blobs automatically.

    Args:
        root_dir: Directory to ingest relative to repo root (default ".").
        max_bytes_per_file: Byte cap per file (default 512 KB).
        clear_first: Wipe all existing manifold keys before ingesting.
        compute_signatures: Also compute structural byte signatures.

    Returns:
        Summary of ingested files, bytes, and unique signatures.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable on localhost:6379."

    # Avoid .resolve() which follows symlinks/bind-mounts and breaks relative_to
    target = REPO_ROOT / root_dir
    if not target.exists():
        return f"❌ Directory not found: {target}"

    if clear_first:
        for key in v.r.scan_iter("manifold:*"):
            v.r.delete(key)

    text_count = 0
    binary_count = 0
    total_bytes = 0
    sig_count = 0
    skipped = 0
    errors = []
    t0 = time.time()

    # Collect file paths into Valkey sorted set for fast listing
    pipe = v.r.pipeline(transaction=False)

    # Use os.walk with followlinks=True to traverse symlinked dirs (e.g. data/)
    all_files = []
    for dirpath, dirnames, filenames in os.walk(str(target), followlinks=True):
        # Prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            all_files.append(Path(dirpath) / fname)
    all_files.sort()

    for path in all_files:
        if not path.is_file():
            continue
        if _should_skip(path):
            skipped += 1
            continue

        # Compute relative path, preserving symlink structure (data/ not /sep/data/)
        try:
            rel = str(path.relative_to(target))
            if root_dir != ".":
                rel = str(Path(root_dir) / rel)
        except ValueError:
            rel = str(path)

        try:
            raw = _read_capped(path, max_bytes_per_file)
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
            continue

        if not raw:
            continue

        total_bytes += len(raw)

        if _is_text(path):
            text = raw.decode("utf-8", errors="replace")
            pipe.set(f"{DOC_PREFIX}{rel}", text)
            text_count += 1
        else:
            # Binary: store SHA + size placeholder, keep hex for small files
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) <= 4096:
                pipe.set(f"{DOC_PREFIX}{rel}", raw.hex())
            else:
                pipe.set(
                    f"{DOC_PREFIX}{rel}",
                    f"[BINARY sha256={digest} bytes={len(raw)}]",
                )
            binary_count += 1

        # File list entry (sorted set scored by file size for prioritised search)
        pipe.zadd(FILE_LIST_KEY, {rel: len(raw)})

        # Structural signature
        if compute_signatures and len(raw) >= 512:
            sig = _compute_sig(raw)
            if sig:
                pipe.set(f"{SIG_PREFIX}{rel}", sig)
                sig_count += 1

        # Flush pipeline every 200 files to bound memory
        if (text_count + binary_count) % 200 == 0:
            pipe.execute()
            pipe = v.r.pipeline(transaction=False)

    pipe.execute()

    elapsed = time.time() - t0
    meta = {
        "text_files": text_count,
        "binary_files": binary_count,
        "total_bytes": total_bytes,
        "signatures": sig_count,
        "skipped": skipped,
        "errors": len(errors),
        "elapsed_s": round(elapsed, 2),
        "root": str(target),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    v.r.set(META_KEY, json.dumps(meta))

    err_report = ""
    if errors:
        err_report = f"\nFirst 5 errors:\n" + "\n".join(errors[:5])

    return (
        f"✅ Ingest complete in {elapsed:.1f}s\n"
        f"  Text files : {text_count}\n"
        f"  Binary files: {binary_count}\n"
        f"  Total bytes : {total_bytes:,}\n"
        f"  Signatures : {sig_count}\n"
        f"  Skipped    : {skipped}\n"
        f"  Errors     : {len(errors)}{err_report}"
    )


# ===================================================================
# TOOL: search_code
# ===================================================================
@mcp.tool()
def search_code(
    query: str,
    max_results: int = 20,
    file_pattern: str = "*",
    case_sensitive: bool = False,
) -> str:
    """Search indexed codebase files by keyword or regex pattern.

    Scans all text documents stored in Valkey and returns matching
    file paths with surrounding context lines.

    Args:
        query: Search string or Python regex pattern.
        max_results: Maximum number of matching files to return (default 20).
        file_pattern: Glob pattern to filter file paths (e.g. "*.py", "src/*").
        case_sensitive: Whether the search is case-sensitive.

    Returns:
        Matching files with context snippets showing where the query appears.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error:
        # Fall back to literal search
        escaped = re.escape(query)
        pattern = re.compile(escaped, flags)

    results: List[str] = []
    scanned = 0

    for key in v.r.scan_iter(f"{DOC_PREFIX}*", count=500):
        rel = key[len(DOC_PREFIX) :]
        if file_pattern != "*" and not fnmatch(rel, file_pattern):
            continue

        content = v.r.get(key)
        if not content or content.startswith("[BINARY"):
            continue

        scanned += 1
        matches = list(pattern.finditer(content))
        if not matches:
            continue

        # Build context snippets (3 lines around each match)
        lines = content.split("\n")
        snippets = []
        seen_lines = set()
        for m in matches[:5]:  # cap snippets per file
            # Find line number of match start
            line_start = content[: m.start()].count("\n")
            ctx_start = max(0, line_start - 2)
            ctx_end = min(len(lines), line_start + 3)
            for i in range(ctx_start, ctx_end):
                if i not in seen_lines:
                    seen_lines.add(i)
                    prefix = ">>>" if i == line_start else "   "
                    snippets.append(f"  {prefix} L{i+1}: {lines[i]}")

        hit_text = "\n".join(snippets)
        results.append(
            f"📄 {rel}  ({len(matches)} match{'es' if len(matches)>1 else ''})\n{hit_text}"
        )
        if len(results) >= max_results:
            break

    if not results:
        return f"No matches found for '{query}' (scanned {scanned} files, pattern='{file_pattern}')."

    header = f"Found {len(results)} file(s) matching '{query}' (scanned {scanned}):\n\n"
    return header + "\n\n".join(results)


# ===================================================================
# TOOL: get_file
# ===================================================================
@mcp.tool()
def get_file(path: str) -> str:
    """Read a specific file from the Valkey index.

    Returns the full text content of the file as stored during ingest.
    Use list_indexed_files() first to discover available paths.

    Args:
        path: File path relative to repo root (e.g. "src/manifold/sidecar.py").

    Returns:
        The file content, or an error if not found.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    content = v.r.get(f"{DOC_PREFIX}{path}")
    if content is None:
        # Try a fuzzy match
        candidates = []
        for key in v.r.scan_iter(f"{DOC_PREFIX}*{Path(path).name}*", count=200):
            candidates.append(key[len(DOC_PREFIX) :])
        if candidates:
            suggestion = "\n".join(f"  • {c}" for c in candidates[:10])
            return f"❌ '{path}' not found. Did you mean:\n{suggestion}"
        return f"❌ '{path}' not found in the index."

    # Add line numbers for easy reference
    lines = content.split("\n")
    numbered = "\n".join(f"{i+1:>5} | {line}" for i, line in enumerate(lines))
    return f"📄 {path} ({len(lines)} lines, {len(content)} chars):\n\n{numbered}"


# ===================================================================
# TOOL: list_indexed_files
# ===================================================================
@mcp.tool()
def list_indexed_files(
    pattern: str = "*",
    max_results: int = 200,
) -> str:
    """List files in the Valkey codebase index.

    Args:
        pattern: Glob pattern to filter (e.g. "*.py", "src/manifold/*", "scripts/**").
        max_results: Maximum paths to return.

    Returns:
        Newline-separated list of indexed file paths.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    # Try sorted set first (fast)
    paths = []
    members = v.r.zrangebyscore(FILE_LIST_KEY, "-inf", "+inf", start=0, num=100000)
    if members:
        for m in members:
            if pattern == "*" or fnmatch(m, pattern):
                paths.append(m)
                if len(paths) >= max_results:
                    break
    else:
        # Fallback: scan doc keys
        for key in v.r.scan_iter(f"{DOC_PREFIX}*", count=500):
            rel = key[len(DOC_PREFIX) :]
            if pattern == "*" or fnmatch(rel, pattern):
                paths.append(rel)
                if len(paths) >= max_results:
                    break

    paths.sort()
    if not paths:
        return f"No indexed files match pattern '{pattern}'."

    return f"Indexed files ({len(paths)}):\n" + "\n".join(paths)


# ===================================================================
# TOOL: get_index_stats
# ===================================================================
@mcp.tool()
def get_index_stats() -> str:
    """Return statistics about the current Valkey codebase index.

    Returns:
        A summary of documents, signatures, memory usage, and last ingest time.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    doc_count = 0
    sig_count = 0
    for key in v.r.scan_iter(f"{DOC_PREFIX}*", count=1000):
        doc_count += 1
    for key in v.r.scan_iter(f"{SIG_PREFIX}*", count=1000):
        sig_count += 1

    file_list_size = v.r.zcard(FILE_LIST_KEY) or 0
    db_size = v.r.dbsize()
    info = v.r.info("memory")
    mem_human = info.get("used_memory_human", "?")

    meta_raw = v.r.get(META_KEY)
    meta = json.loads(meta_raw) if meta_raw else {}

    return (
        f"📊 Codebase Index Stats\n"
        f"  Total Valkey keys  : {db_size}\n"
        f"  Indexed documents  : {doc_count}\n"
        f"  File list entries  : {file_list_size}\n"
        f"  Structural sigs   : {sig_count}\n"
        f"  Valkey memory      : {mem_human}\n"
        f"  Last ingest        : {meta.get('timestamp', 'never')}\n"
        f"  Last ingest root   : {meta.get('root', 'N/A')}\n"
        f"  Last text files    : {meta.get('text_files', 0)}\n"
        f"  Last binary files  : {meta.get('binary_files', 0)}\n"
        f"  Last total bytes   : {meta.get('total_bytes', 0):,}\n"
        f"  Last elapsed       : {meta.get('elapsed_s', 0)}s\n"
        f"  Encoder available  : {_ensure_encoder()}"
    )


# ===================================================================
# TOOL: compute_signature
# ===================================================================
@mcp.tool()
def compute_signature(text: str) -> str:
    """Compress text into structural manifold signatures.

    Slides a 512-byte window across the input and computes
    Coherence/Stability/Entropy/Hazard metrics for each window.

    Args:
        text: The raw text string to compress.

    Returns:
        Report with compression ratio, unique signatures, and hazard threshold.
    """
    if not _ensure_encoder():
        return "❌ Native encoder (sep_text_manifold) not available."

    from src.manifold.sidecar import encode_text

    encoded = encode_text(
        text,
        window_bytes=512,
        stride_bytes=384,
        precision=3,
        use_native=True,
    )
    unique_sigs = len(encoded.prototypes)
    bytes_before = encoded.original_bytes
    bytes_after = unique_sigs * 9
    ratio = (bytes_before / bytes_after) if bytes_after else 0.0

    return (
        f"✅ Compressed {bytes_before:,} bytes → {bytes_after} manifold bytes\n"
        f"  Compression ratio  : {ratio:.2f}×\n"
        f"  Unique signatures  : {unique_sigs}\n"
        f"  Total windows      : {len(encoded.windows)}\n"
        f"  Hazard threshold   : {encoded.hazard_threshold:.4f}"
    )


# ===================================================================
# TOOL: verify_snippet
# ===================================================================
@mcp.tool()
def verify_snippet(
    snippet: str,
    coverage_threshold: float = 0.5,
) -> str:
    """Verify a snippet against the codebase manifold index.

    Checks whether a piece of text (e.g. AI-generated code) structurally
    matches patterns already present in the indexed codebase.

    Args:
        snippet: Text snippet to verify (must be ≥512 bytes).
        coverage_threshold: Minimum safe coverage ratio (default 0.5).

    Returns:
        Verification report with hit ratio and matched documents.
    """
    if not _ensure_encoder():
        return "❌ Native encoder not available."

    snippet_bytes = len(snippet.encode("utf-8"))
    if snippet_bytes < 512:
        return f"❌ Snippet too short ({snippet_bytes} bytes). Need ≥512."

    v = _get_valkey()
    index = v.get_or_build_index()
    if index is None:
        return "❌ No manifold index available. Run ingest_repo first."

    from src.manifold.sidecar import verify_snippet as _verify

    result = _verify(
        text=snippet,
        index=index,
        coverage_threshold=coverage_threshold,
        window_bytes=512,
        stride_bytes=384,
        precision=3,
        use_native=True,
    )

    status = "✅ VERIFIED" if result.verified else "❌ FAILED"
    return (
        f"Status: {status}\n"
        f"  Safe coverage   : {result.coverage*100:.2f}%\n"
        f"  Raw match ratio : {result.match_ratio*100:.2f}%\n"
        f"  Gated hits      : {result.gated_hits}/{result.total_windows}\n"
        f"  Matched docs    : {', '.join(result.matched_documents) if result.matched_documents else 'None'}"
    )


# ===================================================================
# TOOL: get_file_signature
# ===================================================================
@mcp.tool()
def get_file_signature(path: str) -> str:
    """Get the structural manifold signature for a specific indexed file.

    Args:
        path: File path relative to repo root.

    Returns:
        The c_s_e signature string, or error if not computed.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    sig = v.r.get(f"{SIG_PREFIX}{path}")
    if sig:
        return f"📐 {path} → signature: {sig}"

    # Try computing on the fly
    content = v.r.get(f"{DOC_PREFIX}{path}")
    if content is None:
        return f"❌ '{path}' not in index."

    raw = content.encode("utf-8", errors="replace")
    computed = _compute_sig(raw)
    if computed:
        v.r.set(f"{SIG_PREFIX}{path}", computed)
        return f"📐 {path} → signature: {computed} (freshly computed)"
    return f"❌ File too short or encoder unavailable for signature computation."


# ===================================================================
# TOOL: search_by_structure
# ===================================================================
@mcp.tool()
def search_by_structure(
    signature: str,
    tolerance: float = 0.05,
    max_results: int = 20,
) -> str:
    """Find files whose structural signature is close to the given one.

    Uses numeric proximity on the c/s/e components within *tolerance*.

    Args:
        signature: Target signature like "c0.418_s0.760_e0.957".
        tolerance: Max per-component deviation (default 0.05).
        max_results: Cap on results.

    Returns:
        List of files with similar structural profiles.
    """
    # Parse target signature
    m = re.match(r"c([\d.]+)_s([\d.]+)_e([\d.]+)", signature)
    if not m:
        return f"❌ Invalid signature format. Expected 'cX.XXX_sX.XXX_eX.XXX'."
    tc, ts, te = float(m.group(1)), float(m.group(2)), float(m.group(3))

    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    matches = []
    for key in v.r.scan_iter(f"{SIG_PREFIX}*", count=500):
        rel = key[len(SIG_PREFIX) :]
        sig_val = v.r.get(key)
        if not sig_val:
            continue
        sm = re.match(r"c([\d.]+)_s([\d.]+)_e([\d.]+)", sig_val)
        if not sm:
            continue
        sc, ss, se = float(sm.group(1)), float(sm.group(2)), float(sm.group(3))
        dist = max(abs(tc - sc), abs(ts - ss), abs(te - se))
        if dist <= tolerance:
            matches.append((dist, rel, sig_val))
            if len(matches) >= max_results * 3:
                break

    matches.sort()
    matches = matches[:max_results]

    if not matches:
        return f"No files within tolerance {tolerance} of {signature}."

    lines = [f"Files structurally similar to {signature} (±{tolerance}):"]
    for dist, rel, sv in matches:
        lines.append(f"  {sv}  Δ={dist:.4f}  {rel}")
    return "\n".join(lines)


# ===================================================================
# TOOL: start_watcher
# ===================================================================
_active_observer = None


@mcp.tool()
def start_watcher(
    watch_dir: str = ".",
    max_bytes_per_file: int = DEFAULT_MAX_BYTES,
) -> str:
    """Start a filesystem watcher that auto-ingests file saves into Valkey.

    Args:
        watch_dir: Directory to monitor (relative to repo root).
        max_bytes_per_file: Byte cap per file.

    Returns:
        Status message.
    """
    global _active_observer
    if _active_observer is not None:
        return "✅ Watcher already running."

    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        return "❌ watchdog not installed. Run: pip install watchdog"

    target = REPO_ROOT / watch_dir
    if not target.exists():
        return f"❌ Directory not found: {target}"

    cap = max_bytes_per_file

    class _Handler(FileSystemEventHandler):
        def _ingest(self, src_path: str) -> None:
            path = Path(src_path)
            if not path.is_file() or _should_skip(path):
                return
            try:
                rel = str(path.relative_to(REPO_ROOT))
            except ValueError:
                return
            try:
                raw = _read_capped(path, cap)
                if not raw:
                    return
                vk = _get_valkey()
                if _is_text(path):
                    text = raw.decode("utf-8", errors="replace")
                    vk.r.set(f"{DOC_PREFIX}{rel}", text)
                else:
                    digest = hashlib.sha256(raw).hexdigest()
                    vk.r.set(
                        f"{DOC_PREFIX}{rel}",
                        f"[BINARY sha256={digest} bytes={len(raw)}]",
                    )
                vk.r.zadd(FILE_LIST_KEY, {rel: len(raw)})
                if len(raw) >= 512:
                    sig = _compute_sig(raw)
                    if sig:
                        vk.r.set(f"{SIG_PREFIX}{rel}", sig)
            except Exception:
                pass

        def on_modified(self, event):
            if not event.is_directory:
                self._ingest(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                self._ingest(event.src_path)

    handler = _Handler()
    observer = Observer()
    observer.schedule(handler, str(target), recursive=True)
    observer.daemon = True
    observer.start()
    _active_observer = observer

    return f"✅ Watcher started for {target} (cap {cap} bytes)"


# ===================================================================
# TOOL: inject_fact
# ===================================================================
@mcp.tool()
def inject_fact(fact_id: str, fact_text: str) -> str:
    """Inject a new fact into the Working Memory codebook (zero-shot learning).

    Args:
        fact_id: Unique identifier for the fact.
        fact_text: The factual text to assimilate.

    Returns:
        Confirmation message.
    """
    v = _get_valkey()
    if not v.ping():
        return "❌ Valkey not reachable."

    v.add_document(fact_id, fact_text)
    return f"🚀 Fact '{fact_id}' injected into the Dynamic Semantic Codebook."


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    mcp.run()
