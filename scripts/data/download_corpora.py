#!/usr/bin/env python3
"""Download Fox + OmniDocBench corpora and regenerate manifests/text dumps."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "benchmark_corpus"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.prepare_benchmarks import (  # noqa: E402
    load_tokenizer,
    prepare_fox,
    prepare_omnidocbench,
    write_manifest,
)


def download_snapshot(
    repo_id: str,
    target_dir: Path,
    allow_patterns: Iterable[str] | None,
    token: str | None,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        token=token,
        tqdm_class=None,
    )
    return Path(snapshot_path)


def stage_fox(
    repo_id: str,
    download_dir: Path,
    raw_root: Path,
    token: str | None,
) -> Path:
    print(f"[fox] downloading dataset '{repo_id}' into {download_dir}")
    snapshot = download_snapshot(repo_id, download_dir, allow_patterns=None, token=token)
    zip_path = next(snapshot.rglob("focus_benchmark_test.zip"), None)
    if zip_path is None:
        raise FileNotFoundError("Could not locate focus_benchmark_test.zip in the downloaded Fox snapshot.")
    fox_zip_dir = raw_root / "Fox_benchmark_data"
    fox_zip_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = fox_zip_dir / zip_path.name
    if not dest_zip.exists() or zip_path.stat().st_mtime > dest_zip.stat().st_mtime:
        print(f"[fox] copying {zip_path} -> {dest_zip}")
        shutil.copy2(zip_path, dest_zip)
    extract_root = raw_root / "focus_benchmark_test"
    if extract_root.exists():
        print(f"[fox] existing extract found at {extract_root}; removing to ensure a clean state")
        shutil.rmtree(extract_root)
    print(f"[fox] extracting {dest_zip} into {raw_root}")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(raw_root)
    return extract_root


def stage_omnidoc(
    repo_id: str,
    download_dir: Path,
    raw_root: Path,
    token: str | None,
) -> Path:
    print(f"[omnidoc] downloading dataset '{repo_id}' into {download_dir}")
    snapshot = download_snapshot(repo_id, download_dir, allow_patterns=None, token=token)
    # The repo ships as OmniDocBench/<files>; copy the folder verbatim.
    source_root = None
    for candidate in snapshot.glob("*"):
        if candidate.name.lower().startswith("omnidocbench") and candidate.is_dir():
            source_root = candidate
            break
    if source_root is None:
        source_root = snapshot
    dest_root = raw_root / source_root.name
    print(f"[omnidoc] syncing {source_root} -> {dest_root}")
    if dest_root.exists():
        shutil.rmtree(dest_root)
    shutil.copytree(source_root, dest_root)
    return dest_root


def prepare_manifests(
    download_fox: bool,
    download_omni: bool,
    tokenizer_name: str,
    manifest_name: str,
) -> None:
    tokenizer = load_tokenizer(tokenizer_name)
    combined_records: List = []

    if download_fox:
        fox_root = DATA_ROOT / "fox"
        fox_raw = fox_root / "raw" / "focus_benchmark_test"
        if not fox_raw.exists():
            raise FileNotFoundError(f"Fox raw directory not found: {fox_raw}. Run without --prepare-only first.")
        print(f"[prepare] generating Fox manifests from {fox_raw}")
        fox_records = prepare_fox(fox_raw, fox_root, tokenizer)
        write_manifest(fox_root / "metadata" / manifest_name, fox_records)
        combined_records.extend(fox_records)

    if download_omni:
        omni_root = DATA_ROOT / "omnidocbench"
        omni_raw = omni_root / "raw" / "OmniDocBench"
        if not omni_raw.exists():
            raise FileNotFoundError(f"OmniDocBench raw directory not found: {omni_raw}. Run without --prepare-only first.")
        print(f"[prepare] generating OmniDocBench manifests from {omni_raw}")
        omni_records = prepare_omnidocbench(omni_raw, omni_root, tokenizer)
        write_manifest(omni_root / "metadata" / manifest_name, omni_records)
        combined_records.extend(omni_records)

    if combined_records:
        combined_manifest_dir = DATA_ROOT / "metadata"
        combined_manifest_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(combined_manifest_dir / manifest_name, combined_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download + prepare Fox and OmniDocBench corpora.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("fox", "omnidoc"),
        default=("fox", "omnidoc"),
        help="Datasets to download/prepare.",
    )
    parser.add_argument("--fox-repo", default="UCASLCL/Fox", help="Hugging Face dataset repo id for Fox.")
    parser.add_argument(
        "--omnidoc-repo",
        default="opendatalab/OmniDocBench",
        help="Hugging Face dataset repo id for OmniDocBench.",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".hf-cache", help="Cache directory for snapshots.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use for token counts.")
    parser.add_argument("--manifest-name", type=str, default="text_manifest.jsonl", help="Manifest filename.")
    parser.add_argument("--prepare-only", action="store_true", help="Skip downloads and only regenerate manifests.")
    parser.add_argument("--download-only", action="store_true", help="Download datasets but skip manifest regeneration.")
    parser.add_argument("--hf-token", type=str, help="Explicit Hugging Face token (falls back to HF_TOKEN env var).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.datasets)
    token = args.hf_token or os.environ.get("HF_TOKEN")

    cache_root = args.cache_dir.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    fox_requested = "fox" in selected
    omni_requested = "omnidoc" in selected

    if not args.prepare_only:
        if fox_requested:
            fox_download_dir = cache_root / "fox"
            fox_raw_root = DATA_ROOT / "fox" / "raw"
            fox_raw_root.mkdir(parents=True, exist_ok=True)
            stage_fox(args.fox_repo, fox_download_dir, fox_raw_root, token)
        if omni_requested:
            omni_download_dir = cache_root / "omnidoc"
            omni_raw_root = DATA_ROOT / "omnidocbench" / "raw"
            omni_raw_root.mkdir(parents=True, exist_ok=True)
            stage_omnidoc(args.omnidoc_repo, omni_download_dir, omni_raw_root, token)

    if not args.download_only:
        prepare_manifests(
            download_fox=fox_requested,
            download_omni=omni_requested,
            tokenizer_name=args.tokenizer,
            manifest_name=args.manifest_name,
        )

    print("[done] datasets ready under data/benchmark_corpus")


if __name__ == "__main__":
    main()
