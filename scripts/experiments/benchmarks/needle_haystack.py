#!/usr/bin/env python3
"""Needle-in-haystack long-context benchmark for manifold vs baseline model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


def build_corpus(text_root: Path, needle: str) -> str:
    chunks = []
    for path in sorted(text_root.rglob("*.txt")):
        chunks.append(path.read_text(encoding="utf-8"))
    if not chunks:
        raise FileNotFoundError("No .txt files found in text_root")
    corpus = "\n\n".join(chunks)
    insertion = f"\n\n{needle}\n\n"
    midpoint = len(corpus) // 2
    return corpus[:midpoint] + insertion + corpus[midpoint:]


def measure_ttft(model, tokenizer, prompt: str, device: torch.device) -> Dict[str, float]:
    max_len = min(tokenizer.model_max_length, 1024)
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    with torch.no_grad():
        _ = model.generate(**tokens, max_new_tokens=1)
    ttft = time.time() - start
    vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {"ttft_seconds": ttft, "vram_bytes": vram}


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-haystack long-context test")
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--needle", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/needle_haystack.json"))
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    corpus = build_corpus(args.text_root, args.needle)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metrics = measure_ttft(model, tokenizer, corpus, device)
    output = {
        "model": args.model,
        "device": str(device),
        "corpus_bytes": len(corpus.encode("utf-8")),
        **metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
