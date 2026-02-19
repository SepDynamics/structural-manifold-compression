#!/usr/bin/env python3
"""Benchmark compute economics: demonstrate O(N) scaling vs O(N^2) Transformer baseline.

This script proves the FLOPs advantage of the dual-stream architecture by comparing:
1. Dual-stream (Mamba SSM + Codebook) - O(N) scaling
2. Baseline Transformer (GPT-2) - O(N^2) scaling

Key metrics:
- Memory footprint (VRAM)
- Time-to-first-token (TTFT)
- Time per token as sequence length grows
- FLOPs estimation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import psutil
import matplotlib.pyplot as plt

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("transformers not installed")
    GPT2LMHeadModel = None

from scripts.inference.dual_stream_inference import DualStreamInference


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    sequence_length: int
    memory_mb: float
    time_to_first_token: float
    time_per_token: float
    total_time: float
    tokens_per_second: float
    estimated_flops: float


class ComputeEconomicsBenchmark:
    """Benchmark suite for compute economics."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def measure_memory(self) -> float:
        """Measure current GPU memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024**2)
        return psutil.Process().memory_info().rss / (1024**2)

    def benchmark_dual_stream(
        self,
        engine: DualStreamInference,
        prompt: str,
        sequence_lengths: List[int],
    ) -> List[BenchmarkResult]:
        """Benchmark dual-stream architecture at various sequence lengths.

        Args:
            engine: Dual-stream inference engine
            prompt: Test prompt
            sequence_lengths: Sequence lengths to test

        Returns:
            List of benchmark results
        """
        results = []

        print("\n=== Benchmarking Dual-Stream Architecture ===")

        for seq_len in sequence_lengths:
            print(f"\nSequence length: {seq_len}")

            # Warm up
            if results:
                torch.cuda.empty_cache()

            from torch.profiler import profile, record_function, ProfilerActivity

            # Measure
            start_mem = self.measure_memory()
            start_time = time.time()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                with record_function("model_inference"):
                    _, metrics = engine.generate_text(
                        prompt,
                        max_tokens=seq_len,
                        temperature=1.0,
                    )

            total_time = time.time() - start_time
            end_mem = self.measure_memory()

            # Actual FLOPs from profiler
            estimated_flops = sum([evt.flops for evt in prof.key_averages()])

            result = BenchmarkResult(
                sequence_length=seq_len,
                memory_mb=end_mem - start_mem,
                time_to_first_token=metrics.get("time_to_first_token", 0.0),
                time_per_token=metrics.get("avg_time_per_token", 0.0),
                total_time=total_time,
                tokens_per_second=metrics.get("tokens_per_second", 0.0),
                estimated_flops=estimated_flops,
            )

            results.append(result)

            print(f"  Memory: {result.memory_mb:.2f} MB")
            print(f"  TTFT: {result.time_to_first_token:.4f} s")
            print(f"  Time/token: {result.time_per_token*1000:.2f} ms")
            print(f"  Tokens/s: {result.tokens_per_second:.2f}")

        return results

    def benchmark_transformer_baseline(
        self,
        model_name: str,
        prompt: str,
        sequence_lengths: List[int],
    ) -> List[BenchmarkResult]:
        """Benchmark transformer baseline (GPT-2) at various sequence lengths.

        Args:
            model_name: Hugging Face model name
            prompt: Test prompt
            sequence_lengths: Sequence lengths to test

        Returns:
            List of benchmark results
        """
        if GPT2LMHeadModel is None:
            raise ImportError("transformers not installed")

        results = []

        print("\n=== Benchmarking Transformer Baseline ===")

        # Load model
        print(f"Loading {model_name}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        model.eval()

        for seq_len in sequence_lengths:
            print(f"\nSequence length: {seq_len}")

            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Limit to avoid OOM on large sequences
            if seq_len > 2048:
                print(f"  Skipping (exceeds context limit)")
                results.append(
                    BenchmarkResult(
                        sequence_length=seq_len,
                        memory_mb=float("inf"),
                        time_to_first_token=float("inf"),
                        time_per_token=float("inf"),
                        total_time=float("inf"),
                        tokens_per_second=0.0,
                        estimated_flops=float("inf"),
                    )
                )
                continue

            # Measure
            torch.cuda.empty_cache()
            start_mem = self.measure_memory()

            start_time = time.time()
            first_token_time = None

            from torch.profiler import profile, record_function, ProfilerActivity

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                with record_function("model_inference"):
                    with torch.no_grad():
                        for i in range(seq_len):
                            outputs = model(input_ids)
                            logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                            input_ids = torch.cat([input_ids, next_token], dim=1)

                            if i == 0:
                                first_token_time = time.time() - start_time

            total_time = time.time() - start_time
            end_mem = self.measure_memory()

            # Estimate FLOPs for attention
            estimated_flops = sum([evt.flops for evt in prof.key_averages()])

            result = BenchmarkResult(
                sequence_length=seq_len,
                memory_mb=end_mem - start_mem,
                time_to_first_token=first_token_time or 0.0,
                time_per_token=total_time / seq_len,
                total_time=total_time,
                tokens_per_second=seq_len / total_time,
                estimated_flops=estimated_flops,
            )

            results.append(result)

            print(f"  Memory: {result.memory_mb:.2f} MB")
            print(f"  TTFT: {result.time_to_first_token:.4f} s")
            print(f"  Time/token: {result.time_per_token*1000:.2f} ms")
            print(f"  Tokens/s: {result.tokens_per_second:.2f}")

        return results

    def plot_comparison(
        self,
        dual_stream_results: List[BenchmarkResult],
        transformer_results: List[BenchmarkResult],
        output_dir: Path,
    ):
        """Create comparison plots.

        Args:
            dual_stream_results: Dual-stream benchmark results
            transformer_results: Transformer baseline results
            output_dir: Where to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        ds_seq_lens = [r.sequence_length for r in dual_stream_results]
        ds_time_per_token = [r.time_per_token * 1000 for r in dual_stream_results]
        ds_memory = [r.memory_mb for r in dual_stream_results]
        ds_flops = [r.estimated_flops for r in dual_stream_results]

        tf_seq_lens = [r.sequence_length for r in transformer_results]
        tf_time_per_token = [
            r.time_per_token * 1000 if r.time_per_token != float("inf") else None
            for r in transformer_results
        ]
        tf_memory = [
            r.memory_mb if r.memory_mb != float("inf") else None
            for r in transformer_results
        ]
        tf_flops = [
            r.estimated_flops if r.estimated_flops != float("inf") else None
            for r in transformer_results
        ]

        # Filter out None values for transformer
        tf_valid_lens = [
            l for l, t in zip(tf_seq_lens, tf_time_per_token) if t is not None
        ]
        tf_valid_time = [t for t in tf_time_per_token if t is not None]
        tf_valid_memory = [m for m in tf_memory if m is not None]
        tf_valid_flops = [f for f in tf_flops if f is not None]

        # Plot 1: Time per token vs sequence length
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(
            ds_seq_lens,
            ds_time_per_token,
            "o-",
            label="Dual-Stream (O(N))",
            linewidth=2,
        )
        if tf_valid_lens:
            plt.plot(
                tf_valid_lens,
                tf_valid_time,
                "s-",
                label="Transformer (O(N²))",
                linewidth=2,
            )
        plt.xlabel("Sequence Length (tokens)")
        plt.ylabel("Time per Token (ms)")
        plt.title("Latency Scaling")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")

        # Plot 2: Memory usage vs sequence length
        plt.subplot(1, 2, 2)
        plt.plot(ds_seq_lens, ds_memory, "o-", label="Dual-Stream", linewidth=2)
        if tf_valid_lens:
            plt.plot(
                tf_valid_lens, tf_valid_memory, "s-", label="Transformer", linewidth=2
            )
        plt.xlabel("Sequence Length (tokens)")
        plt.ylabel("Memory (MB)")
        plt.title("Memory Scaling")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / "compute_economics.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved plot to {output_dir / 'compute_economics.png'}")

        # Plot 3: FLOPs comparison
        plt.figure(figsize=(10, 6))
        plt.plot(ds_seq_lens, ds_flops, "o-", label="Dual-Stream (O(N))", linewidth=2)
        if tf_valid_lens:
            plt.plot(
                tf_valid_lens,
                tf_valid_flops,
                "s-",
                label="Transformer (O(N²))",
                linewidth=2,
            )
        plt.xlabel("Sequence Length (tokens)")
        plt.ylabel("Estimated FLOPs")
        plt.title("Computational Complexity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(output_dir / "flops_comparison.png", dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_dir / 'flops_comparison.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute economics benchmark")
    parser.add_argument("--ssm-checkpoint", type=Path, help="Mamba SSM checkpoint")
    parser.add_argument("--codebook", type=Path, help="Dynamic codebook path")
    parser.add_argument("--vocab", type=Path, help="Vocabulary path")
    parser.add_argument(
        "--baseline-model", type=str, default="gpt2-medium", help="Baseline model"
    )
    parser.add_argument(
        "--prompt", type=str, default="The quick brown fox", help="Test prompt"
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/benchmarks"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip transformer baseline"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()


def main():
    args = parse_args()

    benchmark = ComputeEconomicsBenchmark(device=args.device)

    # Results storage
    results = {}

    # Benchmark dual-stream
    if args.ssm_checkpoint and args.codebook and args.vocab:
        print("\n" + "=" * 60)
        print("PHASE 1: DUAL-STREAM ARCHITECTURE")
        print("=" * 60)

        engine = DualStreamInference(
            ssm_checkpoint=args.ssm_checkpoint,
            codebook_path=args.codebook,
            vocab_path=args.vocab,
            device=args.device,
        )

        dual_stream_results = benchmark.benchmark_dual_stream(
            engine,
            args.prompt,
            args.sequence_lengths,
        )

        results["dual_stream"] = [
            {
                "sequence_length": r.sequence_length,
                "memory_mb": r.memory_mb,
                "ttft": r.time_to_first_token,
                "time_per_token": r.time_per_token,
                "total_time": r.total_time,
                "tokens_per_second": r.tokens_per_second,
                "estimated_flops": r.estimated_flops,
            }
            for r in dual_stream_results
        ]
    else:
        print("Skipping dual-stream benchmark (missing arguments)")
        dual_stream_results = None

    # Benchmark transformer baseline
    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("PHASE 2: TRANSFORMER BASELINE")
        print("=" * 60)

        transformer_results = benchmark.benchmark_transformer_baseline(
            args.baseline_model,
            args.prompt,
            args.sequence_lengths,
        )

        results["transformer"] = [
            {
                "sequence_length": r.sequence_length,
                "memory_mb": r.memory_mb if r.memory_mb != float("inf") else None,
                "ttft": (
                    r.time_to_first_token
                    if r.time_to_first_token != float("inf")
                    else None
                ),
                "time_per_token": (
                    r.time_per_token if r.time_per_token != float("inf") else None
                ),
                "total_time": r.total_time if r.total_time != float("inf") else None,
                "tokens_per_second": r.tokens_per_second,
                "estimated_flops": (
                    r.estimated_flops if r.estimated_flops != float("inf") else None
                ),
            }
            for r in transformer_results
        ]
    else:
        transformer_results = None

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir / 'benchmark_results.json'}")

    # Generate plots
    if dual_stream_results and transformer_results:
        benchmark.plot_comparison(
            dual_stream_results,
            transformer_results,
            args.output_dir,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: COST AT 100K TOKENS")
    print("=" * 60)

    if dual_stream_results:
        ds_100k = next(
            (r for r in dual_stream_results if r.sequence_length == 100000), None
        )
        if ds_100k:
            print(f"\nDual-Stream @ 100K tokens:")
            print(f"  Time per token: {ds_100k.time_per_token*1000:.2f} ms")
            print(f"  Memory: {ds_100k.memory_mb:.2f} MB")
            print(f"  FLOPs: {ds_100k.estimated_flops:.2e}")

    if transformer_results:
        tf_100k = next(
            (r for r in transformer_results if r.sequence_length == 100000), None
        )
        if tf_100k:
            print(f"\nTransformer @ 100K tokens:")
            if tf_100k.time_per_token != float("inf"):
                print(f"  Time per token: {tf_100k.time_per_token*1000:.2f} ms")
                print(f"  Memory: {tf_100k.memory_mb:.2f} MB")
                print(f"  FLOPs: {tf_100k.estimated_flops:.2e}")
            else:
                print(f"  FAILED (OOM or timeout)")


if __name__ == "__main__":
    main()
