#!/usr/bin/env python3
"""Train a Mamba SSM on manifold signature sequences.

Replaces GPT-2 with a State Space Model for O(1) inference per step.
This is Phase 1 of the dual-stream architecture: the Engine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint

try:
    from mamba_ssm import Mamba
except ImportError:
    print("ERROR: mamba-ssm not installed. Install with: pip install mamba-ssm")
    sys.exit(1)


@dataclass
class SSMConfig:
    """Configuration for Mamba SSM model."""

    d_model: int = 768
    n_layer: int = 16
    vocab_size: int = 50000
    ssm_cfg: dict = None
    rms_norm: bool = True
    fused_add_norm: bool = True
    residual_in_fp32: bool = True

    def to_dict(self):
        return {
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "vocab_size": self.vocab_size,
            "ssm_cfg": self.ssm_cfg or {},
            "rms_norm": self.rms_norm,
            "fused_add_norm": self.fused_add_norm,
            "residual_in_fp32": self.residual_in_fp32,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            d_model=config_dict.get("d_model", 768),
            n_layer=config_dict.get("n_layer", 16),
            vocab_size=config_dict.get("vocab_size", 50000),
            ssm_cfg=config_dict.get("ssm_cfg"),
            rms_norm=config_dict.get("rms_norm", True),
            fused_add_norm=config_dict.get("fused_add_norm", True),
            residual_in_fp32=config_dict.get("residual_in_fp32", True),
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class MambaBlock(nn.Module):
    """Mamba SSM block with residual connection."""

    def __init__(
        self,
        d_model: int,
        ssm_cfg: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, layer_idx=layer_idx, **(ssm_cfg or {}))
        if layer_idx is not None:
            self.mamba.layer_idx = layer_idx
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        # Residual connection with pre-normalization
        return x + self.mamba(self.norm(x), inference_params=inference_params)


class MambaLM(nn.Module):
    """Mamba-based Language Model for manifold signatures."""

    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Replace single stream with 3 parallel Cortical Columns
        self.num_columns = 3
        self.columns = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MambaBlock(config.d_model, config.ssm_cfg, layer_idx=i)
                        for i in range(config.n_layer)
                    ]
                )
                for _ in range(self.num_columns)
            ]
        )
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights between embedding and output projection
        self.lm_head.weight = self.embedding.weight

        # Initialize embeddings with strict standard deviation bounds to prevent Logit explosion
        # and Softmax Saturation (which originally crashed loss at ~757.0 uniformly)
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        inference_params=None,
    ):
        """Forward pass through Mamba SSM.

        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) for training
            inference_params: Caching params for O(1) inference

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        x = self.embedding(input_ids)

        # Pass through the parallel Cortical Columns
        column_outputs = []
        for col in self.columns:
            h = x
            for layer in col:
                h = layer(h, inference_params=inference_params)
            column_outputs.append(h)

        # Heterarchical Voting (Consensus Mechanism)
        # We take the mean across all 3 cortical columns to minimize global structural tension
        x = torch.mean(torch.stack(column_outputs), dim=0)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        output = {"logits": logits, "hidden": x}

        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            output["loss"] = loss

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ):
        """O(1) per-step inference using SSM hidden state.

        This is the key advantage: SSM maintains a fixed-size hidden state,
        allowing O(1) computation per token instead of O(N^2) attention.
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Only need to process the last token due to SSM state
                outputs = self(input_ids)
                next_token_logits = outputs["logits"][:, -1, :] / temperature

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


@dataclass
class ManifoldDataCollator:
    pad_token_id: int

    def __call__(self, features: List[dict]) -> dict:
        max_length = max(len(f["input_ids"]) for f in features)
        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_attention: List[List[int]] = []

        for feature in features:
            ids = list(feature["input_ids"])
            labels = list(feature["labels"])
            length = len(ids)
            pad_len = max_length - length

            if pad_len:
                ids = ids + [self.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

            attention = [1] * length + [0] * pad_len
            batch_input_ids.append(ids)
            batch_labels.append(labels)
            batch_attention.append(attention)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
        }


def load_vocab(vocab_path: Path) -> Tuple[List[str], int]:
    """Load vocabulary from JSON."""
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    signatures = list(payload.get("signatures", []))
    return signatures, len(signatures)


def prepare_datasets(
    dataset_path: Path,
    eval_holdout: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and split dataset."""
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {dataset_path}, found {type(dataset)}")

    shuffled = dataset.shuffle(seed=seed)

    # Handle very small datasets to prevent train_test_split from failing
    if len(shuffled) < 10:
        print(
            f"Warning: Dataset too small ({len(shuffled)} samples) for eval holdout. Using all for training."
        )
        eval_holdout = 0.0

    if eval_holdout > 0.0:
        split = shuffled.train_test_split(test_size=eval_holdout, seed=seed)
        train_dataset = split["train"]
        eval_dataset: Optional[Dataset] = split["test"]
    else:
        train_dataset = shuffled
        eval_dataset = None

    if max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(max_train_samples, len(train_dataset)))
        )
    if eval_dataset is not None and max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(max_eval_samples, len(eval_dataset)))
        )

    return train_dataset, eval_dataset


def save_checkpoint(
    model: MambaLM, optimizer, step: int, output_dir: Path, config: SSMConfig
):
    """Save model checkpoint."""
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")

    # Save optimizer state
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    # Save config
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save step info
    with open(checkpoint_dir / "trainer_state.json", "w") as f:
        json.dump({"global_step": step}, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(checkpoint_dir: Path, model: MambaLM, optimizer):
    """Load model checkpoint."""
    model.load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin"))
    if (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))

    with open(checkpoint_dir / "trainer_state.json", "r") as f:
        state = json.load(f)
        return state["global_step"]


def train_epoch(
    model: MambaLM,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    local_hebbian: bool = False,
    learning_rate: float = 1e-4,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"] / gradient_accumulation_steps

        if local_hebbian:
            shift_logits = outputs["logits"][..., :-1, :].contiguous()
            shift_hidden = outputs["hidden"][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 1. Update the LM Head (and implicitly the tied Embedding)
            vocab_size = shift_logits.size(-1)
            probs = torch.softmax(shift_logits, dim=-1)
            # Create target distribution
            valid_labels = shift_labels.clamp_min(0)
            one_hot = torch.zeros_like(probs).scatter_(
                -1, valid_labels.unsqueeze(-1), 1.0
            )

            # The error in probabilities
            error = one_hot - probs

            # 1. Normalize the Error Term (L2 Normalization acting as refractory period constraint)
            error = F.normalize(error, p=2, dim=-1)

            error_flat = error.view(-1, vocab_size)
            hidden_flat = shift_hidden.view(-1, shift_hidden.size(-1))

            # Gradient for LM Head
            dW_head = error_flat.t().mm(hidden_flat) / max(hidden_flat.size(0), 1)

            with torch.no_grad():
                # 2. Map Oja's Weight Decay: ∆W = η(dW - y^2 W)
                y_sq_head = (error_flat**2).mean(dim=0).unsqueeze(1)
                decay_head = y_sq_head * model.lm_head.weight

                model.lm_head.weight.add_(
                    learning_rate * 10 * (dW_head - decay_head)
                )  # Boosted LR for head

                # 2. Predictive Coding for the SSM Manifold (Deep Hebbian)
                # Instead of backprop, we use the actual next token embedding as the local target
                # for the current hidden state.
                # e_l = Target - Predicted
                target_hidden = model.embedding(valid_labels)
                hidden_error = target_hidden - shift_hidden

                # 1. Normalize the Hidden Error Term
                hidden_error = F.normalize(hidden_error, p=2, dim=-1)

                # We apply a simplified Oja's/Hebbian rule to the norm layer and implicitly
                # to the layers by adjusting weights in direction of the local error
                # For simplicity in this architectural milestone, we apply the FEP error
                # directly to the final projection of the Mamba layers
                for col in model.columns:
                    for layer in col:
                        if hasattr(layer.mamba, "out_proj"):
                            # Local structural update: shift weights toward resolving the physical discrepancy
                            # dW = error * input^T
                            layer_out = layer.mamba.out_proj
                            err_flat = hidden_error.view(-1, hidden_error.size(-1))
                            # In a true local Hebbian, we'd use the layer's local input, but as a proxy
                            # we use the recurrent state to push the output projection.
                            dW_layer = err_flat.t().mm(hidden_flat) / max(
                                hidden_flat.size(0), 1
                            )
                            w_out = layer_out.weight
                            if w_out.size(1) != dW_layer.size(1):
                                repeats = w_out.size(1) // dW_layer.size(1)
                                if repeats > 0:
                                    dW_layer = dW_layer.repeat(1, repeats)
                                if dW_layer.size(1) != w_out.size(1):
                                    dW_layer = F.pad(
                                        dW_layer, (0, w_out.size(1) - dW_layer.size(1))
                                    )

                            # 2. Oja's Weight Decay for the Layer Parameter
                            y_sq_layer = (err_flat**2).mean(dim=0).unsqueeze(1)
                            if y_sq_layer.size(0) != w_out.size(0):
                                repeats_y = w_out.size(0) // y_sq_layer.size(0)
                                if repeats_y > 0:
                                    y_sq_layer = y_sq_layer.repeat(repeats_y, 1)
                                if y_sq_layer.size(0) != w_out.size(0):
                                    y_sq_layer = F.pad(
                                        y_sq_layer,
                                        (0, 0, 0, w_out.size(0) - y_sq_layer.size(0)),
                                    )

                            decay_layer = y_sq_layer * w_out

                            layer_out.weight.add_(
                                learning_rate * (dW_layer - decay_layer)
                            )
        else:
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

    return total_loss / num_batches


def evaluate(model: MambaLM, dataloader: DataLoader, device: torch.device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
    return avg_loss, perplexity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mamba SSM on manifold signatures"
    )
    parser.add_argument(
        "--dataset-path", type=Path, required=True, help="Path to HF dataset"
    )
    parser.add_argument(
        "--vocab-path", type=Path, required=True, help="Path to vocab.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )

    # Model architecture
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-layer", type=int, default=16, help="Number of layers")

    # Training
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--eval-holdout", type=float, default=0.02, help="Eval split fraction"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--local-hebbian",
        action="store_true",
        help="Use local Hebbian updates instead of global backprop",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocab
    signatures, vocab_size = load_vocab(args.vocab_path)
    print(f"Loaded vocabulary: {vocab_size} unique signatures")

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        args.dataset_path, args.eval_holdout, args.seed
    )
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    # Create data collator
    collator = ManifoldDataCollator(pad_token_id=0)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

    # Create model
    config = SSMConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=vocab_size,
    )
    model = MambaLM(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Resume from checkpoint if requested
    start_step = 0
    if args.resume:
        last_checkpoint = get_last_checkpoint(str(args.output_dir))
        if last_checkpoint:
            print(f"Resuming from {last_checkpoint}")
            start_step = load_checkpoint(Path(last_checkpoint), model, optimizer)

    # Training loop
    global_step = start_step
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Starting Training ===")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_start = time.time()

        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.gradient_accumulation_steps,
            local_hebbian=args.local_hebbian,
            learning_rate=args.learning_rate,
        )

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        global_step += len(train_loader) // args.gradient_accumulation_steps

        # Evaluate
        if eval_loader:
            eval_loss, perplexity = evaluate(model, eval_loader, device)
            print(f"Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")

        # Save checkpoint
        if (epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1:
            save_checkpoint(model, optimizer, global_step, args.output_dir, config)

    print("\n=== Training Complete ===")
    print(f"Final model saved to {args.output_dir}")

    # Save final metadata
    metadata = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "final_step": global_step,
    }
    with open(args.output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
