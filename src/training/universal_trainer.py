"""Universal Multi-Domain TypeSafe System One Trainer for Pandu.

Trains Pandu on diverse software engineering, routing, tool dispatching,
and triage tasks using ModernBERT (both 19.3M Tiny and 149M Base).
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.optim import AdamW

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.universal_dataset import generate_universal_training_samples
from models.jev_nlp import (
    PanduJevNLP,
    PanduJevNLPConfig,
    build_sequence,
    render_options,
)

console = Console()


def prepare_training_batches(
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int = 16,
    max_len: int = 256,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """Tokenize and batch multi-task System One training samples."""
    items = []

    for s in samples:
        state = s["state"]
        questions = s["questions"]
        targets = s["targets"]

        for qid, qdef in questions.items():
            qtype_str = qdef.get("type", "choice")
            qtype_int = 0 if qtype_str == "choice" else (1 if qtype_str == "score" else 2)

            ids, markers = build_sequence(tokenizer, state, qdef, max_len=max_len)
            k = len(markers)
            if k < 2:
                continue

            # Target probabilities formulation
            target_val = targets.get(qid)
            if qtype_str == "choice":
                opts = list(qdef["criteria"].keys())
                probs = [float(target_val.get(opt, 1.0 / k)) for opt in opts[:k]]
                norm = sum(probs) or 1.0
                target_dist = [p / norm for p in probs]
            elif qtype_str == "score":
                # Distribution centered around target score
                target_idx = int(target_val)
                target_dist = [0.05 / max(1, k - 1)] * k
                if 0 <= target_idx < k:
                    target_dist[target_idx] = 0.95
            else:  # noul
                p_yes = float(target_val)
                target_dist = [1.0 - p_yes, p_yes]

            items.append({
                "ids": ids,
                "markers": markers,
                "k": k,
                "qtype": qtype_int,
                "target_dist": target_dist,
            })

    # Group into batches
    batches = []
    pad_id = tokenizer.pad_token_id or 50283

    for b_start in range(0, len(items), batch_size):
        b_items = items[b_start : b_start + batch_size]
        B = len(b_items)
        max_seq = max(len(it["ids"]) for it in b_items)
        max_k = max(it["k"] for it in b_items)

        input_ids = torch.full((B, max_seq), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, max_seq), dtype=torch.float, device=device)
        marker_pos = torch.zeros((B, max_k), dtype=torch.long, device=device)
        marker_mask = torch.zeros((B, max_k), dtype=torch.float, device=device)
        qtypes = torch.zeros(B, dtype=torch.long, device=device)
        target_probs = torch.zeros((B, max_k), dtype=torch.float, device=device)

        for i, it in enumerate(b_items):
            s_len = len(it["ids"])
            k_len = it["k"]
            input_ids[i, :s_len] = torch.tensor(it["ids"], dtype=torch.long, device=device)
            attn_mask[i, :s_len] = 1.0
            marker_pos[i, :k_len] = torch.tensor(it["markers"], dtype=torch.long, device=device)
            marker_mask[i, :k_len] = 1.0
            qtypes[i] = it["qtype"]
            target_probs[i, :k_len] = torch.tensor(it["target_dist"][:k_len], dtype=torch.float, device=device)

        batches.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtypes,
            "target_probs": target_probs,
        })

    return batches


def train_universal_pandu(
    use_base: bool = False,
    epochs: int = 15,
    lr: float = 2e-4,
    batch_size: int = 16,
    multiplier: int = 6,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train Pandu-Jev on multi-domain universal System One tasks."""
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tier_name = "ModernBERT-Base (149M)" if use_base else "ModernBERT-Tiny (19.3M)"

    console.print(Panel.fit(
        f"[bold cyan]Pandu Universal Multi-Domain System One Training[/bold cyan]\n"
        f"Backbone: [bold yellow]{tier_name}[/bold yellow]\n"
        f"Target Device: [bold green]{device.upper()}[/bold green]\n"
        f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}",
        border_style="cyan"
    ))

    # 1. Synthesize Data
    console.print(f"[cyan]Generating universal multi-domain training samples (multiplier={multiplier})...[/cyan]")
    samples = generate_universal_training_samples(multiplier=multiplier, seed=42)
    console.print(f"[green]Synthesized {len(samples)} diverse multi-task scenarios.[/green]")

    # 2. Instantiate Model
    config = PanduJevNLPConfig(use_pretrained_base=use_base)
    model = PanduJevNLP(config).to(device)
    model.train()

    # If base model, freeze earlier transformer layers for fast fine-tuning
    if use_base:
        # Fine-tune upper 4 layers + heads
        for name, param in model.encoder.named_parameters():
            if "layers" in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num < 18:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total Parameters: [bold magenta]{model.num_parameters:,}[/bold magenta] ({model.memory_footprint_mb:.1f} MB)")
    console.print(f"Trainable Parameters: [bold green]{trainable_params:,}[/bold green]")

    # 3. Prepare Vectorized Batches
    console.print("[cyan]Pre-collating tokenized batches for zero-overhead training...[/cyan]")
    batches = prepare_training_batches(samples, model.tokenizer, batch_size=batch_size, device=device)
    console.print(f"[green]Created {len(batches)} vectorized batches.[/green]\n")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)

    # 4. Training Loop
    start_time = time.perf_counter()
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct_top1 = 0
        total_items = 0
        t0 = time.perf_counter()

        for b in batches:
            optimizer.zero_grad()
            logits, act_logits = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                marker_pos=b["marker_pos"],
                marker_mask=b["marker_mask"],
                qtype=b["qtype"],
            )

            # KL-Divergence / Cross-Entropy on active options
            log_p = F.log_softmax(logits, dim=-1)
            target_p = b["target_probs"]
            kl_loss = F.kl_div(log_p, target_p, reduction="batchmean")

            # Auxiliary action confidence loss
            act_target = torch.zeros(b["input_ids"].size(0), dtype=torch.long, device=device)
            act_loss = F.cross_entropy(act_logits, act_target)

            loss = kl_loss + 0.1 * act_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())

            # Accuracy calculation
            preds = logits.argmax(dim=-1)
            targets = target_p.argmax(dim=-1)
            correct_top1 += int((preds == targets).sum().item())
            total_items += b["input_ids"].size(0)

        epoch_time = time.perf_counter() - t0
        avg_loss = epoch_loss / len(batches)
        acc = (correct_top1 / total_items) * 100.0 if total_items else 0.0
        losses.append(avg_loss)

        if epoch % 3 == 0 or epoch == epochs or epoch == 1:
            console.print(
                f"Epoch {epoch:02d}/{epochs:02d} | Loss: [bold yellow]{avg_loss:.4f}[/bold yellow] | "
                f"Accuracy: [bold green]{acc:.1f}%[/bold green] | Speed: {epoch_time:.2f}s"
            )

    total_duration = time.perf_counter() - start_time

    # 5. Save Checkpoint
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_name:
        checkpoint_name = "pandu_universal_base.pt" if use_base else "pandu_universal_tiny.pt"
    ckpt_path = ckpt_dir / checkpoint_name
    torch.save(model.state_dict(), ckpt_path)

    console.print(f"\n[bold green]✓ Training Complete! Saved checkpoint to {ckpt_path.name}[/bold green]")
    console.print(f"Total Duration: {total_duration:.2f}s | Final Holdout Accuracy: {acc:.1f}%\n")

    return {
        "model_name": tier_name,
        "parameters": model.num_parameters,
        "memory_mb": model.memory_footprint_mb,
        "epochs": epochs,
        "final_loss": avg_loss,
        "final_accuracy": acc,
        "duration_sec": total_duration,
        "checkpoint": str(ckpt_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Pandu on Universal Multi-Domain System One tasks.")
    parser.add_argument("--base", action="store_true", help="Train 149M ModernBERT-Base")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_universal_pandu(
        use_base=args.base,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
