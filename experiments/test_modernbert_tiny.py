"""Test 1: ModernBERT-Tiny Language-Conditioned Policy via Hugging Face.

Architecture: ModernBERT-Tiny (answerdotai/ModernBERT-base architecture, ~22M params)
Task: Natural language instruction + environment state -> Action distribution & confidence.

Usage:
    python experiments/test_modernbert_tiny.py
    python experiments/test_modernbert_tiny.py --download-weights
"""

import sys
import time
import math
from typing import Dict, Any, List
import click
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification, AutoModelForSequenceClassification

console = Console()

MODEL_ID = "answerdotai/ModernBERT-base"
ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT"]


def format_state_prompt(obs: Dict[str, Any], instruction: str) -> str:
    return (
        f"Instruction: {instruction} "
        f"Position: ({obs['agent_x']}, {obs['agent_y']}). "
        f"Goal: ({obs['goal_x']}, {obs['goal_y']}). "
        f"Obstacles: {obs['walls_near']}. "
        f"Decide action (UP, DOWN, LEFT, RIGHT):"
    )


@click.command()
@click.option("--download-weights", is_flag=True, help="Download full pretrained weights from HuggingFace Hub (~600MB).")
def main(download_weights: bool):
    console.print(Panel.fit(
        f"[bold cyan]Test 1: ModernBERT-Tiny from Hugging Face[/bold cyan]\n"
        f"Model Base: [yellow]{MODEL_ID}[/yellow]\n"
        f"Role: High-Speed Text & State Classifier for Language Policies"
    ))

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Hardware Compute Device: [magenta]{device}[/magenta]")

    # Load Tokenizer from Hugging Face Hub (cached)
    console.print(f"Loading Hugging Face Tokenizer: [cyan]{MODEL_ID}[/cyan]...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if download_weights:
        console.print(f"Downloading full weights for [yellow]{MODEL_ID}[/yellow] from HuggingFace Hub...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=4,
            ignore_mismatched_sizes=True,
        )
    else:
        console.print(f"Instantiating ModernBERT-Tiny architecture (~22M parameters)...")
        config = ModernBertConfig(
            vocab_size=len(tokenizer),
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_labels=4,
            classifier_dropout=0.1,
        )
        model = ModernBertForSequenceClassification(config)

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    console.print(f"Total Parameters: [bold green]{n_params:,}[/bold green] (~{n_params/1e6:.1f}M params)")
    console.print(f"Memory Footprint: [bold green]{mem_mb:.2f} MB[/bold green]\n")

    # Scenarios
    test_cases = [
        {
            "desc": "Direct Path East",
            "instruction": "Navigate to the eastern goal quickly.",
            "obs": {"agent_x": 1, "agent_y": 3, "goal_x": 10, "goal_y": 3, "walls_near": "UP, DOWN"},
        },
        {
            "desc": "Vertical Descent",
            "instruction": "Avoid obstacle wall on the right and descend to bottom exit.",
            "obs": {"agent_x": 4, "agent_y": 1, "goal_x": 4, "goal_y": 6, "walls_near": "RIGHT, UP"},
        },
        {
            "desc": "Ambiguous Hazard",
            "instruction": "Navigate cautiously around sudden dynamic hazard.",
            "obs": {"agent_x": 5, "agent_y": 4, "goal_x": 1, "goal_y": 1, "walls_near": "UP, LEFT, RIGHT"},
        },
    ]

    # Warmup
    dummy = tokenizer("warmup query", return_tensors="pt")
    dummy = {k: v.to(device) for k, v in dummy.items()}
    with torch.no_grad():
        _ = model(**dummy)

    table = Table(title="ModernBERT-Tiny Decision Benchmark")
    table.add_column("Scenario", style="bold")
    table.add_column("Instruction", style="dim")
    table.add_column("Predicted Action", style="cyan")
    table.add_column("Probabilities (UP / DOWN / LEFT / RIGHT)", style="magenta")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Latency", justify="right", style="yellow")

    latencies = []

    for case in test_cases:
        prompt = format_state_prompt(case["obs"], case["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        t_end = time.perf_counter()

        lat_ms = (t_end - t_start) * 1000.0
        latencies.append(lat_ms)

        pred_idx = int(probs.argmax())
        pred_action = ACTION_LABELS[pred_idx]
        conf = float(probs[pred_idx])

        prob_str = " | ".join(f"{ACTION_LABELS[i]}:{probs[i]*100:.1f}%" for i in range(4))

        table.add_row(
            case["desc"],
            case["instruction"][:32] + "...",
            pred_action,
            prob_str,
            f"{conf*100:.1f}%",
            f"{lat_ms:.2f} ms",
        )

    console.print(table)

    avg_lat = sum(latencies) / len(latencies)
    throughput = 1000.0 / max(0.001, avg_lat)

    console.print(Panel.fit(
        f"[bold green]ModernBERT-Tiny Benchmark Summary[/bold green]\n"
        f"• Steady-State Decision Latency: [bold]{avg_lat:.2f} ms[/bold]\n"
        f"• Throughput: [bold]~{throughput:.0f} decisions/sec[/bold]\n"
        f"• Parameter Budget: [bold]{n_params:,}[/bold] ({mem_mb:.1f} MB)\n"
        f"• Suitability: Super-fast text state encoder for language-conditioned reflexes."
    ))


if __name__ == "__main__":
    main()
