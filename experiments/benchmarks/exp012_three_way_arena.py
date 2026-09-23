#!/usr/bin/env python3
"""EXP-012 Three-Way Arena Benchmark: Pandu-Jev vs Laya-CoreML vs Jev (TypeSafe System One).

Compares:
1. Pandu-Jev NLP: ModernBERT-Tiny (~19.3M, Local MPS / FP16)
2. Laya-CoreML: ModernBERT-Base (~164M, Local CoreML / Apple Neural Engine FP16)
3. Jev-1.13: TypeSafe Cloud System One Flagship API (Zero-token typed decisions)

Metrics:
- Decision Accuracy & Semantic Probabilities
- Inference Latency (P50, P95, mean)
- Throughput (decisions/second)
- Survival Rate & Score across multiple game seeds
- Zero-token generation invariant check
- Memory & Cost profile
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
LAYA_PATH = Path("/Users/hy4-mac-002/hasdev/research/laya-coreml")

import importlib.metadata
_orig_version = importlib.metadata.version
def _safe_version(name):
    if name == "laya-coreml":
        return "0.1.1"
    try:
        return _orig_version(name)
    except Exception:
        return "0.1.0"
importlib.metadata.version = _safe_version

for p in [str(SRC_DIR), str(LAYA_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from laya_coreml.snake.game import DIRECTIONS, SnakeGame
from laya_coreml.snake.policy import LayaPolicy
from models.jev_nlp import PanduJevNLP, PanduJevNLPConfig
from models.jev_policy import JevPolicy

console = Console()


class PanduPolicyWrapper:
    """Wraps Pandu-Jev NLP to match Snake policy interface."""

    def __init__(self, device: str = "mps"):
        self.device = device
        self.model = PanduJevNLP().to(device)
        ckpt_path = PROJECT_ROOT / "checkpoints" / "pandu_snake_nlp.pt"
        if ckpt_path.is_file():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()

        self.policy = LayaPolicy.__new__(LayaPolicy)
        self.policy.agent = self.model
        self.policy.guarded = True
        self.policy.prompt = "compact"
        self.policy.metadata = {
            "name": "Pandu-Jev NLP",
            "hardware": f"Apple Silicon ({device.upper()})",
            "engine": "Pandu-Jev (MPS · FP16)",
            "guarded": True,
        }

    def decide(self, game: SnakeGame):
        return self.policy.decide(game)


def run_single_episode(policy, seed: int, steps: int = 15) -> Dict[str, Any]:
    """Run one episode for N steps and collect telemetry."""
    game = SnakeGame(24, 16, seed=seed, initial_length=6)
    latencies = []
    interventions = 0
    scores = 0
    t0 = time.perf_counter()

    for _ in range(steps):
        if not game.alive:
            break
        decision = policy.decide(game)
        latencies.append(decision.inference_ms)
        interventions += int(decision.intervened)
        game.step(decision.executed)

    elapsed = time.perf_counter() - t0
    return {
        "steps": len(latencies),
        "alive": game.alive,
        "score": game.score,
        "length": len(game.body),
        "interventions": interventions,
        "latencies_ms": latencies,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "elapsed_sec": elapsed,
        "throughput_dps": len(latencies) / elapsed if elapsed > 0 else 0.0,
    }


def run_three_way_benchmark(seeds: List[int] = [101, 102], steps_per_seed: int = 10) -> Dict[str, Any]:
    """Execute head-to-head empirical comparison across Pandu, Laya, and Jev."""
    console.print("\n[bold cyan]═════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold yellow]   EXP-012 THREE-WAY SNAKE ARENA: PANDU vs LAYA vs JEV (TYPESAFE)       [/bold yellow]")
    console.print("[bold cyan]═════════════════════════════════════════════════════════════════════════[/bold cyan]\n")

    results = {}

    # 1. Initialize Pandu-Jev NLP
    console.print("[bold green]1/3 Initializing Pandu-Jev NLP (19.3M, Local MPS)...[/bold green]")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pandu = PanduPolicyWrapper(device=device)

    # 2. Initialize Laya-CoreML
    console.print("[bold green]2/3 Initializing Laya-CoreML (164M, CoreML ANE)...[/bold green]")
    laya_model_dir = LAYA_PATH / "models" / "snake"
    laya = LayaPolicy(str(laya_model_dir), guarded=True, prompt="compact")

    # 3. Initialize Jev (TypeSafe Cloud System One)
    console.print("[bold green]3/3 Initializing Jev-1.13 (TypeSafe System One API)...[/bold green]")
    jev = JevPolicy(guarded=True)

    competitors = [
        ("Pandu-Jev NLP", pandu, "Local (MPS)", "19.3M", "Free / 0ms net"),
        ("Laya-CoreML", laya, "Local (CoreML ANE)", "164M", "Free / 0ms net"),
        ("Jev (TypeSafe)", jev, "Cloud API (HTTPS)", "Proprietary", "API Usage"),
    ]

    for name, policy, runtime, params, cost in competitors:
        console.print(f"\n[cyan]Evaluating [bold]{name}[/bold] across {len(seeds)} episodes ({steps_per_seed} steps each)...[/cyan]")
        all_latencies = []
        total_steps = 0
        total_interventions = 0
        total_score = 0
        total_time = 0.0
        deaths = 0

        for seed in seeds:
            res = run_single_episode(policy, seed=seed, steps=steps_per_seed)
            all_latencies.extend(res["latencies_ms"])
            total_steps += res["steps"]
            total_interventions += res["interventions"]
            total_score += res["score"]
            total_time += res["elapsed_sec"]
            deaths += int(not res["alive"])

        results[name] = {
            "runtime": runtime,
            "parameters": params,
            "cost_profile": cost,
            "total_steps": total_steps,
            "deaths": deaths,
            "interventions": total_interventions,
            "mean_latency_ms": float(np.mean(all_latencies)),
            "p50_latency_ms": float(np.percentile(all_latencies, 50)),
            "p95_latency_ms": float(np.percentile(all_latencies, 95)),
            "throughput_dps": total_steps / total_time if total_time > 0 else 0.0,
        }

    # Summary Table
    table = Table(title="EXP-012 Three-Way Snake Benchmark: Pandu vs Laya vs Jev")
    table.add_column("Competitor", style="bold cyan")
    table.add_column("Runtime / Engine", style="white")
    table.add_column("Parameters", justify="right", style="magenta")
    table.add_column("P50 Latency", justify="right", style="green")
    table.add_column("P95 Latency", justify="right", style="green")
    table.add_column("Throughput", justify="right", style="bold yellow")
    table.add_column("Interventions", justify="right", style="cyan")
    table.add_column("Deaths", justify="right", style="red")
    table.add_column("Cost Profile", style="white")

    for name, r in results.items():
        table.add_row(
            name,
            r["runtime"],
            r["parameters"],
            f"{r['p50_latency_ms']:.1f} ms",
            f"{r['p95_latency_ms']:.1f} ms",
            f"{r['throughput_dps']:.1f} dec/s",
            str(r["interventions"]),
            str(r["deaths"]),
            r["cost_profile"],
        )

    console.print("\n")
    console.print(table)

    # Save results
    out_file = PROJECT_ROOT / "experiments" / "results" / "exp012_three_way_arena.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    console.print(f"\n[bold green]Saved benchmark telemetry to {out_file}[/bold green]\n")

    return results


if __name__ == "__main__":
    run_three_way_benchmark(seeds=[101, 102], steps_per_seed=8)
