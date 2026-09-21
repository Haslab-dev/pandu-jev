"""Representation Scaling Benchmark for Pandu (pandu-jev).

Evaluates whether environment sensory resolution (16d -> 32d -> 64d -> 128d)
breaks the ~86% policy accuracy ceiling under controlled policy capacity.
"""

import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from env.gridworld import GridWorld, Action
from models.policy import TinyPolicy
from datasets.trajectory import collect_expert_trajectories
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop

console = Console()

REPRESENTATION_CONFIGS = [
    {
        "name": "16d (Canonical Baseline)",
        "dim": 16,
        "hidden_dims": (32, 64),
        "desc": "Coordinates, distances, 4-wall sensors, 4 cardinal raycasts",
    },
    {
        "name": "32d (Medium Resolution)",
        "dim": 32,
        "hidden_dims": (48, 64),
        "desc": "16d + 4 diagonal rays, 4 dynamic hazard features, 3x3 local patch",
    },
    {
        "name": "64d (High Resolution)",
        "dim": 64,
        "hidden_dims": (64, 64),
        "desc": "32d + 5x5 occupancy ring, 8 goal projections, 8 corridor depths",
    },
    {
        "name": "128d (Dense Sensory / LIDAR)",
        "dim": 128,
        "hidden_dims": (96, 64),
        "desc": "64d + 7x7 occupancy ring, 16-ray circular LIDAR, 24 Fourier encodings",
    },
]


def run_representation_scaling_benchmark(
    num_episodes: int = 1500,
    epochs: int = 35,
    eval_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate policies across 16d, 32d, 64d, and 128d sensory representations."""
    console.print(Panel(
        "[bold cyan]Pandu Representation Scaling Benchmark[/bold cyan]\n"
        "[italic]Testing sensory resolution (16d -> 32d -> 64d -> 128d) as the true empirical bottleneck[/italic]",
        border_style="cyan"
    ))

    device = torch.device("cpu")
    results = []

    for cfg in REPRESENTATION_CONFIGS:
        dim = cfg["dim"]
        hidden_dims = cfg["hidden_dims"]
        name = cfg["name"]

        console.print(f"\n[bold yellow]► Evaluating {name} (dim={dim})...[/bold yellow]")

        # 1. Collect trajectories with matching feature dimension
        t0 = time.time()
        X, y, stats = collect_expert_trajectories(
            num_episodes=num_episodes,
            include_random_maps=True,
            feature_dim=dim,
            seed=seed,
        )
        collect_time = time.time() - t0

        # Split 80/20 train/val for standalone metric evaluation
        n_samples = len(y)
        indices = np.random.RandomState(seed).permutation(n_samples)
        split_idx = int(0.8 * n_samples)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 2. Build Policy with controlled capacity
        model = TinyPolicy(input_dim=dim, num_actions=4, hidden_dims=hidden_dims)
        params = model.count_parameters()

        # 3. Supervised Behavioral Cloning
        t0 = time.time()
        train_behavioral_cloning(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=64,
            lr=1e-3,
            device=device,
        )
        train_time = time.time() - t0

        # 4. Measure Holdout Validation Accuracy
        model.eval()
        with torch.no_grad():
            t_val_X = torch.tensor(X_val, dtype=torch.float32)
            logits = model(t_val_X)
            preds = logits.argmax(dim=-1).numpy()
            val_acc = float((preds == y_val).mean())

        # 5. Measure Latency
        dummy = torch.randn(1, dim)
        for _ in range(50):
            _ = model(dummy)
        latencies = []
        for _ in range(200):
            t_start = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t_start) * 1000.0)
        mean_latency = float(np.mean(latencies))

        # 6. Closed-loop evaluation on GridWorld
        closed_loop = evaluate_policy_closed_loop(
            model,
            num_episodes=eval_episodes,
            map_types="random",
            seed=seed + 999,
        )

        row = {
            "Representation": f"{dim}d",
            "Hidden Dims": f"{hidden_dims[0]}x{hidden_dims[1]}",
            "Params": f"{params:,}",
            "Val Acc": f"{val_acc * 100.0:.1f}%",
            "Closed-Loop Succ": f"{closed_loop['success_rate'] * 100.0:.1f}%",
            "Collisions/Ep": f"{closed_loop['wall_collisions_per_episode']:.2f}",
            "Avg Reward": f"{closed_loop['avg_reward']:.1f}",
            "Latency": f"{mean_latency:.3f} ms",
            "val_acc_raw": val_acc,
            "success_rate_raw": closed_loop["success_rate"],
            "params_raw": params,
            "latency_raw": mean_latency,
        }
        results.append(row)

        console.print(
            f"   [green]✔ Params:[/green] {params:,} | "
            f"[green]Val Acc:[/green] {val_acc * 100.0:.1f}% | "
            f"[green]Closed-Loop Succ:[/green] {closed_loop['success_rate'] * 100.0:.1f}% | "
            f"[green]Latency:[/green] {mean_latency:.3f} ms"
        )

    # Display Rich Summary Table
    table = Table(
        title="Environment Representation Scaling Results",
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Rep Dim", justify="center")
    table.add_column("Hidden Dims", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Val Accuracy", justify="right")
    table.add_column("Closed-Loop Succ", justify="right")
    table.add_column("Collisions/Ep", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Latency", justify="right")

    for r in results:
        table.add_row(
            r["Representation"],
            r["Hidden Dims"],
            r["Params"],
            r["Val Acc"],
            r["Closed-Loop Succ"],
            r["Collisions/Ep"],
            r["Avg Reward"],
            r["Latency"],
        )

    import json
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "representation_scaling.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    console.print(f"\n[green]✔ Results saved to {out_path}[/green]")

    return {"results": results}


if __name__ == "__main__":
    run_representation_scaling_benchmark(num_episodes=1000, epochs=30, eval_episodes=80)
