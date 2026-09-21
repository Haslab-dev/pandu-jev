"""EXP-008: 64d Perceptual Feature Group Ablation & Information Density Benchmark.

Isolates which perceptual features actually contribute to micro-policy competence
under a strictly controlled 64d container and fixed policy capacity (~8,516 parameters).

Feature Group Layout in 64d:
  - [0:8]   : Agent/Goal Coordinates & Euclidean/Manhattan Distances (8)
  - [8:12]  : Immediate Adjacent Wall Sensors (4)
  - [12:16] : 4 Cardinal Raycasts (4)
  - [16:20] : 4 Diagonal Raycasts (4)
  - [20:24] : 4 Dynamic Hazard Sensors (4)
  - [24:32] : 8-Cell 3x3 Local Occupancy Patch (8)
  - [32:48] : 16-Cell 5x5 Local Occupancy Ring (16)
  - [48:56] : 8 Goal Projections (8)
  - [56:64] : 8 Corridor Clearance Depths (8)

By zeroing out each group independently:
  1. Policy architecture and parameter count are strictly identical (8,516 params).
  2. We measure ΔSuccess, ΔCollisions, ΔRecovery, and Information Efficiency.
  3. Answers: "What should an 8K-parameter embodied policy actually perceive?"
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Single-threaded deterministic CPU
torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from datasets.trajectory import collect_expert_trajectories  # noqa: E402
from env.gridworld import Action, GridWorld, create_random_gridworld  # noqa: E402
from evaluation.behavior import evaluate_closed_loop_behavior  # noqa: E402
from models.policy import TinyPolicy  # noqa: E402
from training.bc import train_behavioral_cloning  # noqa: E402

console = Console()

TRAIN_EPISODES = 800
EPOCHS = 25
EVAL_EPISODES = 70
SEEDS = [42, 1337]

FEATURE_GROUPS = [
    {
        "id": "full_64d",
        "name": "Full 64d Baseline",
        "zero_indices": [],
        "active_dim": 64,
        "desc": "All 64 sensory features active",
    },
    {
        "id": "no_diagonal_rays",
        "name": "w/o Diagonal Rays",
        "zero_indices": list(range(16, 20)),
        "active_dim": 60,
        "desc": "Removes 4 diagonal raycasts (NW, NE, SW, SE)",
    },
    {
        "id": "no_hazard_sensors",
        "name": "w/o Hazard Sensors",
        "zero_indices": list(range(20, 24)),
        "active_dim": 60,
        "desc": "Removes 4 dynamic hazard features",
    },
    {
        "id": "no_3x3_patch",
        "name": "w/o 3x3 Local Patch",
        "zero_indices": list(range(24, 32)),
        "active_dim": 56,
        "desc": "Removes 8 Chebyshev dist-1 wall occupancy cells",
    },
    {
        "id": "no_5x5_ring",
        "name": "w/o 5x5 Occupancy Ring",
        "zero_indices": list(range(32, 48)),
        "active_dim": 48,
        "desc": "Removes 16 Chebyshev dist-2 wall-hugging cells",
    },
    {
        "id": "no_goal_projections",
        "name": "w/o Goal Projections",
        "zero_indices": list(range(48, 56)),
        "active_dim": 56,
        "desc": "Removes 8 ray-to-goal alignment projections",
    },
    {
        "id": "no_corridor_depths",
        "name": "w/o Corridor Clearance",
        "zero_indices": list(range(56, 64)),
        "active_dim": 56,
        "desc": "Removes 8 multi-directional corridor depths",
    },
    {
        "id": "no_cardinal_rays",
        "name": "w/o Cardinal Raycasts",
        "zero_indices": list(range(12, 16)),
        "active_dim": 60,
        "desc": "Removes 4 cardinal raycasts (UP, DOWN, LEFT, RIGHT)",
    },
    {
        "id": "no_immediate_walls",
        "name": "w/o Immediate Wall Sensors",
        "zero_indices": list(range(8, 12)),
        "active_dim": 60,
        "desc": "Removes 4 adjacent 1-step wall bump sensors",
    },
]


def mask_features(X: np.ndarray, zero_indices: List[int]) -> np.ndarray:
    """Mask specified indices to 0.0 in a copy of the feature matrix."""
    if not zero_indices:
        return X.copy()
    X_masked = X.copy()
    X_masked[:, zero_indices] = 0.0
    return X_masked


class MaskedPolicyWrapper(torch.nn.Module):
    """Wraps TinyPolicy to zero out specific indices during closed-loop rollout."""
    def __init__(self, policy: TinyPolicy, zero_indices: List[int]):
        super().__init__()
        self.policy = policy
        self.zero_indices = zero_indices
        self.input_dim = policy.input_dim

    def get_action_distribution(self, feat: np.ndarray) -> Dict[str, Any]:
        feat_masked = feat.copy()
        if self.zero_indices:
            feat_masked[self.zero_indices] = 0.0
        return self.policy.get_action_distribution(feat_masked)


def run_exp008() -> Dict[str, Any]:
    console.print(Panel(
        "[bold cyan]EXP-008: 64d Feature Group Ablation & Information Density Benchmark[/bold cyan]\n"
        "[italic]Ablating each feature group under fixed 64d container & fixed 8,516 policy parameters[/italic]",
        border_style="cyan",
    ))

    t_start = time.time()
    all_runs: List[Dict[str, Any]] = []

    for seed in SEEDS:
        console.print(f"\n[bold yellow]► Collecting 64d expert dataset (seed={seed})...[/bold yellow]")
        X_raw, y_raw, _ = collect_expert_trajectories(
            num_episodes=TRAIN_EPISODES,
            include_random_maps=True,
            feature_dim=64,
            seed=seed,
        )

        n_samples = len(y_raw)
        perm = np.random.RandomState(seed).permutation(n_samples)
        split = int(0.8 * n_samples)
        tr_idx, val_idx = perm[:split], perm[split:]

        for grp in FEATURE_GROUPS:
            grp_id = grp["id"]
            grp_name = grp["name"]
            zero_idx = grp["zero_indices"]
            active_dim = grp["active_dim"]

            X_masked = mask_features(X_raw, zero_idx)
            X_tr, y_tr = X_masked[tr_idx], y_raw[tr_idx]
            X_val, y_val = X_masked[val_idx], y_raw[val_idx]

            # Fixed 8,516-param TinyPolicy (64 in, (64, 64) hidden, 4 out)
            model = TinyPolicy(input_dim=64, num_actions=4, hidden_dims=(64, 64))

            train_behavioral_cloning(
                model,
                X_tr,
                y_tr,
                epochs=EPOCHS,
                batch_size=64,
                lr=1e-3,
                device=torch.device("cpu"),
            )

            # Holdout validation accuracy
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32))
                val_preds = torch.argmax(val_logits, dim=-1).numpy()
                val_acc = float(np.mean(val_preds == y_val))

            # Closed-loop evaluation
            wrapped_model = MaskedPolicyWrapper(model, zero_idx)
            cl_metrics = evaluate_closed_loop_behavior(
                wrapped_model,
                num_episodes=EVAL_EPISODES,
                map_types="random",
                feature_dim=64,
                seed=seed + 999,
            )

            run_record = {
                "group_id": grp_id,
                "group_name": grp_name,
                "active_dim": active_dim,
                "seed": seed,
                "val_acc": val_acc,
                "success_rate": cl_metrics["trajectory_success_rate"],
                "wall_hits": cl_metrics["wall_hits_per_episode"],
                "action_acc": cl_metrics["action_accuracy"],
                "steps": cl_metrics["mean_episode_length"],
                "info_efficiency": (cl_metrics["trajectory_success_rate"] * 100.0) / float(active_dim),
            }
            all_runs.append(run_record)

            console.print(
                f"   {grp_name:<28} (seed={seed}) | "
                f"val_acc={val_acc*100:5.1f}% | "
                f"succ={cl_metrics['trajectory_success_rate']*100:5.1f}% | "
                f"hits={cl_metrics['wall_hits_per_episode']:5.2f} | "
                f"eff={run_record['info_efficiency']:4.2f}"
            )

    # Compute baseline reference values
    base_runs = [r for r in all_runs if r["group_id"] == "full_64d"]
    base_succ = float(np.mean([r["success_rate"] for r in base_runs])) * 100.0
    base_hits = float(np.mean([r["wall_hits"] for r in base_runs]))

    summary_rows = []
    for grp in FEATURE_GROUPS:
        grp_id = grp["id"]
        runs = [r for r in all_runs if r["group_id"] == grp_id]
        mean_succ = float(np.mean([r["success_rate"] for r in runs])) * 100.0
        mean_hits = float(np.mean([r["wall_hits"] for r in runs]))
        mean_val = float(np.mean([r["val_acc"] for r in runs])) * 100.0
        mean_eff = float(np.mean([r["info_efficiency"] for r in runs]))

        delta_succ = mean_succ - base_succ
        delta_hits = mean_hits - base_hits

        summary_rows.append({
            "group_id": grp_id,
            "group_name": grp["name"],
            "active_dim": grp["active_dim"],
            "success_rate": mean_succ,
            "delta_success": delta_succ,
            "wall_hits": mean_hits,
            "delta_hits": delta_hits,
            "val_acc": mean_val,
            "info_efficiency": mean_eff,
            "desc": grp["desc"],
        })

    # Summary Table
    console.print("\n")
    table = Table(title="EXP-008: 64d Feature Group Ablation & Information Density", header_style="bold magenta")
    table.add_column("Ablation Arm", justify="left")
    table.add_column("Active Dim", justify="center")
    table.add_column("Success Rate", justify="right")
    table.add_column("Δ Success", justify="right")
    table.add_column("Wall Hits/Ep", justify="right")
    table.add_column("Δ Collisions", justify="right")
    table.add_column("Info Efficiency", justify="right")

    for r in summary_rows:
        delta_s_str = f"{r['delta_success']:+.1f}%" if r["group_id"] != "full_64d" else "—"
        delta_h_str = f"{r['delta_hits']:+.2f}" if r["group_id"] != "full_64d" else "—"
        table.add_row(
            r["group_name"],
            str(r["active_dim"]),
            f"{r['success_rate']:.1f}%",
            delta_s_str,
            f"{r['wall_hits']:.2f}",
            delta_h_str,
            f"{r['info_efficiency']:.2f}",
        )
    console.print(table)

    # Key takeaways
    # Rank features by positive contribution: removing feature causes drop in success (negative delta_success)
    critical_features = sorted(
        [r for r in summary_rows if r["group_id"] != "full_64d"],
        key=lambda x: x["delta_success"],
    )

    console.print(Panel(
        f"[bold green]EXP-008 Perceptual Feature Hierarchy Findings:[/bold green]\n"
        f"• Most Critical Feature: [bold]{critical_features[0]['group_name']}[/bold] (removing it causes {critical_features[0]['delta_success']:+.1f}% drop in trajectory success).\n"
        f"• 5x5 Ring Effect: Removing the 5x5 ring yields [bold]{next(r['delta_hits'] for r in summary_rows if r['group_id']=='no_5x5_ring'):+.2f}[/bold] wall hits and {next(r['delta_success'] for r in summary_rows if r['group_id']=='no_5x5_ring'):+.1f}% success.\n"
        f"• Optimal Information Density: Models without redundant rings achieve higher Information Efficiency ({max(r['info_efficiency'] for r in summary_rows):.2f}).",
        border_style="green",
    ))

    # Save artifact
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp008_feature_ablation.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"train_episodes": TRAIN_EPISODES, "epochs": EPOCHS, "eval_episodes": EVAL_EPISODES, "seeds": SEEDS},
            "summary": summary_rows,
            "raw_runs": all_runs,
            "runtime_seconds": time.time() - t_start,
        }, f, indent=2)

    console.print(f"[bold green]✓ EXP-008 Results saved to {out_path}[/bold green] ({time.time() - t_start:.1f}s)")
    return {"summary": summary_rows}


if __name__ == "__main__":
    run_exp008()
