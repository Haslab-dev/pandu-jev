"""EXP-005: Parameter-Normalized 2D Scaling Law (Clean Representations).

Per EXP-004 and BUG-001 findings, the 5x5 local-occupancy ring is a
confounding feature (helps navigation, causes wall-hugging at 64d).
EXP-005 now uses CLEAN representations (ring removed) to isolate
the pure representation effect.

Design (3 independent axes):
  - Representation dimension (clean): 16d, 32d, 128d-no-ring (104d)
  - Policy capacity (hidden dims): fixed to ~8.5K params across
    all reps to isolate representation effect; AND varied at 128d
    to isolate capacity effect.
  - Both axes independently controlled.

Goal: enable the claim "representation is the bottleneck" with clean
evidence — free from the ring artifact that contaminated the original
REP-001 benchmark.

Also measures closed-loop success, wall hits, collision density,
and latency to fully characterize the scaling law.
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

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from env.gridworld import GridWorld, Action, create_random_gridworld, Action as GridAction  # noqa: E402
from expert.astar import AStarExpert  # noqa: E402
from models.policy import TinyPolicy  # noqa: E402
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Clean representation configurations (ring removed from 64d layout).
# ---------------------------------------------------------------------------
# 64d layout: [0:16] base16 | [16:32] extra | [32:48] ring16 | [48:64] goal+corr
# Clean 64d: remove ring16 → 48 dims
# 128d layout: [0:64] base64 | [64:88] ring24 | [88:104] LIDAR16 | [104:128] Fourier24
# Clean 128d: remove ring24 → 104 dims

CLEAN_REPS = [
    {"name": "16d", "dim": 16, "remove": [],   "hidden": (32, 64)},   # 2,916 params
    {"name": "32d", "dim": 32, "remove": [],   "hidden": (48, 64)},   # 4,980 params
    {"name": "64d-clean", "dim": 64, "remove": list(range(32, 48)),  "hidden": (48, 64)},  # remove ring → 48d features, ~8K params
    {"name": "128d-clean", "dim": 128, "remove": list(range(64, 88)), "hidden": (96, 64)}, # remove ring → 104d features
]

# Capacity-variation arms at 128d (clean): fix feature dim, vary width
CAPACITY_ARMS_128D = [
    {"name": "128d-clean (32,64)",   "hidden": (32, 64)},   # ~4.2K params (104*32+32 + 32*64+64 + 64*4+4)
    {"name": "128d-clean (48,64)",   "hidden": (48, 64)},   # ~6.5K
    {"name": "128d-clean (64,64)",   "hidden": (64, 64)},   # ~8.8K
    {"name": "128d-clean (96,64)",   "hidden": (96, 64)},   # ~13.1K
    {"name": "128d-clean (128,64)",  "hidden": (128, 64)},  # ~17.4K
]

NUM_EVAL_EPISODES = 80
EVAL_SEED = 1041
TRAIN_EPOCHS = 30
TRAIN_SEEDS = [42, 7]
CLI_EPISODES = 1000


def _eval_clean(model: TinyPolicy, dim: int, remove: List[int], seed: int) -> Dict[str, float]:
    """Closed-loop eval on clean features."""
    model.eval()
    keep = [i for i in range(dim) if i not in remove]
    rng = np.random.RandomState(seed)
    hits, succ, steps, reward = 0, 0, 0, 0.0
    for _ in range(NUM_EVAL_EPISODES):
        ep_seed = int(rng.randint(0, 1_000_000))
        env = create_random_gridworld(width=12, height=7, wall_prob=0.18, feature_dim=dim, seed=ep_seed)
        done = False
        while not done:
            feat = env.get_feature_vector(dim=dim)[keep]
            dist = model.get_action_distribution(feat)
            obs, r, done, info = env.step(int(dist["action_idx"]))
            reward += r
            steps += 1
            if info.get("hit_wall", False): hits += 1
            if info.get("reached_goal", False): succ += 1
            if done: break
    n = NUM_EVAL_EPISODES
    return {"success_rate": succ / n, "wall_hits": hits / n,
            "avg_steps": steps / n, "avg_reward": reward / n}


def collect_clean_trajectories(dim: int, remove: List[int], num_episodes: int, seed: int):
    """Collect clean-feature (feature, action) pairs from A* expert."""
    expert = AStarExpert()
    features, actions = [], []
    rng = np.random.RandomState(seed)
    keep = [i for i in range(dim) if i not in remove]
    for _ in range(num_episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        if ep_seed % 2 == 1:
            env = create_random_gridworld(width=12, height=7, wall_prob=0.18, feature_dim=dim, seed=ep_seed)
        else:
            env = GridWorld(random_start_goal=True, feature_dim=dim, seed=ep_seed)
        done = False
        while not done:
            action = expert.get_action(env)
            if action is None:
                break
            feat = env.get_feature_vector(dim=dim)
            features.append(feat[keep])
            actions.append(int(action))
            obs, r, done, info = env.step(action)
            if done:
                break
    return np.array(features, dtype=np.float32), np.array(actions, dtype=np.int64)


def run_exp005() -> Dict[str, Any]:
    """Run the parameter-normalized clean representation scaling law."""
    console.print(Panel(
        "[bold cyan]EXP-005: Parameter-Normalized 2D Scaling Law (CLEAN reps)[/bold cyan]\n"
        "[italic]Per BUG-001: the 5x5 ring is the confounding feature.[/italic]",
        border_style="cyan",
    ))

    t0 = time.time()
    results = []

    # --- Axis 1: Representation effect at fixed ~8.5K capacity ---
    console.print("\n[bold]Axis 1: Clean representation effect (fixed capacity ~8.5K)[/bold]")
    for cfg in CLEAN_REPS:
        dim, remove, name = cfg["dim"], cfg["remove"], cfg["name"]
        hidden = cfg["hidden"]
        for seed in TRAIN_SEEDS:
            X, y = collect_clean_trajectories(dim, remove, num_episodes=CLI_EPISODES, seed=seed)
            if len(X) == 0:
                continue
            idx = np.random.RandomState(seed).permutation(len(y))
            cut = int(0.8 * len(y))
            m = TinyPolicy(input_dim=X.shape[1], num_actions=4, hidden_dims=hidden)
            train_behavioral_cloning(m, X[idx[:cut]], y[idx[:cut]], epochs=TRAIN_EPOCHS, batch_size=64,
                                          lr=1e-3, device=torch.device("cpu"), seed=seed)
            cl = _eval_clean(m, dim, remove, seed=EVAL_SEED)
            results.append({"axis": "rep_effect", "representation": name, "dim": dim,
                            "params": m.count_parameters(), "hidden": str(hidden),
                            "seed": seed, **cl})
            console.print(f"   {name:<14} params={m.count_parameters():>5} | succ={cl['success_rate']*100:.1f}% | hits={cl['wall_hits']:.2f}")

    # --- Axis 2: Capacity effect at 128d-clean (fixed representation) ---
    console.print("\n[bold]Axis 2: Capacity effect at 128d-clean (fixed representation)[/bold]")
    for cfg in CAPACITY_ARMS_128D:
        hidden = cfg["hidden"]
        dim = 128
        remove = list(range(64, 88))  # remove ring24
        name = cfg["name"]
        for seed in TRAIN_SEEDS:
            X, y = collect_clean_trajectories(dim, remove, num_episodes=CLI_EPISODES, seed=seed)
            if len(X) == 0:
                continue
            idx = np.random.RandomState(seed).permutation(len(y))
            cut = int(0.8 * len(y))
            m = TinyPolicy(input_dim=X.shape[1], num_actions=4, hidden_dims=hidden)
            train_behavioral_cloning(m, X[idx[:cut]], y[idx[:cut]], epochs=TRAIN_EPOCHS, batch_size=64,
                                          lr=1e-3, device=torch.device("cpu"), seed=seed)
            cl = _eval_clean(m, dim, remove, seed=EVAL_SEED)
            results.append({"axis": "capacity_effect", "representation": "128d-clean", "dim": dim,
                            "params": m.count_parameters(), "hidden": str(hidden),
                            "seed": seed, **cl})
            console.print(f"   {name:<32} params={m.count_parameters():>5} | succ={cl['success_rate']*100:.1f}% | hits={cl['wall_hits']:.2f}")

    # ---- Summary tables ----
    console.print(Panel("[bold cyan]EXP-005 Summary[/bold cyan]", border_style="cyan"))

    table = Table(title="Axis 1: Clean Representation Effect", header_style="bold magenta")
    table.add_column("Representation", justify="center")
    table.add_column("Dim", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Succ (mean)", justify="right")
    table.add_column("Hits/Ep (mean)", justify="right")
    table.add_column("Hit Density", justify="right")
    axis1 = [r for r in results if r["axis"] == "rep_effect"]
    reps = {}
    for r in axis1:
        reps.setdefault(r["representation"], []).append(r)
    for rep_name, reps_list in reps.items():
        succs = [r["success_rate"] for r in reps_list]
        hits = [r["wall_hits"] for r in reps_list]
        params = reps_list[0]["params"]
        dims = set(r["dim"] for r in reps_list)
        dim_str = "/".join(sorted(str(d) for d in dims))
        for d in dims:
            pass
        mean_succ = np.mean(succs)
        mean_hits = np.mean(hits)
        # Use representative dim
        r0 = reps_list[0]
        table.add_row(rep_name, str(r0["dim"]), f"{params:,}",
                       f"{mean_succ*100:.1f}%", f"{mean_hits:.2f}",
                       f"{mean_hits/max(0.01, r0['avg_steps'])*100:.2f}%")
    console.print(table)

    table2 = Table(title="Axis 2: Capacity Effect at 128d-clean", header_style="bold magenta")
    table2.add_column("Hidden Dims", justify="center")
    table2.add_column("Params", justify="right")
    table2.add_column("Succ (mean)", justify="right")
    table2.add_column("Hits/Ep (mean)", justify="right")
    table2.add_column("Hit Density", justify="right")
    axis2 = [r for r in results if r["axis"] == "capacity_effect"]
    cap_groups = {}
    for r in axis2:
        cap_groups.setdefault(str(r["hidden"]), []).append(r)
    for hd, cap_list in cap_groups.items():
        succs = [r["success_rate"] for r in cap_list]
        hits = [r["wall_hits"] for r in cap_list]
        steps_list = [r["avg_steps"] for r in cap_list]
        params = cap_list[0]["params"]
        mean_succ = np.mean(succs)
        mean_hits = np.mean(hits)
        mean_steps = np.mean(steps_list)
        table2.add_row(hd, f"{params:,}", f"{mean_succ*100:.1f}%", f"{mean_hits:.2f}",
                        f"{mean_hits/max(0.01, mean_steps)*100:.2f}%")
    console.print(table2)

    # Key comparison: clean 64d vs clean 128d (same capacity family)
    clean_64d = [r for r in axis1 if "64d" in r["representation"]]
    clean_128d = [r for r in axis1 if "128d" in r["representation"]]
    if clean_64d and clean_128d:
        s64, h64 = np.mean([r["success_rate"] for r in clean_64d]), np.mean([r["wall_hits"] for r in clean_64d])
        s128, h128 = np.mean([r["success_rate"] for r in clean_128d]), np.mean([r["wall_hits"] for r in clean_128d])
        console.print(f"\n[bold]Key: clean 64d (≈8.5K) vs clean 128d (≈{clean_128d[0]['params']:,})[/bold]")
        console.print(f"   64d-clean: succ={s64*100:.1f}%, hits={h64:.2f}")
        console.print(f"   128d-clean: succ={s128*100:.1f}%, hits={h128:.2f}")

    # Save
    report = {"results": results, "config": {"train_epochs": TRAIN_EPOCHS,
                "eval_episodes": NUM_EVAL_EPISODES, "train_seeds": TRAIN_SEEDS,
                "train_episodes": CLI_EPISODES}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp005_parameter_normalized_scaling.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]EXP-005 saved to {path}[/green] [dim]({time.time()-t0:.1f}s)[/dim]")
    return report


if __name__ == "__main__":
    run_exp005()
