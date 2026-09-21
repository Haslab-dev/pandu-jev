"""EXP-004: Memory x Representation Matrix — Clean Representations.

Revised 2026-09-21 per BUG-001 audit finding: the standard 64d
representation is contaminated by the 16-cell 5x5 local-occupancy
ring artifact (which alone doubles wall hits vs the 32d baseline).
EXP-004 therefore uses CLEAN representations — base 16d, 32d, and
128d with the ring REMOVED — to cleanly separate two fundamental
capabilities:
  - spatial resolution  (instantaneous perception of the local
    geometry)
  - temporal memory      (belief about what lies behind walls,
    i.e. Fog-of-War / POMDP reasoning)

Design: the 3x3 patch (32d) and 5x5 ring (64d) features both encode
local wall occupancy at Chebyshev distances 1 and 2. The ring is the
contaminated block. A "clean" version removes the ring while keeping
the ray-based sensing (which is informative but not local-geometry).

The matrix is: representation (clean-16d, clean-32d, clean-128d,
ring-contaminated-64d as control) x memory type (stateless MLP vs
recurrent GRU) x observation regime (full-obs vs Fog-of-War).

Key question: does temporal memory (GRU) rescue Fog-of-War
competence, and does it interact with representation dimension?
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

# Pin threads for reproducibility (method finding of BUG-001).
torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from env.gridworld import GridWorld, Action, create_random_gridworld, Action as GridAction  # noqa: E402
from expert.astar import AStarExpert  # noqa: E402
from models.policy import TinyPolicy  # noqa: E402
from models.recurrent import RecurrentPanduPolicy  # noqa: E402
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Representation definitions.
# ---------------------------------------------------------------------------
# Base 16d:  [0:16]  agent coords, goal coords, distances, 4 wall sensors, 4 cardinal raycasts
# Base 32d:  [0:32]  base16 + diagonal rays, dynamic features, 3x3 patch
# 64d ring-contaminated (control): base32 + 5x5 ring + goal projections + corridor depths
# Clean 32d:  [0:32]  same as base32 but with the 3x3 patch REPLACED by zeros (no local ring)
#   Actually cleaner: keep base32 as-is (the 3x3 patch is small, only 8 cells at dist 1 —
#   not the ring). The RING is only at 64d. So "clean" 16d/32d are the base vectors,
#   and "clean-128d" removes the 7x7 ring.
# Clean 128d: [0:104] base32 + ring16 + goal8 + corridor8 + 16 LIDAR + 24 Fourier,
#             with the 7x7 ring (24 cells, [64:88]) zeroed out.

# The ring indices in the 128d layout: base32(32) + ring16(16) + goal8(8) + corridor8(8) = 64,
# then 7x7 ring = 24 cells at [64:88], then LIDAR(16) at [88:104], Fourier(24) at [104:128].
RING_128 = list(range(64, 88))
# The ring indices in the 64d layout: ring16 at [32:48].
RING_64 = list(range(32, 48))

CLEAN_CONFIGS = [
    {"name": "clean-16d", "dim": 16, "remove": [],   "base": 16},
    {"name": "clean-32d", "dim": 32, "remove": [],   "base": 32},
    {"name": "clean-128d (no ring)", "dim": 128, "remove": RING_128, "base": 128},
    {"name": "ring-contaminated 64d (control)", "dim": 64, "remove": [], "base": 64},
]

MEMORY_TYPES = ["stateless", "recurrent"]
FOG_LEVELS = [False, True]  # full-obs vs Fog-of-War
NUM_EVAL_EPISODES = 50
EVAL_SEED = 123
TRAIN_EPOCHS = 30
TRAIN_SEEDS = [42, 7]  # report mean +/- std


def _make_clean_feature_vector(dim: int, remove: List[int]) -> torch.Tensor:
    """Build a clean-feature-dimension mask identity of length dim."""
    keep = [i for i in range(dim) if i not in remove]
    return torch.tensor(keep, dtype=torch.long)


def _eval_clean_rollout(model: torch.nn.Module, keep: List[int], dim: int,
                        partial_obs: bool, seed: int, n: int = NUM_EVAL_EPISODES) -> Dict[str, float]:
    """Rollout with clean-feature slicing. Works for stateless TinyPolicy AND RecurrentPanduPolicy."""
    model.eval()
    rng = np.random.RandomState(seed)
    hits, succ, steps, reward = 0, 0, 0, 0.0
    for _ in range(n):
        ep_seed = int(rng.randint(0, 1_000_000))
        env = create_random_gridworld(width=12, height=7, wall_prob=0.18,
                                       feature_dim=dim, seed=ep_seed)
        if partial_obs:
            env.partial_obs = True
            env.obs_radius = 3
        h = None
        done = False
        while not done and not env.done and steps < env.max_steps:
            feat = env.get_feature_vector(dim=dim)[keep]
            dist = model.get_action_distribution(feat)
            action = int(dist["action_idx"])
            obs, r, done, info = env.step(action)
            reward += r
            steps += 1
            if info.get("hit_wall", False):
                hits += 1
            if info.get("reached_goal", False):
                succ += 1
            if done:
                break
    n_eff = max(1, n)
    return {"success_rate": succ / n_eff, "wall_hits": hits / n_eff,
            "avg_steps": steps / n_eff, "avg_reward": reward / n_eff}


def _eval_recurrent_rollout(model: RecurrentPanduPolicy, keep: List[int], dim: int,
                            partial_obs: bool, seed: int, n: int = NUM_EVAL_EPISODES) -> Dict[str, float]:
    """Rollout for recurrent models, maintaining GRU state across steps."""
    model.eval()
    rng = np.random.RandomState(seed)
    hits, succ, steps, reward = 0, 0, 0, 0.0
    for _ in range(n):
        ep_seed = int(rng.randint(0, 1_000_000))
        env = create_random_gridworld(width=12, height=7, wall_prob=0.18,
                                       feature_dim=dim, seed=ep_seed)
        if partial_obs:
            env.partial_obs = True
            env.obs_radius = 3
        h = model.init_hidden(1)
        done = False
        while not done and steps < env.max_steps:
            feat = env.get_feature_vector(dim=dim)[keep]
            with torch.no_grad():
                dist = model.get_action_distribution(feat, h)
                h = dist["hidden"]
            action = int(dist["action_idx"])
            obs, r, done, info = env.step(action)
            reward += r
            steps += 1
            if info.get("hit_wall", False):
                hits += 1
            if info.get("reached_goal", False):
                succ += 1
            if done:
                break
    n_eff = max(1, n)
    return {"success_rate": succ / n_eff, "wall_hits": hits / n_eff,
            "avg_steps": steps / n_eff, "avg_reward": reward / n_eff}


def collect_clean_trajectories(dim: int, remove: List[int], num_episodes: int, seed: int):
    """Collect (clean-feature, action) pairs from A* expert, with the ring removed."""
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


def run_exp004() -> Dict[str, Any]:
    """Run the EXP-004 Memory x Representation matrix."""
    console.print(Panel(
        "[bold cyan]EXP-004: Memory x Representation Matrix (clean representations)[/bold cyan]\n"
        "[italic]Per BUG-001, the 64d representation is contaminated by the 5x5 ring. "
        "Clean reps: 16d, 32d (base, no ring), 128d-minus-ring. Control: ring-contaminated 64d.[/italic]",
        border_style="cyan",
    ))

    t0 = time.time()
    results = []

    for cfg in CLEAN_CONFIGS:
        dim, remove, name = cfg["dim"], cfg["remove"], cfg["name"]
        keep = [i for i in range(dim) if i not in remove]
        n_params_stateless = None

        for mem_type in MEMORY_TYPES:
            for fog in FOG_LEVELS:
                for train_seed in TRAIN_SEEDS:
                    # --- collect clean expert data ---
                    X, y = collect_clean_trajectories(dim, remove, num_episodes=600, seed=train_seed)
                    if len(X) == 0:
                        continue

                    if mem_type == "stateless":
                        model = TinyPolicy(input_dim=X.shape[1], num_actions=4, hidden_dims=(48, 64))
                        # NOTE: TinyPolicy input_dim = len(keep), not original dim.
                        train_behavioral_cloning(model, X, y, epochs=TRAIN_EPOCHS, batch_size=64,
                                                    lr=1e-3, device=torch.device("cpu"), seed=train_seed)
                        # eval
                        cl = _eval_clean_rollout(model, keep, dim, partial_obs=fog, seed=EVAL_SEED)
                        n_params_stateless = model.count_parameters()
                    else:
                        # recurrent: use full dim for init_hidden, but input_dim = len(keep)
                        model = RecurrentPanduPolicy(input_dim=X.shape[1], hidden_dim=32, num_actions=4)
                        # Train with BPTT over full episodes
                        from training.bc import train_behavioral_cloning as _  # noqa: F401
                        # Build episode sequences for BPTT
                        expert = AStarExpert()
                        episodes = []
                        rng = np.random.RandomState(train_seed)
                        for _ in range(200):
                            ep_seed = int(rng.randint(0, 1_000_000))
                            e = create_random_gridworld(width=12, height=7, wall_prob=0.18, feature_dim=dim, seed=ep_seed)
                            feats, acts = [], []
                            done = False
                            while not done and len(acts) < e.max_steps:
                                a = expert.get_action(e)
                                if a is None: break
                                f = e.get_feature_vector(dim=dim)[keep]
                                feats.append(f); acts.append(int(a))
                                o, r, done, i = e.step(a)
                            if acts and i.get("reached_goal", False):
                                episodes.append((np.array(feats, dtype=np.float32), np.array(acts, dtype=np.int64)))
                        # train recurrent
                        model.to(torch.device("cpu"))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                        for epoch in range(TRAIN_EPOCHS):
                            model.train()
                            perm = np.random.permutation(len(episodes))
                            for idx in perm:
                                x_seq = torch.tensor(episodes[idx][0], dtype=torch.float32).unsqueeze(0)
                                y_seq = torch.tensor(episodes[idx][1], dtype=torch.long)
                                optimizer.zero_grad()
                                all_logits, _ = model.forward_sequence(x_seq)
                                loss = torch.nn.functional.cross_entropy(all_logits.squeeze(0), y_seq)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                        cl = _eval_recurrent_rollout(model, keep, dim, partial_obs=fog, seed=EVAL_SEED)
                        n_params_stateless = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    result = {
                        "representation": name, "dim": dim, "memory": mem_type,
                        "fog_of_war": fog, "train_seed": train_seed,
                        "success_rate": cl["success_rate"], "wall_hits": cl["wall_hits"],
                        "avg_steps": cl["avg_steps"], "avg_reward": cl["avg_reward"],
                        "params": n_params_stateless,
                    }
                    results.append(result)
                    console.print(f"   {name:<28} {mem_type:<10} fog={fog!s:<5} seed={train_seed} | "
                                  f"succ={cl['success_rate']*100:.1f}% | hits={cl['wall_hits']:.2f} | "
                                  f"params={n_params_stateless:,}")

    # ---- Summary table ----
    console.print(Panel("[bold cyan]EXP-004 Summary Matrix[/bold cyan]", border_style="cyan"))
    table = Table(header_style="bold magenta")
    table.add_column("Representation", justify="center")
    table.add_column("Memory", justify="center")
    table.add_column("Fog-of-War", justify="center")
    table.add_column("Success", justify="right")
    table.add_column("Hits/Ep", justify="right")
    table.add_column("Params", justify="right")
    for r in results:
        table.add_row(
            r["representation"][:16], r["memory"], str(r["fog_of_war"]),
            f"{r['success_rate']*100:.1f}%", f"{r['wall_hits']:.2f}", f"{r['params']:,}",
        )
    console.print(table)

    # ---- Key comparisons ----
    # Does recurrent memory rescue Fog-of-War? Compare clean-32d stateless vs recurrent under FoW.
    def best(result_set: List[Dict], rep: str, mem: str, fog: bool) -> Dict:
        matches = [r for r in result_set if r["representation"] == rep and r["memory"] == mem and r["fog_of_war"] == fog]
        return min(matches, key=lambda x: x["wall_hits"]) if matches else {}

    console.print("\n[bold]Key comparison: does memory help under Fog-of-War?[/bold]")
    for rep in ["clean-16d", "clean-32d"]:
        for mem in ["stateless", "recurrent"]:
            for fog in [False, True]:
                succs = [r["success_rate"] for r in results if r["representation"] == rep and r["memory"] == mem and r["fog_of_war"] == fog]
                hits = [r["wall_hits"] for r in results if r["representation"] == rep and r["memory"] == mem and r["fog_of_war"] == fog]
                if succs:
                    console.print(f"   {rep:<12} {mem:<10} fog={fog!s:<5} succ={np.mean(succs)*100:.1f}% (mean), hits={np.mean(hits):.2f}")

    # Save
    report = {"results": results, "config": {"train_epochs": TRAIN_EPOCHS, "eval_episodes": NUM_EVAL_EPISODES,
                 "train_seeds": TRAIN_SEEDS, "fog_obs_radius": 3},
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp004_memory_representation_matrix.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]EXP-004 saved to {path}[/green] [dim]({time.time()-t0:.1f}s)[/dim]")
    return report


if __name__ == "__main__":
    run_exp004()
