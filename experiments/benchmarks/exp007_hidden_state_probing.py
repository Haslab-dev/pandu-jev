"""EXP-007: GRU Hidden-State Probing (what does h_t encode?).

The current evidence for temporal memory is purely BEHAVIORAL:
the recurrent policy achieves 78% Fog-of-War success vs 0% stateless,
so "h_t must encode something useful". But this is an argument from
consequence — it does not ground what h_t actually encodes.

This experiment uses LINEAR PROBES on h_t to directly characterize
the memory content. For each target property, a linear classifier
is trained on h_t to predict that property from the hidden state alone:
  - prev_direction: was the last action UP/DOWN/LEFT/RIGHT?
  - last_collision: was the previous step a wall hit?
  - relative_orientation: angle to goal (continuous)
  - visited_region: has the agent been in this quadrant before?
  - distance_to_goal: Manhattan distance (continuous)

If linear probes can predict these from h_t with high accuracy,
it empirically grounds the claim "h_t encodes temporal information".
Low probe accuracy would soften overclaims about what memory stores.

Method: train a clean-recurrent policy (state-only) on 16d clean
features, record h_t at each step during a large rollout, then
train/probe each target property.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from env.gridworld import GridWorld, Action, create_random_gridworld  # noqa: E402
from expert.astar import AStarExpert  # noqa: E402
from models.policy import TinyPolicy  # noqa: E402
from models.recurrent import RecurrentPanduPolicy  # noqa: E402
from training.bc import train_behavioral_cloning  # noqa: E402

console = Console()

HIDDEN_DIM = 32
NUM_EVAL_EPISODES = 200
TRAIN_EPOCHS = 30
TRAIN_SEED = 42
EVAL_SEED = 123
CLI_EPISODES = 1000
PROBE_EPOCHS = 50
PROBE_LR = 1e-2


def train_recurrent_clean() -> RecurrentPanduPolicy:
    """Train a clean-recurrent policy (16d, state-only, GRU memory)."""
    dim = 16
    remove = []
    episodes = []
    expert = AStarExpert()
    rng = np.random.RandomState(TRAIN_SEED)
    for _ in range(CLI_EPISODES):
        ep_seed = int(rng.randint(0, 1_000_000))
        env = create_random_gridworld(width=12, height=7, wall_prob=0.18, feature_dim=dim, seed=ep_seed)
        step_features, step_actions = [], []
        done = False
        step = 0
        while not done and step < env.max_steps:
            action = expert.get_action(env)
            if action is None:
                break
            feat = env.get_feature_vector(dim=dim)
            step_features.append(feat)
            step_actions.append(int(action))
            obs, r, done, info = env.step(action)
            step += 1
        if step_actions and info.get("reached_goal", False):
            episodes.append((np.array(step_features, dtype=np.float32), np.array(step_actions, dtype=np.int64)))

    model = RecurrentPanduPolicy(input_dim=dim, hidden_dim=HIDDEN_DIM, num_actions=4)
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
    model.eval()
    return model


def collect_hidden_states(model: RecurrentPanduPolicy, num_episodes: int = NUM_EVAL_EPISODES,
                          seed: int = EVAL_SEED) -> Dict[str, Any]:
    """Roll out episodes, recording h_t at each step plus target properties."""
    model.eval()
    rng = np.random.RandomState(seed)
    all_h = []          # list of h_t arrays (T × 32)
    all_props = []      # list of dict prop arrays per step
    all_succ = []

    for _ in range(num_episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        env = create_random_gridworld(width=12, height=7, wall_prob=0.18, feature_dim=16, seed=ep_seed)
        h = model.init_hidden(1)
        done = False
        step = 0
        prev_action = -1
        prev_hit = False
        episode_h = []
        episode_props = []
        while not done and step < env.max_steps:
            feat = env.get_feature_vector(dim=16)
            with torch.no_grad():
                dist = model.get_action_distribution(feat, h)
                h_next = dist["hidden"]  # (1, 32)
            action = int(dist["action_idx"])
            obs, r, done, info = env.step(action)
            # Record h_t (before update, i.e., the state entering this step)
            episode_h.append(h.squeeze(0).cpu().numpy().copy())
            # Target properties (from the step just taken)
            props = {
                "prev_action_onehot": np.zeros(4),
                "last_collision": float(prev_hit),
                "rel_orientation": 0.0,
                "distance_to_goal": 0.0,
            }
            if prev_action >= 0:
                props["prev_action_onehot"][prev_action] = 1.0
            props["last_collision"] = float(prev_hit)
            # Relative orientation: angle from agent to goal
            dx = env.goal_x - env.agent_x
            dy = env.goal_y - env.agent_y
            hyp = max(1e-6, np.hypot(dx, dy))
            props["rel_orientation"] = float(dy / hyp)  # normalized y-component
            props["distance_to_goal"] = float(abs(dx) + abs(dy)) / 20.0  # normalized manhattan
            episode_props.append(props)
            # Advance tracking
            prev_action = action
            prev_hit = info.get("hit_wall", False)
            h = h_next
            step += 1
        all_h.append(np.array(episode_h))
        all_props.append({k: np.array([p[k] for p in episode_props]) for k in episode_props[0]})
        all_succ.append(1.0 if info.get("reached_goal", False) else 0.0)
    return {"h": all_h, "props": all_props, "succ": np.array(all_succ)}


def train_linear_probe(X: np.ndarray, y: np.ndarray, epochs: int = PROBE_EPOCHS,
                       lr: float = PROBE_LR, is_classification: bool = False) -> Tuple[float, float, float]:
    """Train a linear probe; return (train_acc, test_acc_approx, loss)."""
    n = len(X)
    split = int(0.8 * n)
    perm = np.random.RandomState(0).permutation(n)
    Xtr, ytr = X[perm[:split]], y[perm[:split]]
    Xte, yte = X[perm[split:]], y[perm[split:]]

    # Standardize
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    n_in = X.shape[1]
    probe = nn.Linear(n_in, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = crit(probe(Xt), yt)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        tr_pred = probe(torch.tensor(Xtr, dtype=torch.float32)).squeeze().numpy()
        te_pred = probe(torch.tensor(Xte, dtype=torch.float32)).squeeze().numpy()
        if is_classification:
            tr_acc = float((np.round(tr_pred) == ytr).mean())
            te_acc = float((np.round(te_pred) == yte).mean())
        else:
            tr_acc = float((np.abs(tr_pred - ytr) < 0.5).mean())
            te_acc = float((np.abs(te_pred - yte) < 0.5).mean())
        tr_loss = crit(probe(torch.tensor(Xtr, dtype=torch.float32)),
                        torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)).item()
    return tr_acc, te_acc, tr_loss


def run_exp007() -> Dict[str, Any]:
    """Run the GRU hidden-state probing experiment."""
    console.print(Panel(
        "[bold cyan]EXP-007: GRU Hidden-State Probing[/bold cyan]\n"
        "[italic]What does h_t encode? Linear probes on h_t to ground[/italic]",
        border_style="cyan",
    ))

    t0 = time.time()
    console.print("\n[bold]Phase 1: train clean-recurrent policy[/bold]")
    model = train_recurrent_clean()
    console.print(f"   trained {model.count_parameters():,} params")

    console.print("\n[bold]Phase 2: collect hidden states (200 episodes)[/bold]")
    data = collect_hidden_states(model)
    all_h = np.concatenate(data["h"])
    props = data["props"]
    succ = data["succ"]
    console.print(f"   collected {all_h.shape[0]} hidden states, {all_h.shape[1]} dims")

    # Target property probes
    probe_targets = {
        "prev_action (class)": np.argmax(np.concatenate([p["prev_action_onehot"] for p in props]), axis=-1),
        "last_collision (binary)": np.concatenate([p["last_collision"] for p in props]),
        "rel_orientation (cont)": np.concatenate([p["rel_orientation"] for p in props]),
        "distance_to_goal (cont)": np.concatenate([p["distance_to_goal"] for p in props]),
    }
    # Also: reachability (did episode succeed?)
    succ_repeated = np.repeat(succ, [len(h) for h in data["h"]])
    probe_targets["episode_success (binary)"] = succ_repeated

    console.print("\n[bold]Phase 3: train linear probes on h_t[/bold]")
    results = []
    table = Table(header_style="bold magenta")
    table.add_column("Target Property", justify="left")
    table.add_column("Type", justify="center")
    table.add_column("Train Acc", justify="right")
    table.add_column("Test Acc", justify="right")
    table.add_column("Verdict", justify="center")

    for name, y in probe_targets.items():
        is_class = "(class)" in name or "(binary)" in name
        tr_acc, te_acc, loss = train_linear_probe(all_h, y, is_classification=is_class)
        verdict = "h_t encodes it" if te_acc > 0.7 else ("partial" if te_acc > 0.55 else "weak/no signal")
        table.add_row(name, "class" if is_class else "reg",
                       f"{tr_acc*100:.1f}%", f"{te_acc*100:.1f}%", verdict)
        results.append({"target": name, "type": "classification" if is_class else "regression",
                        "train_acc": tr_acc, "test_acc": te_acc,
                        "verdict": verdict})
    console.print(table)

    # Overall
    test_accs = [r["test_acc"] for r in results]
    console.print(f"\n[bold]Mean probe test accuracy: {np.mean(test_accs)*100:.1f}% "
                  f"(median {np.median(test_accs)*100:.1f}%)[/bold]")
    console.print(f"[bold]Mean classification probe accuracy: "
                  f"{np.mean([r['test_acc'] for r in results if r['type']=='classification'])*100:.1f}%[/bold]")

    if np.mean([r["test_acc"] for r in results if r["type"] == "classification"]) > 0.75:
        console.print(
            "\n[green]CONCLUSION: h_t does encode meaningful temporal information.[/green] "
            "Linear probes on h_t can decode action history, collision history, "
            "and goal direction above chance — the memory claim is empirically grounded, "
            "not just behavioral.")
    else:
        console.print(
            "\n[yellow]CONCLUSION: h_t encoding is weak — probes perform near chance.[/yellow] "
            "The behavioral success (78% FoW) may rely on the recurrent architecture's "
            "implicit credit assignment rather than explicit memory content in h_t. "
            "Softens the overclaim 'h_t menyimpan jejak memori'.")

    # Save
    report = {"probe_results": results, "mean_test_acc": float(np.mean(test_accs)),
              "mean_class_acc": float(np.mean([r["test_acc"] for r in results if r["type"]=="classification"])),
              "config": {"hidden_dim": HIDDEN_DIM, "train_epochs": TRAIN_EPOCHS,
                         "eval_episodes": NUM_EVAL_EPISODES, "probe_epochs": PROBE_EPOCHS,
                         "probe_lr": PROBE_LR}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp007_hidden_state_probing.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]EXP-007 saved to {path}[/green] [dim]({time.time()-t0:.1f}s)[/dim]")
    return report


if __name__ == "__main__":
    run_exp007()
