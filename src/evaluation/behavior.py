"""Closed-Loop Behavioral Metrics for Pandu (pandu-jev).

EVL-001: Pandu is a *controller*, not a classifier. Single-step validation
accuracy does not equal closed-loop competence: small per-step errors
compound over a trajectory. This module instruments full-episode rollouts
with the behavioral metrics that matter for control:

- action_accuracy: agreement with the A* expert at every visited state
- trajectory_success_rate: episodes reaching the goal within max_steps
- mean_episode_length: average steps per episode (successes and failures)
- collision_rate / wall_hits_per_episode: wall-bump frequency
- error_recovery_rate: fraction of wall hits followed by an episode that
  still reaches the goal
- failure_outcome breakdown: timeout vs stuck vs reached
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from env.gridworld import GridWorld, Action, create_random_gridworld
from expert.astar import AStarExpert


def rollout_episode(
    model: torch.nn.Module,
    env: GridWorld,
    expert: Optional[AStarExpert] = None,
    device: Optional[torch.device] = None,
    record_gru_hidden: bool = False,
    feature_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Roll out one closed-loop episode, instrumenting per-step behavior.

    Works for feedforward models exposing ``get_action_distribution(feat)``
    and recurrent models exposing ``get_action_distribution(feat, h)``
    (auto-detected). Returns per-episode aggregates plus per-step traces
    for downstream probing.
    """
    is_recurrent = hasattr(model, "init_hidden")
    h = model.init_hidden(1, device) if is_recurrent else None

    expert_model = expert if expert is not None else AStarExpert()

    features: List[np.ndarray] = []
    actions: List[int] = []
    expert_actions: List[int] = []
    hiddens: List[np.ndarray] = []

    ep_reward = 0.0
    ep_steps = 0
    wall_hits = 0
    wall_hit_steps: List[int] = []
    recovered_after_hit = False
    reached_goal = False
    latencies_ms: List[float] = []

    while ep_steps < env.max_steps:
        feat = env.get_feature_vector()
        if feature_indices is not None:
            feat = feat[feature_indices]

        t0 = time.perf_counter()
        if is_recurrent:
            dist = model.get_action_distribution(feat, h)
            h = dist["hidden"]
        else:
            dist = model.get_action_distribution(feat)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        action = int(dist["action_idx"])

        features.append(feat)
        actions.append(action)
        if record_gru_hidden and is_recurrent and h is not None:
            hiddens.append(np.asarray(h.detach().cpu().numpy()[0], dtype=np.float32))

        if expert_model is not None:
            expert_action = expert_model.get_action(env)
            expert_actions.append(int(expert_action) if expert_action is not None else action)

        obs, r, done, info = env.step(action)
        ep_reward += float(r)
        ep_steps += 1

        if info.get("hit_wall", False):
            wall_hits += 1
            wall_hit_steps.append(ep_steps)
            recovered_after_hit = False  # reset: recovery not yet demonstrated
        elif wall_hits > 0 and not recovered_after_hit:
            recovered_after_hit = True  # a clean step after a hit = recovery

        if info.get("reached_goal", False):
            reached_goal = True
        if done:
            break

    # Outcome taxonomy: reached | stuck (wall-thrash: >=50% of steps hit walls) | timeout
    if reached_goal:
        outcome = "reached"
    elif wall_hits >= 0.5 * max(1, ep_steps):
        outcome = "stuck"
    else:
        outcome = "timeout"

    result: Dict[str, Any] = {
        "reached_goal": reached_goal,
        "steps": ep_steps,
        "reward": ep_reward,
        "wall_hits": wall_hits,
        "wall_hit_steps": wall_hit_steps,
        "recovered_after_hit": recovered_after_hit,
        "outcome": outcome,
        "action_accuracy": float(np.mean(np.array(actions) == np.array(expert_actions))) if actions else 0.0,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "visited_states": len(features),
    }
    if record_gru_hidden:
        result["features"] = np.array(features, dtype=np.float32)
        result["actions"] = np.array(actions, dtype=np.int64)
        result["expert_actions"] = np.array(expert_actions, dtype=np.int64)
        result["hiddens"] = np.array(hiddens, dtype=np.float32) if hiddens else None
    return result


def evaluate_closed_loop_behavior(
    model: torch.nn.Module,
    num_episodes: int = 100,
    map_types: str = "random",
    fog_of_war: bool = False,
    fog_radius: int = 3,
    feature_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = 123,
    record_traces: bool = False,
    feature_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Closed-loop behavioral evaluation across procedurally generated episodes.

    This is the controller-grade replacement for accuracy-only reporting:
    every downstream experiment (EXP-004..007) reports these metrics.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    feat_dim = feature_dim if feature_dim is not None else getattr(model, "input_dim", 16)
    rng = np.random.RandomState(seed)

    episodes: List[Dict[str, Any]] = []
    expert = AStarExpert()

    for ep in range(num_episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        if map_types == "random":
            env = create_random_gridworld(
                width=12, height=7, wall_prob=0.18,
                feature_dim=feat_dim, seed=ep_seed,
            )
        else:
            env = GridWorld(random_start_goal=True, feature_dim=feat_dim, seed=ep_seed)

        if fog_of_war:
            env.partial_obs = True
            env.obs_radius = fog_radius

        rec = rollout_episode(
            model, env, expert=expert, device=device, record_gru_hidden=record_traces,
            feature_indices=feature_indices,
        )
        rec["episode"] = ep
        rec["seed"] = ep_seed
        episodes.append(rec)

    n = max(1, num_episodes)
    successes = sum(1 for e in episodes if e["reached_goal"])
    total_wall_hits = sum(e["wall_hits"] for e in episodes)
    episodes_with_hits = [e for e in episodes if e["wall_hits"] > 0]
    recovered = sum(
        1 for e in episodes_with_hits
        if e["wall_hit_steps"] and e["wall_hit_steps"][-1] < e["steps"]
    )

    metrics = {
        "episodes": num_episodes,
        "trajectory_success_rate": successes / n,
        "action_accuracy": float(np.mean([e["action_accuracy"] for e in episodes])),
        "mean_episode_length": float(np.mean([e["steps"] for e in episodes])),
        "mean_episode_length_success": float(np.mean([e["steps"] for e in episodes if e["reached_goal"]])) if successes else 0.0,
        "wall_hits_per_episode": total_wall_hits / n,
        "wall_hits_per_episode_median": float(np.median([e["wall_hits"] for e in episodes])),
        "collision_rate_per_step": total_wall_hits / max(1, sum(e["steps"] for e in episodes)),
        "error_recovery_rate": recovered / max(1, len(episodes_with_hits)),
        "episodes_with_collisions_pct": len(episodes_with_hits) / n,
        "outcome_breakdown": {
            "reached": sum(1 for e in episodes if e["outcome"] == "reached"),
            "timeout": sum(1 for e in episodes if e["outcome"] == "timeout"),
            "stuck": sum(1 for e in episodes if e["outcome"] == "stuck"),
        },
        "mean_latency_ms": float(np.mean([e["mean_latency_ms"] for e in episodes])),
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
    }
    if record_traces:
        metrics["episode_traces"] = episodes
    return metrics


def format_behavior_report(metrics: Dict[str, Any]) -> str:
    """Compact one-line-per-metric report for console output."""
    ob = metrics.get("outcome_breakdown", {})
    lines = [
        f"  Trajectory Success : {metrics['trajectory_success_rate'] * 100:.1f}%",
        f"  Action Accuracy    : {metrics['action_accuracy'] * 100:.1f}%  (A* agreement, per-step)",
        f"  Mean Ep Length     : {metrics['mean_episode_length']:.1f} steps "
        f"(successes: {metrics.get('mean_episode_length_success', 0.0):.1f})",
        f"  Wall Hits / Ep     : {metrics['wall_hits_per_episode']:.2f} "
        f"(median {metrics.get('wall_hits_per_episode_median', 0.0):.1f}; "
        f"{metrics['episodes_with_collisions_pct'] * 100:.0f}% of episodes)",
        f"  Collision Rate     : {metrics['collision_rate_per_step'] * 100:.2f}% of steps",
        f"  Error Recovery     : {metrics['error_recovery_rate'] * 100:.1f}% of collision episodes escape thrash",
        f"  Outcomes           : reached={ob.get('reached', 0)} timeout={ob.get('timeout', 0)} stuck(thrash)={ob.get('stuck', 0)}",
        f"  Mean Latency       : {metrics['mean_latency_ms']:.3f} ms",
    ]
    return "\n".join(lines)
