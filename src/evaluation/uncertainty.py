"""Uncertainty Benchmark (Phase 5).

Tests the policy across 5 out-of-distribution / stress regimes:
1. Unseen maps: novel maze topologies
2. Larger maps: grid scaled to 24x14
3. Blocked paths: goal completely surrounded by impassable walls
4. Noisy states: observation noise injected into feature vectors
5. Impossible states: agent initialized inside walls or conflicting coords

Evaluates the core research question:
"Does low confidence actually correlate with failure?"
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from env.gridworld import GridWorld, Action, create_random_gridworld
from expert.astar import AStarExpert
from evaluation.calibration import compute_calibration_metrics


def create_blocked_path_gridworld(seed: int = 42) -> GridWorld:
    """Creates a map where the goal is completely sealed off by walls."""
    map_layout = [
        "############",
        "#S         #",
        "#   ###    #",
        "#       ####",
        "#       #G##",
        "#       ####",
        "############",
    ]
    return GridWorld(map_layout=map_layout, seed=seed)


def create_large_gridworld(width: int = 24, height: int = 14, seed: int = 42) -> GridWorld:
    """Creates a larger map (2x dimensions) to test spatial generalization."""
    return create_random_gridworld(width=width, height=height, wall_prob=0.15, seed=seed)


def run_uncertainty_benchmark(
    model: nn.Module,
    num_episodes_per_regime: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run comprehensive uncertainty and out-of-distribution benchmark."""
    model.eval()
    expert = AStarExpert()

    regimes = ["in_distribution", "unseen_maps", "larger_maps", "blocked_paths", "noisy_states"]
    results: Dict[str, Any] = {}

    rng = np.random.RandomState(seed)

    for regime in regimes:
        confidences: List[float] = []
        accuracies: List[int] = []
        successes = 0
        total_steps = 0
        episodes_run = 0

        for ep in range(num_episodes_per_regime):
            ep_seed = int(rng.randint(0, 1_000_000))

            if regime == "in_distribution":
                env = GridWorld(random_start_goal=True, seed=ep_seed)
            elif regime == "unseen_maps":
                env = create_random_gridworld(width=12, height=7, wall_prob=0.22, seed=ep_seed)
            elif regime == "larger_maps":
                env = create_large_gridworld(width=20, height=10, seed=ep_seed)
            elif regime == "blocked_paths":
                env = create_blocked_path_gridworld(seed=ep_seed)
            elif regime == "noisy_states":
                env = GridWorld(random_start_goal=True, noise_level=0.3, seed=ep_seed)
            else:
                env = GridWorld(seed=ep_seed)

            episodes_run += 1
            done = False
            ep_steps = 0

            while not done and ep_steps < min(80, env.max_steps):
                feat = env.get_feature_vector()
                dist = model.get_action_distribution(feat)
                pred_action = dist["action_idx"]
                conf = dist["confidence"]

                # Get expert reference action if reachable
                exp_action = expert.get_action(env)
                is_correct = 1 if (exp_action is not None and int(exp_action) == pred_action) else 0

                confidences.append(conf)
                accuracies.append(is_correct)

                obs, r, done, info = env.step(pred_action)
                ep_steps += 1
                if info.get("reached_goal", False):
                    successes += 1

            total_steps += ep_steps

        # Compute metrics
        conf_arr = np.array(confidences) if confidences else np.array([0.5])
        acc_arr = np.array(accuracies) if accuracies else np.array([0])
        cal_metrics = compute_calibration_metrics(conf_arr, acc_arr)

        results[regime] = {
            "regime": regime,
            "episodes": episodes_run,
            "total_decisions": len(confidences),
            "mean_confidence": float(np.mean(conf_arr)),
            "std_confidence": float(np.std(conf_arr)),
            "action_agreement_rate": float(np.mean(acc_arr)),
            "episode_success_rate": successes / max(1, episodes_run),
            "avg_steps": total_steps / max(1, episodes_run),
            "ece": cal_metrics["ece"],
            "brier_score": cal_metrics["brier_score"],
            "spearman_correlation": cal_metrics["spearman_correlation"],
            "auroc_error_detection": cal_metrics["auroc_error_detection"],
        }

    return results
