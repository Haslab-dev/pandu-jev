"""Stress Testing Suite (Phase 8).

Evaluates policy robustness across 5 escalating environments:
- Version A: Fixed maps (Memorization vs generalization baseline)
- Version B: Random procedural maps (Novel obstacle layouts)
- Version C: Dynamic obstacles (Moving hazards / patrol bots)
- Version D: Partial observability (Fog of war / limited sensing radius)
- Version E: Noisy observations (Stochastic sensory corruption)
"""

from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

from env.gridworld import GridWorld, create_random_gridworld


def run_stress_test_suite(
    model: nn.Module,
    num_episodes_per_variant: int = 50,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Execute evaluation across Version A through Version E."""
    model.eval()
    variants = ["Version A (Fixed)", "Version B (Random)", "Version C (Dynamic)", "Version D (Partial)", "Version E (Noisy)"]
    report: Dict[str, Dict[str, Any]] = {}

    rng = np.random.RandomState(seed)

    for variant in variants:
        successes = 0
        total_steps = 0
        total_reward = 0.0
        wall_hits = 0
        confidences: List[float] = []

        for ep in range(num_episodes_per_variant):
            ep_seed = int(rng.randint(0, 1_000_000))

            if "Version A" in variant:
                env = GridWorld(random_start_goal=False, seed=ep_seed)
            elif "Version B" in variant:
                env = create_random_gridworld(width=12, height=7, wall_prob=0.18, seed=ep_seed)
            elif "Version C" in variant:
                env = GridWorld(dynamic_obstacles=True, seed=ep_seed)
            elif "Version D" in variant:
                env = GridWorld(partial_obs=True, obs_radius=3, seed=ep_seed)
            elif "Version E" in variant:
                env = GridWorld(noise_level=0.25, seed=ep_seed)
            else:
                env = GridWorld(seed=ep_seed)

            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done and ep_steps < env.max_steps:
                feat = env.get_feature_vector()
                dist = model.get_action_distribution(feat)
                action = dist["action_idx"]
                confidences.append(dist["confidence"])

                obs, r, done, info = env.step(action)
                ep_reward += r
                ep_steps += 1
                if info.get("hit_wall", False):
                    wall_hits += 1
                if info.get("reached_goal", False):
                    successes += 1

            total_steps += ep_steps
            total_reward += ep_reward

        report[variant] = {
            "variant": variant,
            "episodes": num_episodes_per_variant,
            "success_rate": successes / max(1, num_episodes_per_variant),
            "avg_steps": total_steps / max(1, num_episodes_per_variant),
            "avg_reward": total_reward / max(1, num_episodes_per_variant),
            "avg_wall_hits": wall_hits / max(1, num_episodes_per_variant),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        }

    return report
