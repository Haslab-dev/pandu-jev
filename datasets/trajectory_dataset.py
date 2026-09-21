"""Trajectory generation and dataset structures for imitation learning."""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from env.gridworld import GridWorld, Action, create_random_gridworld
from expert.astar import AStarExpert


def collect_expert_trajectories(
    num_episodes: int = 1000,
    include_random_maps: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Collect (feature, action) pairs using A* expert across default and random maps."""
    expert = AStarExpert()
    features_list: List[np.ndarray] = []
    actions_list: List[int] = []

    successful_episodes = 0
    total_steps = 0
    total_reward = 0.0

    rng = np.random.RandomState(seed)

    for ep in range(num_episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        # 50% default map (with random start/goal), 50% random maps if requested
        if include_random_maps and (ep % 2 == 1):
            env = create_random_gridworld(width=12, height=7, wall_prob=0.18, seed=ep_seed)
        else:
            env = GridWorld(random_start_goal=True, seed=ep_seed)

        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done and ep_steps < env.max_steps:
            action = expert.get_action(env)
            if action is None:
                # No path found or already at goal
                break

            feat = env.get_feature_vector()
            features_list.append(feat)
            actions_list.append(int(action))

            obs, r, done, info = env.step(action)
            ep_reward += r
            ep_steps += 1
            if info.get("reached_goal", False):
                successful_episodes += 1

        total_steps += ep_steps
        total_reward += ep_reward

    X = np.array(features_list, dtype=np.float32)
    y = np.array(actions_list, dtype=np.int64)

    stats = {
        "num_episodes": num_episodes,
        "total_samples": len(y),
        "success_rate": successful_episodes / max(1, num_episodes),
        "avg_steps_per_episode": total_steps / max(1, num_episodes),
        "avg_reward": total_reward / max(1, num_episodes),
    }

    return X, y, stats


def save_dataset(filepath: str, X: np.ndarray, y: np.ndarray, stats: Optional[Dict[str, float]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(filepath, features=X, labels=y, stats=stats or {})


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(filepath, allow_pickle=True)
    stats = {}
    if "stats" in data:
        stats = data["stats"].item() if hasattr(data["stats"], "item") else dict(data["stats"])
    return data["features"], data["labels"], stats
