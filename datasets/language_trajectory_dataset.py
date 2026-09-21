"""Language-Conditioned Trajectory Dataset Generator for Pandu.

Generates tuples of:
    (environment_features, natural_language_instruction, structured_intent, expert_action)
with synthetic linguistic variations and out-of-distribution paraphrases.
"""

import random
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from env.gridworld import GridWorld, Action, create_random_gridworld
from expert.astar import AStarExpert
from models.grounded_policy import StructuredIntent

IN_DISTRIBUTION_TEMPLATES = {
    "east": [
        "Advance eastward directly towards the target extraction point.",
        "Proceed east along the open corridor to reach the goal.",
        "Move east towards the objective while maintaining clearance.",
    ],
    "west": [
        "Head westward away from the hazard toward the extraction zone.",
        "Advance west down the main pathway to the destination.",
        "Navigate westward toward the designated goal coordinate.",
    ],
    "north": [
        "Climb northward through the vertical channel to reach the beacon.",
        "Advance north along the corridor toward the goal position.",
        "Head north avoiding the surrounding boundary walls.",
    ],
    "south": [
        "Descend southward along the pathway to the target point.",
        "Move south towards the lower goal coordinates.",
        "Advance south through the open path while dodging hazards.",
    ],
}

OOD_PARAPHRASES = {
    "east": [
        "Head right now!",
        "Go east asap.",
        "Navigate rightwards to the objective.",
        "Rush toward the eastern quadrant.",
    ],
    "west": [
        "Turn left towards safety.",
        "Sprint west immediately.",
        "Head leftwards across the room.",
        "Backtrack west to the target point.",
    ],
    "north": [
        "Go up through the gap.",
        "Ascend vertically right away.",
        "Shift upward to reach the goal.",
        "Head up towards the objective point.",
    ],
    "south": [
        "Go down through the lower passage.",
        "Descend straight downwards.",
        "Drop down towards the target mark.",
        "Move downward immediately.",
    ],
}


def get_dominant_direction(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    else:
        return "south" if dy > 0 else "north"


def collect_language_grounded_trajectories(
    num_episodes: int = 200, seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect closed-loop expert episodes paired with natural language instructions."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    expert = AStarExpert()
    records = []
    successes = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep * 13
        env = create_random_gridworld(
            width=12, height=8, wall_prob=0.18, seed=ep_seed
        )

        dx = env.goal_x - env.agent_x
        dy = env.goal_y - env.agent_y
        direction = get_dominant_direction(dx, dy)

        # Sample language templates
        prompt_in_dist = rng.choice(IN_DISTRIBUTION_TEMPLATES[direction])
        prompt_ood = rng.choice(OOD_PARAPHRASES[direction])

        intent = StructuredIntent(
            objective="reach_goal",
            direction=direction,
            avoid_obstacle=True,
            urgency=float(np_rng.uniform(0.5, 0.9)),
            exploration=0.1,
        )

        ep_steps = 0
        done = False
        reached_goal = False

        while not done and ep_steps < 50:
            action = expert.get_action(env)
            if action is None:
                break

            obs_feat = env.get_feature_vector()
            records.append({
                "env_features": obs_feat,
                "action": action.value,
                "direction": direction,
                "instruction": prompt_in_dist,
                "ood_instruction": prompt_ood,
                "intent": intent,
                "intent_vector": intent.encode_to_vector(16).numpy(),
            })

            _, r, done, info = env.step(action)
            ep_steps += 1
            if info.get("reached_goal"):
                reached_goal = True

        if reached_goal:
            successes += 1

    stats = {
        "episodes": num_episodes,
        "success_rate": successes / max(1, num_episodes),
        "total_transitions": len(records),
    }
    return records, stats


class GroundedLanguageDataset(Dataset):
    """PyTorch Dataset for language-grounded policy training."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.env_feats = torch.tensor(
            np.array([r["env_features"] for r in records], dtype=np.float32)
        )
        self.intent_vectors = torch.tensor(
            np.array([r["intent_vector"] for r in records], dtype=np.float32)
        )
        self.actions = torch.tensor(
            np.array([r["action"] for r in records], dtype=np.int64)
        )
        self.instructions = [r["instruction"] for r in records]
        self.ood_instructions = [r["ood_instruction"] for r in records]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "env_features": self.env_feats[idx],
            "intent_vector": self.intent_vectors[idx],
            "action": self.actions[idx],
            "instruction": self.instructions[idx],
            "ood_instruction": self.ood_instructions[idx],
        }
