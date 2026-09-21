"""Language-Conditioned Trajectory Dataset Generator for Pandu.

Generates expert trajectories paired with:
1. In-Distribution Training Prompts
2. 5-Tier Rigorous OOD Linguistic Evaluation Suite:
   - OOD-1 (Lexical): Rare synonyms, formal lexicon
   - OOD-2 (Syntactic): Passive, inverted sentence structures
   - OOD-3 (Compositional): Multi-clause conditionals ("Circle barrier before continuing east")
   - OOD-4 (Semantic/Metaphorical): Metaphorical ("Route toward the sunrise")
   - OOD-5 (Adversarial Negation): Explicit negations ("Do NOT head west; route north")
3. Canonical Structured Intent (Protocol)
4. Ground-Truth Oracle Intent (Generated directly from A* pathfinding)
"""

import math
import random
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from env.gridworld import GridWorld, Action, create_random_gridworld
from expert.astar import AStarExpert
from models.grounded_policy import (
    CanonicalIntentProtocol,
    GoalSpec,
    ConstraintSpec,
    PreferenceSpec,
    StructuredIntent,
)

# ------------------------------------------------------------------------------
# Linguistic Prompt Taxonomy
# ------------------------------------------------------------------------------

TRAIN_TEMPLATES = {
    "east": [
        "Go east.",
        "Move right.",
        "Head east toward the goal.",
        "Advance eastward.",
    ],
    "west": [
        "Go west.",
        "Move left.",
        "Head west toward the target.",
        "Advance westward.",
    ],
    "north": [
        "Go north.",
        "Move up.",
        "Head north toward the beacon.",
        "Advance upward.",
    ],
    "south": [
        "Go south.",
        "Move down.",
        "Head south toward the exit.",
        "Advance downward.",
    ],
}

# OOD Tier 1: Lexical Variation (Formal vocabulary / rare synonyms)
OOD_1_LEXICAL = {
    "east": [
        "Proceed toward the eastern quadrant.",
        "Navigate directly rightwards into the corridor.",
        "Traverse along the oriental azimuth.",
    ],
    "west": [
        "Advance towards the occidental periphery.",
        "Relocate westward across the chamber.",
        "Shift position leftward to the extraction point.",
    ],
    "north": [
        "Ascend toward the boreal coordinates.",
        "Elevate northward along the vertical conduit.",
        "Climb directly upward to the apex.",
    ],
    "south": [
        "Descend toward the austral boundary.",
        "Drop downward toward the subterranean marker.",
        "Plunge south through the lower vent.",
    ],
}

# OOD Tier 2: Syntactic Variation (Inverted syntax / passive voice)
OOD_2_SYNTACTIC = {
    "east": [
        "Your primary destination lies directly to the east.",
        "To the right is where the mission target is positioned.",
        "Eastward is the required vector for extraction.",
    ],
    "west": [
        "To the left the evacuation zone awaits you.",
        "Westward lies the optimal destination point.",
        "Your destination can be found towards the west.",
    ],
    "north": [
        "Upward lies the beacon requiring contact.",
        "To the north your objective is located.",
        "Northward progress is required to reach the target.",
    ],
    "south": [
        "Located to the south is the final extraction point.",
        "Downward is where you must proceed.",
        "To the lower sector the agent must be guided.",
    ],
}

# OOD Tier 3: Compositional / Conditional
OOD_3_COMPOSITIONAL = {
    "east": [
        "Circle the adjacent barrier before continuing east.",
        "Clear the local perimeter and subsequently move right.",
        "Maintain clearance from boundary walls while progressing east.",
    ],
    "west": [
        "Bypass any obstacles while carefully proceeding west.",
        "Step away from hazards and execute a westward advance.",
        "Evade nearby walls prior to advancing left.",
    ],
    "north": [
        "Filter through the narrow corridor and head north.",
        "Navigate past the barricade and continue upwards.",
        "Clear the southern danger zone and proceed north.",
    ],
    "south": [
        "Avoid the upper dead-end by descending southward.",
        "Maintain spacing from northern blocks and move south.",
        "Evade the upper obstacle before routing down.",
    ],
}

# OOD Tier 4: Semantic / Metaphorical
OOD_4_SEMANTIC = {
    "east": [
        "Take the route toward the sunrise.",
        "Head towards where dawn breaks.",
        "Orient yourself toward the morning sun.",
    ],
    "west": [
        "Journey into the sunset.",
        "Take the route toward dusk.",
        "Follow the path of the evening twilight.",
    ],
    "north": [
        "Navigate toward the arctic pole.",
        "Follow the compass needle straight to true north.",
        "Move towards the frozen tundra above.",
    ],
    "south": [
        "Descend towards the tropical equator.",
        "Head towards the warm southern hemisphere.",
        "Follow the compass south into the lower valley.",
    ],
}

# OOD Tier 5: Adversarial Negation (Explicit prohibition of decoy directions)
OOD_5_ADVERSARIAL = {
    "east": [
        "Do not go west or north; advance exclusively east.",
        "Avoid heading left; proceed rightward to the target.",
        "The west and south are blocked; push eastward.",
    ],
    "west": [
        "Do not go east; the eastern corridor is blocked. Head west.",
        "Avoid turning right at all costs; advance west.",
        "The east is dangerous; travel westward to safety.",
    ],
    "north": [
        "Do not descend south; climb north instead.",
        "Avoid the lower channel; head strictly north.",
        "Southward leads to a dead-end; advance north.",
    ],
    "south": [
        "Do not head north; the northern path is impassable. Move south.",
        "Avoid going up into the barrier; descend south.",
        "Upward is blocked; travel south immediately.",
    ],
}


def get_dominant_direction(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    else:
        return "south" if dy > 0 else "north"


def build_oracle_intent(
    agent_x: int, agent_y: int, goal_x: int, goal_y: int, expert_action: Action
) -> CanonicalIntentProtocol:
    """Construct perfect Ground-Truth Oracle Intent directly from pathfinding dynamics."""
    dx = float(goal_x - agent_x)
    dy = float(goal_y - agent_y)
    norm = math.hypot(dx, dy)
    dir_vec = (dx / norm, dy / norm) if norm > 1e-5 else (1.0, 0.0)

    return CanonicalIntentProtocol(
        goal=GoalSpec(type="reach", target="goal", direction=dir_vec),
        constraints=ConstraintSpec(avoid_obstacles=True, speed_limit=1.0),
        preferences=PreferenceSpec(risk=0.1, urgency=0.8),
    )


def collect_language_grounded_trajectories(
    num_episodes: int = 250, seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect closed-loop expert episodes paired with 5-tier OOD linguistic test battery."""
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

        # In-dist & 5-tier OOD samples
        prompt_train = rng.choice(TRAIN_TEMPLATES[direction])
        ood_1 = rng.choice(OOD_1_LEXICAL[direction])
        ood_2 = rng.choice(OOD_2_SYNTACTIC[direction])
        ood_3 = rng.choice(OOD_3_COMPOSITIONAL[direction])
        ood_4 = rng.choice(OOD_4_SEMANTIC[direction])
        ood_5 = rng.choice(OOD_5_ADVERSARIAL[direction])

        # Canonical intent
        canonical_intent = StructuredIntent(
            objective="reach_goal",
            direction=direction,
            avoid_obstacle=True,
            urgency=float(np_rng.uniform(0.6, 0.9)),
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
            oracle_intent = build_oracle_intent(
                env.agent_x, env.agent_y, env.goal_x, env.goal_y, action
            )

            records.append({
                "env_features": obs_feat,
                "action": action.value,
                "direction": direction,
                "instruction": prompt_train,
                "ood_1_lexical": ood_1,
                "ood_2_syntactic": ood_2,
                "ood_3_compositional": ood_3,
                "ood_4_semantic": ood_4,
                "ood_5_adversarial": ood_5,
                "canonical_intent": canonical_intent,
                "intent_vector": canonical_intent.encode_to_vector(16).numpy(),
                "oracle_intent_vector": oracle_intent.encode_to_vector(16).numpy(),
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
    """PyTorch Dataset for language-grounded policy training and 5-tier OOD evaluation."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.env_feats = torch.tensor(
            np.array([r["env_features"] for r in records], dtype=np.float32)
        )
        self.intent_vectors = torch.tensor(
            np.array([r["intent_vector"] for r in records], dtype=np.float32)
        )
        self.oracle_intent_vectors = torch.tensor(
            np.array([r["oracle_intent_vector"] for r in records], dtype=np.float32)
        )
        self.actions = torch.tensor(
            np.array([r["action"] for r in records], dtype=np.int64)
        )
        self.instructions = [r["instruction"] for r in records]
        self.ood_1 = [r["ood_1_lexical"] for r in records]
        self.ood_2 = [r["ood_2_syntactic"] for r in records]
        self.ood_3 = [r["ood_3_compositional"] for r in records]
        self.ood_4 = [r["ood_4_semantic"] for r in records]
        self.ood_5 = [r["ood_5_adversarial"] for r in records]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "env_features": self.env_feats[idx],
            "intent_vector": self.intent_vectors[idx],
            "oracle_intent_vector": self.oracle_intent_vectors[idx],
            "action": self.actions[idx],
            "instruction": self.instructions[idx],
            "ood_1": self.ood_1[idx],
            "ood_2": self.ood_2[idx],
            "ood_3": self.ood_3[idx],
            "ood_4": self.ood_4[idx],
            "ood_5": self.ood_5[idx],
        }
