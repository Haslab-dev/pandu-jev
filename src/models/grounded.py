"""Grounded Language Policy, Tiny Semantic Adapters, and Canonical Intent Protocol for Pandu.

Decouples semantic comprehension (handled by frozen foundation LMs or structured
symbolic schemas) from high-speed motor execution (handled by Pandu).

Architecture:
    Natural Language
           │
           ▼
    Frozen Foundation LM
           │
           ▼
    Canonical Intent Protocol (Symbolic or 16d Latent Adapter)
           │
           ▼
    GroundedPanduPolicy (~4,980 parameters) ◄── Environment Observation (16d)
           │
           ▼
    Real-Time Action [UP, DOWN, LEFT, RIGHT] (<0.03 ms)
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLanguageAdapter(nn.Module):
    """Compresses high-dimensional foundation model representations into compact latent states."""

    def __init__(self, lm_dim: int, latent_dim: int = 16, hidden_dim: Optional[int] = None):
        super().__init__()
        self.lm_dim = lm_dim
        self.latent_dim = latent_dim

        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(lm_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            # Direct linear projection with layer norm
            self.net = nn.Sequential(
                nn.Linear(lm_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GroundedPanduPolicy(nn.Module):
    """Tiny ~4.9K parameter decision policy grounded in environment and language latent state.

    Input: [env_state (16d) || z_language (16d)] -> 32d
    Architecture:
        Linear(32 -> 48) -> GELU -> Linear(48 -> 64) -> GELU -> Linear(64 -> 4)
    Total Parameters: 4,980
    """

    def __init__(self, env_dim: int = 16, latent_lang_dim: int = 16, num_actions: int = 4):
        super().__init__()
        self.env_dim = env_dim
        self.latent_lang_dim = latent_lang_dim
        self.num_actions = num_actions
        total_in = env_dim + latent_lang_dim

        self.net = nn.Sequential(
            nn.Linear(total_in, 48),
            nn.GELU(),
            nn.Linear(48, 64),
            nn.GELU(),
            nn.Linear(64, num_actions),
        )
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    def forward(self, env_state: torch.Tensor, z_lang: torch.Tensor) -> torch.Tensor:
        """Forward pass combining spatial state and language latent vector."""
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)
        if z_lang.dim() == 1:
            z_lang = z_lang.unsqueeze(0)

        # Broadcast z_lang if batch dimensions differ
        if z_lang.shape[0] == 1 and env_state.shape[0] > 1:
            z_lang = z_lang.expand(env_state.shape[0], -1)

        x = torch.cat([env_state, z_lang], dim=-1)
        return self.net(x)

    def get_action_distribution(
        self, env_state: torch.Tensor, z_lang: torch.Tensor, temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Compute calibrated probability distribution over action candidates."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(env_state, z_lang)[0] / max(1e-5, temperature)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])

        return {
            "action": self.action_names[action_idx],
            "action_idx": action_idx,
            "confidence": confidence,
            "probabilities": {self.action_names[i]: float(probs[i]) for i in range(self.num_actions)},
            "logits": logits.cpu().numpy(),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Canonical Intent Protocol: The Universal Semantic Boundary Specification
# ==============================================================================

@dataclass
class GoalSpec:
    type: str = "reach"                           # "reach", "evade", "explore", "backtrack"
    target: str = "goal"                          # "goal", "waypoint", "safety_zone"
    direction: Tuple[float, float] = (1.0, 0.0)   # Continuous 2D unit vector [dx, dy]


@dataclass
class ConstraintSpec:
    avoid_obstacles: bool = True
    speed_limit: float = 1.0


@dataclass
class PreferenceSpec:
    risk: float = 0.2                             # [0.0, 1.0]
    urgency: float = 0.7                          # [0.0, 1.0]


@dataclass
class CanonicalIntentProtocol:
    """Canonical Semantic Boundary Protocol between high-level cognition and fast motor control.

    Serves as an invariant schema: foundation models (or humans) output this
    schema, and Pandu consumes it without knowing the upstream source.
    """
    goal: GoalSpec = field(default_factory=GoalSpec)
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    preferences: PreferenceSpec = field(default_factory=PreferenceSpec)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalIntentProtocol":
        goal_data = data.get("goal", {})
        dir_val = tuple(goal_data.get("direction", (1.0, 0.0)))
        return cls(
            goal=GoalSpec(
                type=goal_data.get("type", "reach"),
                target=goal_data.get("target", "goal"),
                direction=(float(dir_val[0]), float(dir_val[1])),
            ),
            constraints=ConstraintSpec(
                avoid_obstacles=data.get("constraints", {}).get("avoid_obstacles", True),
                speed_limit=float(data.get("constraints", {}).get("speed_limit", 1.0)),
            ),
            preferences=PreferenceSpec(
                risk=float(data.get("preferences", {}).get("risk", 0.2)),
                urgency=float(data.get("preferences", {}).get("urgency", 0.7)),
            ),
        )

    def encode_to_vector(self, target_dim: int = 16) -> torch.Tensor:
        """Deterministically project canonical schema into a normalized 16-dimensional tensor."""
        vec = np.zeros(target_dim, dtype=np.float32)

        # 0..3: Goal Type Categorical
        type_map = {"reach": 0, "evade": 1, "explore": 2, "backtrack": 3}
        vec[type_map.get(self.goal.type, 0)] = 1.0

        # 4..5: Continuous Direction 2D Vector (dx, dy)
        dx, dy = self.goal.direction
        norm_dir = math.hypot(dx, dy)
        if norm_dir > 1e-5:
            vec[4] = dx / norm_dir
            vec[5] = dy / norm_dir

        # 6..9: Cardinal Direction Projections
        vec[6] = max(0.0, float(vec[4]))   # East
        vec[7] = max(0.0, -float(vec[4]))  # West
        vec[8] = max(0.0, -float(vec[5]))  # North (up is negative dy)
        vec[9] = max(0.0, float(vec[5]))   # South (down is positive dy)

        # 10..12: Constraints & Preferences
        vec[10] = 1.0 if self.constraints.avoid_obstacles else 0.0
        vec[11] = float(np.clip(self.preferences.urgency, 0.0, 1.0))
        vec[12] = float(np.clip(self.preferences.risk, 0.0, 1.0))

        # 13: Speed limit
        vec[13] = float(np.clip(self.constraints.speed_limit, 0.0, 1.0))

        # Normalization across active channels
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return torch.from_numpy(vec)


class StructuredIntent(CanonicalIntentProtocol):
    """Backwards-compatible convenience wrapper."""

    def __init__(
        self,
        objective: str = "reach_goal",
        direction: str = "east",
        avoid_obstacle: bool = True,
        urgency: float = 0.5,
        exploration: float = 0.1,
    ):
        dir_vector_map = {
            "east": (1.0, 0.0),
            "west": (-1.0, 0.0),
            "north": (0.0, -1.0),
            "south": (0.0, 1.0),
        }
        type_clean = "reach" if "reach" in objective else ("evade" if "evade" in objective else "explore")
        super().__init__(
            goal=GoalSpec(type=type_clean, target="goal", direction=dir_vector_map.get(direction, (1.0, 0.0))),
            constraints=ConstraintSpec(avoid_obstacles=avoid_obstacle),
            preferences=PreferenceSpec(risk=exploration, urgency=urgency),
        )
        self.objective = objective
        self.direction = direction
        self.avoid_obstacle = avoid_obstacle
        self.urgency = urgency
        self.exploration = exploration
