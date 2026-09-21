"""Grounded Language Policy and Tiny Semantic Adapters for Pandu (pandu-jev).

Decouples semantic comprehension (handled by frozen foundation LMs or structured
symbolic schemas) from high-speed motor execution (handled by Pandu).

Architecture:
    Frozen LM Embedding (768d / 576d / 1024d)
               │
               ▼
    TinyLanguageAdapter (Linear compression) ──► z_lang (16d)
                                                    │
    Environment State (16d) ────────────────────────┤
                                                    ▼
                                          GroundedPanduPolicy (4,980 params)
                                                    │
                                                    ▼
                                           Action Probabilities [UP, DOWN, LEFT, RIGHT]
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
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
        """Forward pass combining spatial state and language latent vector.

        Args:
            env_state: (batch_size, env_dim) or (env_dim,)
            z_lang: (batch_size, latent_lang_dim) or (latent_lang_dim,)

        Returns:
            logits: (batch_size, num_actions)
        """
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


@dataclass
class StructuredIntent:
    """Symbolic semantic intent extracted by language model cortex."""

    objective: str = "reach_goal"  # "reach_goal", "evade_hazard", "explore", "backtrack"
    direction: str = "east"        # "north", "south", "east", "west", "none"
    avoid_obstacle: bool = True
    urgency: float = 0.5           # [0.0, 1.0]
    exploration: float = 0.1       # [0.0, 1.0]

    def encode_to_vector(self, target_dim: int = 16) -> torch.Tensor:
        """Deterministically encode structured schema into a 16-dimensional continuous tensor."""
        vec = np.zeros(target_dim, dtype=np.float32)

        # Objective categorical (0..3)
        obj_map = {"reach_goal": 0, "evade_hazard": 1, "explore": 2, "backtrack": 3}
        idx = obj_map.get(self.objective, 0)
        vec[idx] = 1.0

        # Direction one-hot (4..7)
        dir_map = {"north": 4, "south": 5, "west": 6, "east": 7}
        if self.direction in dir_map:
            vec[dir_map[self.direction]] = 1.0

        # Direction continuous compass sin/cos (8..9)
        angle_map = {"north": -math.pi / 2, "south": math.pi / 2, "west": math.pi, "east": 0.0}
        angle = angle_map.get(self.direction, 0.0)
        vec[8] = math.cos(angle)
        vec[9] = math.sin(angle)

        # Behavioral scalars (10..12)
        vec[10] = 1.0 if self.avoid_obstacle else 0.0
        vec[11] = float(np.clip(self.urgency, 0.0, 1.0))
        vec[12] = float(np.clip(self.exploration, 0.0, 1.0))

        # Normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return torch.from_numpy(vec)
