"""Policy network architectures for Mini-Jev.

Implements:
- TinyPolicy: The canonical <1M parameter MLP architecture from Phase 2
- ScalablePolicy: Parameter-configurable MLP for Phase 10 scaling studies (1K, 50K, 1M, 5M)
- Temperature scaling calibration layer
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class TinyPolicy(nn.Module):
    """The canonical Phase 2 Tiny Policy:

    Input (16) -> Linear(32) -> GELU -> Linear(64) -> GELU -> Linear(4)
    Total parameters: ~2,916 params (< 0.003M).
    """

    def __init__(self, input_dim: int = 16, num_actions: int = 4, hidden_dims: Tuple[int, int] = (32, 64)):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], num_actions)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        logits = self.out(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_action_distribution(
        self,
        features: Union[torch.Tensor, List[float], Any],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute action probabilities, selected action, and calibrated confidence."""
        self.eval()
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):
                t_feat = torch.tensor(features, dtype=torch.float32)
            else:
                t_feat = features.float()

            if t_feat.dim() == 1:
                t_feat = t_feat.unsqueeze(0)

            logits = self.forward(t_feat)
            t_val = temperature if temperature is not None else self.temperature.item()
            scaled_logits = logits / max(1e-4, t_val)
            probs = F.softmax(scaled_logits, dim=-1)[0].cpu().numpy()

            best_idx = int(probs.argmax())
            confidence = float(probs[best_idx])
            entropy = float(-sum(p * math.log(max(1e-12, p)) for p in probs))

            prob_dict = {ACTION_NAMES[i]: float(probs[i]) for i in range(self.num_actions)}

            return {
                "action": ACTION_NAMES[best_idx],
                "action_idx": best_idx,
                "probabilities": prob_dict,
                "confidence": confidence,
                "entropy": entropy,
                "logits": logits[0].cpu().numpy().tolist(),
            }


class ScalablePolicy(nn.Module):
    """Deep & wide MLP for parameter scaling benchmarks (Phase 10)."""

    def __init__(
        self,
        input_dim: int = 16,
        num_actions: int = 4,
        hidden_layers: List[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 64]

        self.input_dim = input_dim
        self.num_actions = num_actions

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation.lower() == "gelu":
                layers.append(nn.GELU())
            elif activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "silu":
                layers.append(nn.SiLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_actions))
        self.net = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_action_distribution(
        self,
        features: Union[torch.Tensor, List[float], Any],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):
                t_feat = torch.tensor(features, dtype=torch.float32)
            else:
                t_feat = features.float()

            if t_feat.dim() == 1:
                t_feat = t_feat.unsqueeze(0)

            logits = self.forward(t_feat)
            t_val = temperature if temperature is not None else self.temperature.item()
            scaled_logits = logits / max(1e-4, t_val)
            probs = F.softmax(scaled_logits, dim=-1)[0].cpu().numpy()

            best_idx = int(probs.argmax())
            confidence = float(probs[best_idx])
            entropy = float(-sum(p * math.log(max(1e-12, p)) for p in probs))
            prob_dict = {ACTION_NAMES[i]: float(probs[i]) for i in range(min(len(probs), len(ACTION_NAMES)))}

            return {
                "action": ACTION_NAMES[best_idx] if best_idx < len(ACTION_NAMES) else str(best_idx),
                "action_idx": best_idx,
                "probabilities": prob_dict,
                "confidence": confidence,
                "entropy": entropy,
                "logits": logits[0].cpu().numpy().tolist(),
            }


def build_scaled_model(size_category: str, input_dim: int = 16, num_actions: int = 4) -> ScalablePolicy:
    """Factory creating policies for specific parameter size regimes (Phase 10):

    - 'tiny': ~3K params (Phase 2 canonical)
    - 'small': ~50K params
    - '1M': ~1M params
    - '5M': ~5M params
    """
    cat = size_category.lower()
    if cat in ("tiny", "3k"):
        return TinyPolicy(input_dim=input_dim, num_actions=num_actions)
    elif cat in ("small", "50k"):
        return ScalablePolicy(input_dim, num_actions, [128, 256, 128])
    elif cat in ("1m", "policy-1m"):
        return ScalablePolicy(input_dim, num_actions, [512, 1024, 768, 512])
    elif cat in ("5m", "policy-5m"):
        return ScalablePolicy(input_dim, num_actions, [1024, 2048, 1536, 1024])
    else:
        raise ValueError(f"Unknown size category: {size_category}")
