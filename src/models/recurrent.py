"""Recurrent Policy architecture for Pandu (pandu-jev).

Provides:
- RecurrentPanduPolicy: A compact GRU-based policy (~7K parameters) that
  maintains internal temporal memory h_t in R^32. Enables robust decision-making
  under Partial Observability (Fog-of-War / POMDP) where stateless MLPs fail.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class RecurrentPanduPolicy(nn.Module):
    """Compact Recurrent Policy with Gated Recurrent Unit (GRU) memory.

    Architecture:
    - Input projection: Linear(input_dim, hidden_dim) + GELU
    - Recurrent core: GRUCell(hidden_dim, hidden_dim)  [h_t in R^hidden_dim]
    - Action head: Linear(hidden_dim, num_actions)
    Total parameters: ~7,012 params (< 0.008M) for input_dim=16, hidden_dim=32.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 32,
        num_actions: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, num_actions)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initialize zero hidden state."""
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=dev)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step-by-step forward pass: x (B, input_dim), h (B, hidden_dim)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        if h is None:
            h = self.init_hidden(batch_size=batch_size, device=x.device)

        e = F.gelu(self.in_proj(x))
        h_next = self.gru(e, h)
        logits = self.out_head(h_next)
        return logits, h_next

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence forward pass for BPTT training: x_seq is (B, T, input_dim)."""
        batch_size, seq_len, _ = x_seq.shape
        h = h_init if h_init is not None else self.init_hidden(batch_size=batch_size, device=x_seq.device)

        logits_list = []
        for t in range(seq_len):
            xt = x_seq[:, t, :]
            logits, h = self.forward(xt, h)
            logits_list.append(logits.unsqueeze(1))

        all_logits = torch.cat(logits_list, dim=1)  # (B, T, num_actions)
        return all_logits, h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_action_distribution(
        self,
        features: Union[torch.Tensor, List[float], Any],
        h: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute action probabilities, selected action, and calibrated confidence with memory."""
        self.eval()
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):
                t_feat = torch.tensor(features, dtype=torch.float32)
            else:
                t_feat = features.float()

            if t_feat.dim() == 1:
                t_feat = t_feat.unsqueeze(0)

            device = next(self.parameters()).device
            t_feat = t_feat.to(device)
            if h is not None:
                h = h.to(device)

            logits, h_next = self.forward(t_feat, h)

            temp = temperature if temperature is not None else float(self.temperature.item())
            scaled_logits = logits / max(temp, 1e-4)
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0)

            confidence, action_idx = torch.max(probs, dim=-1)
            prob_list = probs.cpu().tolist()

            eps = 1e-8
            entropy = -float(torch.sum(probs * torch.log(probs + eps)).item())

            return {
                "action_idx": int(action_idx.item()),
                "action_name": ACTION_NAMES[int(action_idx.item())],
                "action_probs": prob_list,
                "confidence": float(confidence.item()),
                "entropy": entropy,
                "hidden": h_next,
            }
