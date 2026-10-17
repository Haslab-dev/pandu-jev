"""Pandu-Micro: MicroBERT Architecture for Long-Running Software Agents (~1.2MB INT8 / 2.4MB FP16).

Features:
- Pruned 2,048-token domain vocabulary (eliminates 20MB vocabulary tax)
- 3-Layer Bidirectional Transformer Encoder (d=192, 4 heads, intermediate=512)
- Total Parameters: ~1,280,000 weights (~1.2 MB INT8 / 2.46 MB FP16)
- Salient Multi-Turn Sequence Formatting for 100-200+ Turn Sessions:
  [cls] [intent] {intent} [sep] [fail] {fails} failures last tool {tool} [sep] [turn] turn {t} of {total} [sep] [goal] {goal} [sep]
- 6 Typed Multi-Task Decision Heads (Zero Generated Tokens):
  1. Tool Category (9 classes: explore, read, modify, execute, git, web, skills, mcp, none)
  2. Workflow Phase (6 phases: discovery, planning, implementation, debugging, verification, wrapup)
  3. Loop / Stuck Alarm (Continuous probability [0, 1] + calibrated alarm threshold)
  4. Subagent Delegation (Continuous probability [0, 1] - Protects context in long sessions)
  5. Session Completion (Continuous probability [0, 1] - is_done)
  6. Safety Risk Level (3 classes: low, medium, high)
- Latency: ~0.8 ms on Apple Silicon MPS / CPU (~160 ms total across an entire 200-turn run!)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.rakitkode_domain import CATEGORIES, RISK_LEVELS
from datasets.long_running_trajectories import PHASES
from micro.tokenizer import PanduMicroTokenizer


@dataclass
class PanduMicroConfig:
    """Configuration for Pandu-Micro (MicroBERT Architecture)."""
    vocab_size: int = 2048
    hidden_size: int = 192
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 512
    dropout: float = 0.1
    num_categories: int = len(CATEGORIES)  # 9
    num_phases: int = len(PHASES)          # 6
    num_risks: int = len(RISK_LEVELS)      # 3


def format_turn_sequence(
    goal: str,
    turn_index: int,
    total_turns: int,
    consecutive_failures: int,
    last_tool: str,
    intent: str,
) -> str:
    """Format input string for long-running agent turn with salient intent and session markers."""
    return (
        f"[intent] {intent.strip()} [sep] "
        f"[fail] {consecutive_failures} failures last tool {last_tool} [sep] "
        f"[turn] turn {turn_index} of {total_turns} [sep] "
        f"[goal] {goal.strip()}"
    )


class PanduMicro(nn.Module):
    """Pandu-Micro: ~1.28M Parameter Transformer for Long-Running Agents."""

    def __init__(self, config: Optional[PanduMicroConfig] = None):
        super().__init__()
        self.config = config or PanduMicroConfig()
        d = self.config.hidden_size

        # 1. Embeddings
        self.token_embeddings = nn.Embedding(self.config.vocab_size, d, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, d)
        self.emb_norm = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(self.config.dropout)

        # 2. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.intermediate_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_hidden_layers)

        # 3. Decision Pooling Head (Mean pooling + CLS representation)
        self.pooler = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
        )

        # 4. Multi-Task Typed Decision Heads
        self.cat_head = nn.Linear(d, self.config.num_categories)
        self.phase_head = nn.Linear(d, self.config.num_phases)
        self.stuck_head = nn.Linear(d, 1)
        self.delegate_head = nn.Linear(d, 1)
        self.done_head = nn.Linear(d, 1)
        self.risk_head = nn.Linear(d, self.config.num_risks)

        # Temperature parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters (~1.28M)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def fp16_size_mb(self) -> float:
        """Disk/RAM footprint in FP16 precision."""
        return sum(p.numel() * 2 for p in self.parameters()) / (1024 * 1024)

    @property
    def int8_size_mb(self) -> float:
        """Disk/RAM footprint in INT8 precision."""
        return sum(p.numel() * 1 for p in self.parameters()) / (1024 * 1024)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.emb_drop(self.emb_norm(x))

        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)

        h = self.encoder(x, src_key_padding_mask=pad_mask)

        # Rich pooled representation: Masked mean pooling + CLS token
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            sum_h = torch.sum(h * mask_exp, dim=1)
            cnt = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            mean_pooled = sum_h / cnt
            cls_rep = self.pooler(mean_pooled + h[:, 0])
        else:
            cls_rep = self.pooler(h.mean(dim=1) + h[:, 0])

        logits_cat = self.cat_head(cls_rep)
        logits_phase = self.phase_head(cls_rep)
        z_stuck = self.stuck_head(cls_rep).squeeze(-1)
        z_delegate = self.delegate_head(cls_rep).squeeze(-1)
        z_done = self.done_head(cls_rep).squeeze(-1)
        logits_risk = self.risk_head(cls_rep)

        return logits_cat, logits_phase, z_stuck, z_delegate, z_done, logits_risk

    @torch.no_grad()
    def predict_session_turn(
        self,
        goal: str,
        turn_index: int,
        total_turns: int,
        consecutive_failures: int,
        last_tool: str,
        intent: str,
        tokenizer: PanduMicroTokenizer,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Sub-millisecond turn prediction during a long-running agent session."""
        self.eval()
        raw_seq = format_turn_sequence(goal, turn_index, total_turns, consecutive_failures, last_tool, intent)
        token_ids = tokenizer.encode(raw_seq, max_length=384)

        ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
        mask_t = torch.ones_like(ids_t)

        z_cat, z_phase, z_stuck, z_del, z_done, z_risk = self.forward(ids_t, mask_t)

        p_cat = F.softmax(z_cat[0] / self.temperature, dim=-1).cpu().numpy()
        top_cat_idx = int(p_cat.argmax())

        p_phase = F.softmax(z_phase[0], dim=-1).cpu().numpy()
        top_phase_idx = int(p_phase.argmax())

        p_stuck = float(torch.sigmoid(z_stuck[0]).cpu().item())
        p_del = float(torch.sigmoid(z_del[0]).cpu().item())
        p_done = float(torch.sigmoid(z_done[0]).cpu().item())

        p_risk = F.softmax(z_risk[0], dim=-1).cpu().numpy()
        top_risk_idx = int(p_risk.argmax())

        # Calibrated alarm: triggers on high neural stuck prob or deterministic thrashing (>= 3 fails)
        alarm_triggered = bool(p_stuck > 0.50 or consecutive_failures >= 3)

        return {
            "turn": turn_index,
            "category": CATEGORIES[top_cat_idx],
            "category_confidence": round(float(p_cat[top_cat_idx]), 4),
            "phase": PHASES[top_phase_idx],
            "phase_confidence": round(float(p_phase[top_phase_idx]), 4),
            "is_stuck": round(p_stuck, 4),
            "is_stuck_alarm": alarm_triggered,
            "should_delegate": round(p_del, 4),
            "is_done": round(p_done, 4),
            "risk": RISK_LEVELS[top_risk_idx],
            "output_tokens": 0,  # Zero token generation invariant
        }
