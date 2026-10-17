"""MicroBERT: Compact Bidirectional Transformer (~1.5M Params / ~1.8 MB INT8).

Features:
- Pruned 2,048-token domain vocabulary (eliminates the 20MB vocabulary tax)
- 3-layer Transformer Encoder (d=192, 4 heads, intermediate=512)
- Zero-token typed decision heads (category, needs_tools, role, risk, delegate)
- Full self-attention across tokens (enabling precise syntactic negation resolution)
- Sub-millisecond execution on Apple Silicon MPS / CPU
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.rakitkode_domain import CATEGORIES, ROLES, RISK_LEVELS
from tiny.tokenizer import MicroTokenizer


@dataclass
class MicroBERTConfig:
    """Configuration for MicroBERT Architecture."""
    vocab_size: int = 2048
    hidden_size: int = 192
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 512
    dropout: float = 0.1
    num_categories: int = len(CATEGORIES)  # 9
    num_roles: int = len(ROLES)            # 5
    num_risks: int = len(RISK_LEVELS)      # 3


class MicroBERT(nn.Module):
    """Compact 1.5M-parameter Transformer for Typed Decision Tasks."""

    def __init__(self, config: Optional[MicroBERTConfig] = None):
        super().__init__()
        self.config = config or MicroBERTConfig()
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

        # 3. Decision Pooling Head
        self.pooler = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
        )

        # 4. Multi-Task Typed Decision Heads
        self.cat_head = nn.Linear(d, self.config.num_categories)
        self.tools_head = nn.Linear(d, 1)
        self.role_head = nn.Linear(d, self.config.num_roles)
        self.risk_head = nn.Linear(d, self.config.num_risks)
        self.delegate_head = nn.Linear(d, 1)

        # Temperature parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
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
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def disk_size_mb(self) -> float:
        """Resident disk footprint in FP32 / FP16."""
        fp32_mb = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)
        return fp32_mb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transformer and multi-task heads.
        
        Args:
            input_ids: [BatchSize, SeqLen] integer tensor
            attention_mask: [BatchSize, SeqLen] binary mask (1 for valid, 0 for pad)
        
        Returns:
            (logits_cat, z_tools, logits_role, logits_risk, z_delegate)
        """
        b, seq_len = input_ids.shape
        device = input_ids.device

        # Token + Position Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.emb_drop(self.emb_norm(x))

        # PyTorch TransformerEncoder expects src_key_padding_mask where True = ignore
        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)

        # Encode sequence
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        # Pool [CLS] token at position 0
        cls_rep = self.pooler(h[:, 0])

        # Head outputs
        logits_cat = self.cat_head(cls_rep)
        z_tools = self.tools_head(cls_rep).squeeze(-1)
        logits_role = self.role_head(cls_rep)
        logits_risk = self.risk_head(cls_rep)
        z_delegate = self.delegate_head(cls_rep).squeeze(-1)

        return logits_cat, z_tools, logits_role, logits_risk, z_delegate

    @torch.no_grad()
    def predict_turn(
        self,
        text: str,
        tokenizer: MicroTokenizer,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Inference for a single prompt producing typed JSON output."""
        self.eval()
        token_ids = tokenizer.encode(text, max_length=256)
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
        mask_t = torch.ones_like(ids_t)

        z_cat, z_tools, z_role, z_risk, z_del = self.forward(ids_t, mask_t)

        # Calibrated Softmax over categories
        p_cat = F.softmax(z_cat[0] / self.temperature, dim=-1).cpu().numpy()
        top_cat_idx = int(p_cat.argmax())
        cat_confidence = float(p_cat[top_cat_idx])

        # Needs tools probability
        p_tools = float(torch.sigmoid(z_tools[0]).cpu().item())

        # Roles
        p_role = F.softmax(z_role[0], dim=-1).cpu().numpy()
        role_idx = int(p_role.argmax())

        # Risks
        p_risk = F.softmax(z_risk[0], dim=-1).cpu().numpy()
        risk_idx = int(p_risk.argmax())

        # Delegation
        p_del = float(torch.sigmoid(z_del[0]).cpu().item())

        return {
            "category": CATEGORIES[top_cat_idx],
            "confidence": round(cat_confidence, 4),
            "probabilities": {CATEGORIES[i]: round(float(p_cat[i]), 4) for i in range(len(CATEGORIES))},
            "needs_tools": round(p_tools, 4),
            "role": ROLES[role_idx],
            "role_confidence": round(float(p_role[role_idx]), 4),
            "risk": RISK_LEVELS[risk_idx],
            "delegate": bool(p_del > 0.5),
            "output_tokens": 0,  # Zero token generation invariant
        }
