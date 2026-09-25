"""Pandu-Jev ANE Transformer Architecture.

Implements a 4D BC1L (Batch, Channel, 1, Length) Conv2D-lowered transformer
specifically optimized for the Apple Neural Engine (ANE) on Apple Silicon.
Mirrors PanduJevNLP (ModernBERT-Tiny, 19.3M, 6L, d=256, 4 heads) with exact
mathematical parity and zero CPU fallback operations on Core ML.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_from_linear(linear: nn.Linear) -> nn.Conv2d:
    """Lower nn.Linear to nn.Conv2d(1x1) for systolic ANE execution."""
    has_bias = linear.bias is not None
    conv = nn.Conv2d(linear.in_features, linear.out_features, kernel_size=1, bias=has_bias)
    conv.weight = nn.Parameter(linear.weight.detach()[:, :, None, None])
    if has_bias:
        conv.bias = nn.Parameter(linear.bias.detach())
    return conv


def conv_from_weights(weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> nn.Conv2d:
    """Create 1x1 Conv2d directly from weight tensor (out_features, in_features)."""
    has_bias = bias is not None
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], kernel_size=1, bias=has_bias)
    conv.weight = nn.Parameter(weight.detach()[:, :, None, None])
    if has_bias:
        conv.bias = nn.Parameter(bias.detach())
    return conv


class ChannelNorm(nn.Module):
    """Channel-axis normalization for 4D BC1L tensors.
    
    Normalizes across dimension 1 (Channels), perfectly matching LayerNorm
    while staying entirely on ANE without triggering CPU device fallback.
    """

    def __init__(self, source_norm: nn.LayerNorm):
        super().__init__()
        self.eps = float(source_norm.eps)
        # Weight shape: (1, Channels, 1, 1)
        self.weight = nn.Parameter(source_norm.weight.detach()[None, :, None, None])
        if source_norm.bias is not None:
            self.bias = nn.Parameter(source_norm.bias.detach()[None, :, None, None])
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, 1, L)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        rsqrt = (var + self.eps).rsqrt()
        normed = (x - mean) * rsqrt * self.weight
        return normed if self.bias is None else normed + self.bias


class ConvAttention(nn.Module):
    """ANE-optimized Multi-Head Attention using 1x1 Convs and split-head contractions."""

    def __init__(
        self,
        attn_module: nn.Module,
        heads: int,
        dim: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads  # 256 // 4 = 64
        self.dim = dim

        # Wqkv projection: (3*D, D, 1, 1)
        self.qkv = conv_from_linear(attn_module.Wqkv)
        # Wo projection: (D, D, 1, 1)
        self.out = conv_from_linear(attn_module.Wo)

        # Precomputed RoPE tables: (1, head_dim, 1, L)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation within head channels: x is (B, head_dim, 1, L)."""
        left, right = x.chunk(2, dim=1)
        # ModernBERT RoPE rotate_half: [-right, left]
        rot_half = torch.cat((-right, left), dim=1)
        return x * self.cos + rot_half * self.sin

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, D, 1, L)
        # mask: (B, L, 1, L) - additive mask (0 for attend, -1e4 for pad)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)  # Each is (B, D, 1, L)

        output_heads = []
        scale = float(self.head_dim**-0.5)

        for qi, ki, vi in zip(
            q.split(self.head_dim, dim=1),
            k.split(self.head_dim, dim=1),
            v.split(self.head_dim, dim=1),
        ):
            # Apply RoPE
            qi = self.rotate(qi)
            ki = self.rotate(ki)

            # Contract Q and K across head_dim (dim 1):
            # qi: (B, C, 1, Q), ki.transpose(1, 3): (B, K, 1, C)
            # scores: (B, K, 1, Q) -> keys on dim 1, queries on dim 3
            scores = torch.einsum("bchq,bkhc->bkhq", qi, ki.transpose(1, 3)) * scale
            probs = F.softmax(scores + mask, dim=1)

            # Contract probabilities with V:
            # probs: (B, K, 1, Q), vi: (B, C, 1, K)
            # head_out: (B, C, 1, Q)
            head_out = torch.einsum("bkhq,bchk->bchq", probs, vi)
            output_heads.append(head_out)

        concatenated = torch.cat(output_heads, dim=1)  # (B, D, 1, L)
        return self.out(concatenated)


class ConvMLP(nn.Module):
    """ANE-optimized Gated MLP with 1x1 convolutions and exact GELU."""

    def __init__(self, mlp_module: nn.Module):
        super().__init__()
        self.Wi = conv_from_linear(mlp_module.Wi)  # (2*Intermediate, D, 1, 1)
        self.Wo = conv_from_linear(mlp_module.Wo)  # (D, Intermediate, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, 1, L)
        projected = self.Wi(x)
        value, gate = projected.chunk(2, dim=1)
        activated = F.gelu(value) * gate
        return self.Wo(activated)


class ConvEncoderLayer(nn.Module):
    """Single ModernBERT-Tiny encoder layer lowered for Apple Neural Engine."""

    def __init__(
        self,
        layer: nn.Module,
        heads: int,
        dim: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        super().__init__()
        # In ModernBERT, layer 0 does not have attn_norm
        if hasattr(layer, "attn_norm") and layer.attn_norm is not None and not isinstance(layer.attn_norm, nn.Identity):
            self.attn_norm = ChannelNorm(layer.attn_norm)
        else:
            self.attn_norm = nn.Identity()

        self.attn = ConvAttention(layer.attn, heads=heads, dim=dim, cos=cos, sin=sin)
        self.mlp_norm = ChannelNorm(layer.mlp_norm)
        self.mlp = ConvMLP(layer.mlp)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class PanduANEBody(nn.Module):
    """Complete ANE-compiled Transformer Body for Pandu-Jev NLP.

    Takes:
        embeddings: Float16 (B, D, 1, L) looked up on host CPU
        attention_mask: Float16 (B, L, 1, L) additive mask (0 for active, -1e4 for pad)
        type_vectors: Float16 (B, D, 1, 1) from type_emb(qtype)
        marker_map: Float16 (B, L, 1, MaxOptions) one-hot option index matrix

    Returns:
        logits: Float16 (B, MaxOptions) unnormalized candidate scores
        cls_hidden: Float16 (B, D) CLS representation for action telemetry
    """

    def __init__(
        self,
        source_model: nn.Module,
        length: int = 96,
        max_options: int = 32,
    ):
        super().__init__()
        self.length = length
        self.max_options = max_options
        self.dim = source_model.hidden_size
        self.heads = source_model.config.num_attention_heads
        self.head_dim = self.dim // self.heads

        # Precompute RoPE cos/sin buffers for fixed length
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(0)  # (1, L)
        dummy_x = torch.zeros(1, length, self.heads, self.head_dim)
        raw_cos, raw_sin = source_model.encoder.rotary_emb(
            dummy_x, pos, layer_type="full_attention"
        )
        # raw_cos is (1, L, head_dim) -> transpose to (1, head_dim, 1, L)
        cos_table = raw_cos[0].T[None, :, None, :]  # (1, head_dim, 1, L)
        sin_table = raw_sin[0].T[None, :, None, :]  # (1, head_dim, 1, L)

        # Embedding norm
        if hasattr(source_model.encoder.embeddings, "norm") and source_model.encoder.embeddings.norm is not None:
            self.embedding_norm = ChannelNorm(source_model.encoder.embeddings.norm)
        else:
            self.embedding_norm = nn.Identity()

        # Encoder layers
        self.layers = nn.ModuleList([
            ConvEncoderLayer(layer, heads=self.heads, dim=self.dim, cos=cos_table, sin=sin_table)
            for layer in source_model.encoder.layers
        ])

        # Final norm
        self.final_norm = ChannelNorm(source_model.encoder.final_norm)

        # Scorer (lowered to 1x1 convs)
        self.scorer = nn.Sequential(
            ChannelNorm(source_model.scorer[0]),
            conv_from_linear(source_model.scorer[1]),
            nn.GELU(),
            conv_from_linear(source_model.scorer[3]),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        type_vectors: torch.Tensor,
        marker_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Apply embedding norm
        x = self.embedding_norm(embeddings)  # (B, D, 1, L)

        # 2. Pass through ANE-optimized encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # 3. Final normalization and question-type semantic conditioning
        x = self.final_norm(x) + type_vectors

        # 4. Gather marker hidden states via matrix contraction:
        # marker_map: (B, L, 1, K) where K = max_options, L = sequence_length
        # x: (B, D, 1, L)
        # Contract over L (axis 1 of marker_map and axis 3 of x)
        # markers: (B, D, 1, K)
        markers = torch.einsum("bkhq,bchk->bchq", marker_map, x)

        # 5. Score candidate option markers: (B, 1, 1, K) -> (B, K)
        scores = self.scorer(markers).squeeze(1).squeeze(1)

        # 6. Extract CLS state at position 0: (B, D, 1, 1) -> (B, D)
        cls_state = x[:, :, :, 0].squeeze(2)

        return scores, cls_state
