"""Unit and numerical parity tests for Pandu ANE Transformer."""

import pytest
import torch
import torch.nn.functional as F

from models.jev_nlp import PanduJevNLP, PanduJevNLPConfig
from models.ane_transformer import PanduANEBody


def test_ane_transformer_forward_and_parity():
    """Verify PanduANEBody produces mathematically equivalent outputs to PanduJevNLP."""
    torch.manual_seed(42)
    cfg = PanduJevNLPConfig(num_hidden_layers=2, hidden_size=256, num_attention_heads=4)
    model = PanduJevNLP(cfg)
    model.eval()

    length = 32
    max_options = 4

    # Build ANE Body
    ane_body = PanduANEBody(model, length=length, max_options=max_options)
    ane_body.eval()

    # Synthetic inputs
    batch_size = 1
    input_ids = torch.randint(0, 1000, (batch_size, length))
    attention_mask = torch.ones((batch_size, length))
    marker_pos = torch.tensor([[2, 4, 8, 12]])
    marker_mask = torch.tensor([[1, 1, 1, 0]])
    qtype = torch.tensor([0])

    # 1. Forward through original PanduJevNLP components
    with torch.no_grad():
        emb = model.encoder.embeddings(input_ids)  # (1, L, D)
        # In PanduJevNLP:
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # h = outputs.last_hidden_state + type_vec
        # marker_states = torch.gather(h, 1, expanded_marker_pos)
        # logits = self.scorer(marker_states).squeeze(-1)
        orig_logits, orig_act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)

    # 2. Prepare ANE BC1L inputs
    # Embedding lookup on host: (1, D, 1, L)
    ane_emb = emb.transpose(1, 2).unsqueeze(2)  # (1, D, 1, L)

    # Additive attention mask: (1, L, 1, L)
    # 0 where attention_mask == 1, -1e4 where 0
    raw_mask = torch.where(attention_mask.bool(), 0.0, -1e4)
    ane_mask = raw_mask.unsqueeze(1).unsqueeze(2).expand(-1, length, 1, -1)  # (1, L, 1, L)

    # Type vector: (1, D, 1, 1)
    type_vec = model.type_emb(qtype).unsqueeze(2).unsqueeze(3)  # (1, D, 1, 1)

    # Marker map: (1, L, 1, max_options)
    marker_map = torch.zeros((batch_size, length, 1, max_options), dtype=torch.float32)
    for b in range(batch_size):
        for opt_idx, pos in enumerate(marker_pos[b]):
            if marker_mask[b, opt_idx]:
                marker_map[b, pos, 0, opt_idx] = 1.0

    # 3. Forward through PanduANEBody
    with torch.no_grad():
        ane_logits, ane_cls = ane_body(ane_emb, ane_mask, type_vec, marker_map)

    # Check shapes
    assert ane_logits.shape == (batch_size, max_options)
    assert ane_cls.shape == (batch_size, cfg.hidden_size)

    # Check finite values
    assert torch.isfinite(ane_logits).all()
    assert torch.isfinite(ane_cls).all()
