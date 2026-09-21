"""Tests for policy architectures and model parameter scaling."""

import pytest
import torch
from models.policy import TinyPolicy, ScalablePolicy, build_scaled_model


def test_tiny_policy_architecture():
    policy = TinyPolicy(input_dim=16, num_actions=4)
    n_params = policy.count_parameters()
    # 16*32+32 + 32*64+64 + 64*4+4 = 544 + 2112 + 260 = 2916 params
    assert n_params == 2916
    assert n_params < 1_000_000

    dummy_input = torch.randn(1, 16)
    logits = policy(dummy_input)
    assert logits.shape == (1, 4)

    dist = policy.get_action_distribution(dummy_input[0])
    assert "action" in dist
    assert "confidence" in dist
    assert 0.0 <= dist["confidence"] <= 1.0
    assert sum(dist["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_scalable_policy_tiers():
    p_3k = build_scaled_model("3k")
    p_50k = build_scaled_model("50k")
    p_1m = build_scaled_model("1m")
    p_5m = build_scaled_model("5m")

    assert p_3k.count_parameters() < 10_000
    assert 40_000 < p_50k.count_parameters() < 100_000
    assert 800_000 < p_1m.count_parameters() < 2_000_000
    assert 4_000_000 < p_5m.count_parameters() < 8_000_000


def test_recurrent_policy_architecture():
    from models.recurrent import RecurrentPanduPolicy
    rec_policy = RecurrentPanduPolicy(input_dim=16, hidden_dim=32, num_actions=4)
    n_params = rec_policy.count_parameters()
    # 16*32+32 (544) + GRUCell(32,32) (6336) + 32*4+4 (132) = 7012 params
    assert n_params == 7012
    assert n_params < 20_000

    # Step forward
    x = torch.randn(1, 16)
    logits, h = rec_policy(x)
    assert logits.shape == (1, 4)
    assert h.shape == (1, 32)

    # Sequence forward
    x_seq = torch.randn(2, 10, 16)
    all_logits, h_last = rec_policy.forward_sequence(x_seq)
    assert all_logits.shape == (2, 10, 4)
    assert h_last.shape == (2, 32)

    # Action distribution
    dist = rec_policy.get_action_distribution(x[0], h=h)
    assert "action_idx" in dist
    assert "confidence" in dist
    assert "hidden" in dist
    assert 0.0 <= dist["confidence"] <= 1.0
