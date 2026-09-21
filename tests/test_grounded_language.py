"""Unit and integration tests for Grounded Language Policy and Semantic Adapters."""

import pytest
import torch
from models.grounded_policy import TinyLanguageAdapter, GroundedPanduPolicy, StructuredIntent


def test_tiny_language_adapter_dimensions():
    """Verify adapter projects high-dimensional embeddings down to 16d."""
    adapter_bert = TinyLanguageAdapter(lm_dim=768, latent_dim=16)
    assert adapter_bert.count_parameters() == 12336

    dummy_bert_emb = torch.randn(4, 768)
    z_bert = adapter_bert(dummy_bert_emb)
    assert z_bert.shape == (4, 16)

    adapter_smol = TinyLanguageAdapter(lm_dim=576, latent_dim=16)
    assert adapter_smol.count_parameters() == 9264

    dummy_smol_emb = torch.randn(2, 576)
    z_smol = adapter_smol(dummy_smol_emb)
    assert z_smol.shape == (2, 16)


def test_grounded_pandu_policy_parameter_budget():
    """Verify GroundedPanduPolicy satisfies the ~5K parameter budget."""
    policy = GroundedPanduPolicy(env_dim=16, latent_lang_dim=16, num_actions=4)
    n_params = policy.count_parameters()
    # 32*48+48 (1584) + 48*64+64 (3136) + 64*4+4 (260) = 4980
    assert n_params == 4980
    assert n_params < 10000

    env_state = torch.randn(4, 16)
    z_lang = torch.randn(4, 16)
    logits = policy(env_state, z_lang)
    assert logits.shape == (4, 4)


def test_grounded_action_distribution_and_confidence():
    """Verify calibrated probability distribution with language conditioning."""
    policy = GroundedPanduPolicy(env_dim=16, latent_lang_dim=16, num_actions=4)
    env_state = torch.randn(16)
    z_lang = torch.randn(16)

    dist = policy.get_action_distribution(env_state, z_lang)
    assert dist["action"] in ["UP", "DOWN", "LEFT", "RIGHT"]
    assert 0.0 <= dist["confidence"] <= 1.0
    assert sum(dist["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_structured_intent_encoding():
    """Verify deterministic encoding of symbolic intent into normalized 16d vector."""
    intent = StructuredIntent(
        objective="reach_goal",
        direction="east",
        avoid_obstacle=True,
        urgency=0.8,
        exploration=0.2,
    )
    vec = intent.encode_to_vector(target_dim=16)
    assert vec.shape == (16,)
    assert isinstance(vec, torch.Tensor)
    assert pytest.approx(torch.norm(vec).item(), abs=1e-3) == 1.0
