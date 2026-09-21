"""Unit and integration tests for Grounded Language Policy, Semantic Adapters, and Intent Protocol."""

import pytest
import torch
from models.grounded import (
    TinyLanguageAdapter,
    GroundedPanduPolicy,
    CanonicalIntentProtocol,
    GoalSpec,
    ConstraintSpec,
    PreferenceSpec,
    StructuredIntent,
)
from datasets.language import build_oracle_intent, Action


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


def test_canonical_intent_protocol_serialization_and_encoding():
    """Verify CanonicalIntentProtocol encoding and dictionary serialization."""
    intent = CanonicalIntentProtocol(
        goal=GoalSpec(type="reach", target="goal", direction=(1.0, 0.0)),
        constraints=ConstraintSpec(avoid_obstacles=True, speed_limit=1.0),
        preferences=PreferenceSpec(risk=0.2, urgency=0.7),
    )
    d = intent.to_dict()
    assert d["goal"]["type"] == "reach"
    assert d["constraints"]["avoid_obstacles"] is True

    intent_reconstructed = CanonicalIntentProtocol.from_dict(d)
    assert intent_reconstructed.goal.direction == (1.0, 0.0)

    vec = intent.encode_to_vector(target_dim=16)
    assert vec.shape == (16,)
    assert pytest.approx(torch.norm(vec).item(), abs=1e-3) == 1.0


def test_oracle_intent_builder():
    """Verify Oracle Intent constructs valid normalized intent from environment coordinates."""
    oracle = build_oracle_intent(
        agent_x=2, agent_y=3, goal_x=8, goal_y=3, expert_action=Action.RIGHT
    )
    assert oracle.goal.type == "reach"
    assert oracle.goal.direction == (1.0, 0.0)
    vec = oracle.encode_to_vector(16)
    assert vec.shape == (16,)
    assert pytest.approx(torch.norm(vec).item(), abs=1e-3) == 1.0
