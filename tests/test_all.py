"""Comprehensive unit and integration test suite for Mini-Jev."""

import pytest
import numpy as np
import torch

from env.gridworld import GridWorld, Action, create_random_gridworld
from env.racing import RacingEnv, RacingAction
from expert.astar import AStarExpert
from models.policy import TinyPolicy, ScalablePolicy, build_scaled_model
from datasets.trajectory_dataset import collect_expert_trajectories
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop
from evaluation.calibration import compute_calibration_metrics
from evaluation.uncertainty import run_uncertainty_benchmark
from experiments.hybrid_fallback import run_hybrid_fallback_benchmark


def test_gridworld_initialization_and_step():
    env = GridWorld(seed=42)
    obs = env.observe()
    assert obs["agent_x"] == 1
    assert obs["agent_y"] == 1
    assert obs["goal_x"] == 10
    assert obs["goal_y"] == 5

    # Move right
    obs, r, done, info = env.step(Action.RIGHT)
    assert obs["agent_x"] == 2
    assert obs["agent_y"] == 1
    assert r == -1.0
    assert not done

    # Move up into wall at (2, 0)
    obs, r, done, info = env.step(Action.UP)
    assert obs["agent_x"] == 2
    assert obs["agent_y"] == 1  # Agent stays
    assert info["hit_wall"] is True
    assert r == -11.0  # -1 step + -10 wall


def test_gridworld_feature_vector():
    env = GridWorld(seed=42)
    feat = env.get_feature_vector()
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (16,)
    assert np.all(feat >= -1.0) and np.all(feat <= 2.0)


def test_astar_expert():
    env = GridWorld(seed=42)
    expert = AStarExpert()
    path = expert.find_path(
        start=(1, 1),
        goal=(env.goal_x, env.goal_y),
        walls=env.current_walls,
        width=env.width,
        height=env.height,
    )
    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (env.goal_x, env.goal_y)

    action = expert.get_action(env)
    assert action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


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


def test_calibration_metrics_computation():
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.4])
    accuracies = np.array([1, 1, 1, 0, 1, 0])
    metrics = compute_calibration_metrics(confidences, accuracies, num_bins=5)
    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert "auroc_error_detection" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_racing_env():
    env = RacingEnv(seed=42)
    obs = env.observe()
    assert "speed" in obs
    assert "heading" in obs
    assert "track_offset" in obs

    # Step accelerate
    obs, r, done, info = env.step(RacingAction.ACCELERATE)
    assert obs["speed"] > 0.3
    assert not done


def test_dataset_generation_and_bc_training():
    X, y, stats = collect_expert_trajectories(num_episodes=20, seed=42)
    assert len(X) == len(y)
    assert len(X) > 0
    assert stats["success_rate"] == 1.0

    model = TinyPolicy(input_dim=16, num_actions=4)
    res = train_behavioral_cloning(model, X, y, epochs=5, batch_size=32, device=torch.device("cpu"))
    assert "final_val_acc" in res
    assert res["final_val_acc"] >= 0.0
