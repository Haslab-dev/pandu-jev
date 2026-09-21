"""Tests for GridWorld and 2D Racing environments."""

import pytest
import numpy as np
from env.gridworld import GridWorld, Action, create_random_gridworld
from env.racing import RacingEnv, RacingAction


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


def test_gridworld_representation_scaling():
    env = GridWorld(seed=42)
    for dim in [16, 32, 64, 128]:
        f = env.get_feature_vector(dim=dim)
        assert f.shape == (dim,), f"Expected shape ({dim},), got {f.shape}"
        assert not np.isnan(f).any()
        assert not np.isinf(f).any()


def test_racing_env_dynamics():
    env = RacingEnv(seed=42)
    obs = env.observe()
    assert "speed" in obs
    assert "heading" in obs
    assert "track_offset" in obs

    # Step accelerate
    obs, r, done, info = env.step(RacingAction.ACCELERATE)
    assert obs["speed"] > 0.3
    assert not done
