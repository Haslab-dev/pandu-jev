"""Tests for imitation learning and dataset generation."""

import pytest
import torch
from datasets.trajectory import collect_expert_trajectories
from models.policy import TinyPolicy
from training.bc import train_behavioral_cloning


def test_dataset_generation_and_bc_training():
    X, y, stats = collect_expert_trajectories(num_episodes=20, seed=42)
    assert len(X) == len(y)
    assert len(X) > 0
    assert stats["success_rate"] == 1.0

    model = TinyPolicy(input_dim=16, num_actions=4)
    res = train_behavioral_cloning(model, X, y, epochs=5, batch_size=32, device=torch.device("cpu"))
    assert "final_val_acc" in res
    assert res["final_val_acc"] >= 0.0
