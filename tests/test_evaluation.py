"""Tests for uncertainty calibration and evaluation metrics."""

import pytest
import numpy as np
from evaluation.calibration import compute_calibration_metrics


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


def test_behavior_metrics_computation():
    from evaluation.behavior import evaluate_closed_loop_behavior
    from models.policy import TinyPolicy

    policy = TinyPolicy(input_dim=16, num_actions=4, hidden_dims=(16, 16))
    metrics = evaluate_closed_loop_behavior(
        model=policy,
        num_episodes=5,
        map_types="random",
        seed=42,
    )
    assert "trajectory_success_rate" in metrics
    assert "wall_hits_per_episode" in metrics
    assert "mean_episode_length" in metrics
    assert "outcome_breakdown" in metrics
    assert 0.0 <= metrics["trajectory_success_rate"] <= 1.0
    assert metrics["episodes"] == 5

