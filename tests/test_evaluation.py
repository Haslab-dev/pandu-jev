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
