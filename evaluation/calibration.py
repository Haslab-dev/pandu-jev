"""Confidence calibration and uncertainty metrics (Phase 5 & Phase 6).

Implements:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability Diagram table generator
- Selective Classification / Failure Prediction AUROC
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_calibration_metrics(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE), MCE, and Brier Score.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Brier Score = (1/N) * sum_{i=1}^N (p_i - y_i)^2
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    assert len(confidences) == len(accuracies), "Size mismatch between confidences and accuracies"
    n_samples = len(confidences)

    if n_samples == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "bins": []}

    # Brier score: MSE between confidence and binary success/correctness
    brier_score = float(np.mean((confidences - accuracies) ** 2))

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            abs_diff = abs(bin_acc - bin_conf)
            weight = bin_count / n_samples
            ece += weight * abs_diff
            mce = max(mce, abs_diff)
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0
            abs_diff = 0.0

        bin_data.append(
            {
                "bin_idx": i,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
                "count": bin_count,
                "avg_confidence": bin_conf,
                "avg_accuracy": bin_acc,
                "calibration_gap": abs_diff,
            }
        )

    # Uncertainty error detection: correlation and AUROC for detecting mistakes via low confidence
    # (i.e. does confidence rank errors lower than correct decisions?)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(confidences, accuracies)
    if np.isnan(corr):
        corr = 0.0

    # AUROC for error detection (1 - accuracy as positive class, 1 - confidence as score)
    try:
        from sklearn.metrics import roc_auc_score
        y_error = 1.0 - accuracies
        uncertainty_score = 1.0 - confidences
        if len(np.unique(y_error)) > 1:
            auroc_error_detection = float(roc_auc_score(y_error, uncertainty_score))
        else:
            auroc_error_detection = 1.0
    except Exception:
        # Fallback ranking AUROC calculation using Wilcoxon-Mann-Whitney
        errors = confidences[accuracies == 0]
        corrects = confidences[accuracies == 1]
        if len(errors) > 0 and len(corrects) > 0:
            # Fraction of pairs where correct confidence > error confidence
            n_pairs = len(errors) * len(corrects)
            greater = sum(np.sum(c > errors) + 0.5 * np.sum(c == errors) for c in corrects)
            auroc_error_detection = float(greater / n_pairs)
        else:
            auroc_error_detection = 1.0

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "spearman_correlation": float(corr),
        "auroc_error_detection": auroc_error_detection,
        "bins": bin_data,
    }


def format_reliability_table(bin_data: List[Dict[str, Any]]) -> str:
    """Format reliability diagram as markdown table."""
    lines = [
        "| Confidence Bin | Count | Avg Confidence | Actual Accuracy | Gap |",
        "| :------------: | ----: | -------------: | --------------: | --: |",
    ]
    for b in bin_data:
        if b["count"] > 0:
            lines.append(
                f"| {b['range']} | {b['count']} | {b['avg_confidence']:.3f} | {b['avg_accuracy']:.3f} | {b['calibration_gap']:.3f} |"
            )
        else:
            lines.append(f"| {b['range']} | 0 | — | — | — |")
    return "\n".join(lines)
