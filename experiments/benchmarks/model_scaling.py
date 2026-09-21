"""Model Parameter Scaling Benchmark (Phase 10).

Benchmarks policies across multiple parameter scales:
- Policy-3K (Tiny canonical Phase 2 architecture)
- Policy-50K (Small MLP)
- Policy-1M (~1M parameters)
- Policy-5M (~5M parameters)

Measures: Parameters, Accuracy, Success Rate, Inference Latency, Memory Footprint.
"""

import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

from models.policy import build_scaled_model
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop


def run_model_scaling_benchmark(
    features: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    num_eval_episodes: int = 50,
) -> List[Dict[str, Any]]:
    """Train and evaluate models of escalating parameter sizes on identical data."""
    model_tiers = [
        ("Policy-3K (Tiny)", "tiny"),
        ("Policy-50K (Small)", "small"),
        ("Policy-1M", "1m"),
        ("Policy-5M", "5m"),
    ]

    results = []

    for name, tier in model_tiers:
        model = build_scaled_model(tier, input_dim=16, num_actions=4)
        n_params = model.count_parameters()
        mem_mb = (n_params * 4) / (1024 * 1024)

        # Train with behavioral cloning
        train_res = train_behavioral_cloning(
            model=model,
            features=features,
            actions=actions,
            epochs=epochs,
            batch_size=64,
            device=device,
        )

        # Closed-loop evaluation
        eval_res = evaluate_policy_closed_loop(
            model=model,
            num_episodes=num_eval_episodes,
            device=torch.device("cpu"),  # CPU latency reflects edge/embedded deployment
        )

        results.append(
            {
                "model_name": name,
                "tier": tier,
                "parameters": n_params,
                "memory_mb": mem_mb,
                "val_accuracy": train_res["final_val_acc"],
                "success_rate": eval_res["success_rate"],
                "avg_steps": eval_res["avg_steps"],
                "avg_reward": eval_res["avg_reward"],
                "latency_ms": eval_res["avg_latency_ms"],
                "p95_latency_ms": eval_res["p95_latency_ms"],
                "training_time": train_res["training_time_seconds"],
            }
        )

    return results
