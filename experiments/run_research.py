"""Automated Research Experiment Runner for Mini-Jev.

Executes all 7 core research experiments:
1. Dataset collection & Behavioral Cloning training
2. Pre vs Post Temperature Calibration (ECE, Brier, Reliability Diagram)
3. Uncertainty & OOD benchmark (In-Distribution, Unseen, Large, Blocked, Noisy)
4. Reinforcement Learning (PPO) vs Behavioral Cloning comparison
5. Stress Testing Suite (Versions A through E)
6. Parameter Scaling Law (Policy-3K, 50K, 1M, 5M)
7. Hybrid Fallback Architecture (Teacher vs Mini-Jev vs Hybrid)
"""

import json
import os
import time
from typing import Any, Dict
import numpy as np
import torch

from env.gridworld import GridWorld
from models.policy import TinyPolicy, build_scaled_model
from datasets.trajectory_dataset import collect_expert_trajectories, save_dataset
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop, fit_temperature
from training.ppo import train_ppo
from evaluation.calibration import compute_calibration_metrics, format_reliability_table
from evaluation.uncertainty import run_uncertainty_benchmark
from evaluation.stress_testing import run_stress_test_suite
from experiments.model_scaling import run_model_scaling_benchmark
from experiments.hybrid_fallback import run_hybrid_fallback_benchmark

OUTPUT_DIR = "experiments/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_full_research():
    print("=" * 70)
    print("Mini-Jev: Comprehensive Empirical Research Suite")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware compute device: {device}")

    # 1. Dataset Generation
    print("\n[Phase 3] Generating Expert Demonstration Dataset...")
    n_episodes = 2500
    X, y, data_stats = collect_expert_trajectories(num_episodes=n_episodes, seed=42)
    save_dataset("data/expert_trajectories.npz", X, y, data_stats)
    print(f"Collected {len(y):,} state-action transitions from {n_episodes} A* episodes.")
    print(f"Expert success rate: {data_stats['success_rate']*100:.1f}%, Avg steps: {data_stats['avg_steps_per_episode']:.2f}")

    # 2. Behavioral Cloning Training
    print("\n[Phase 4] Training Canonical TinyPolicy (<1M params)...")
    model = TinyPolicy(input_dim=16, num_actions=4)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} (~{n_params/1000:.1f}K params)")

    bc_train_results = train_behavioral_cloning(
        model=model,
        features=X,
        actions=y,
        epochs=35,
        batch_size=64,
        lr=1e-3,
        device=device,
        seed=42,
    )
    print(f"BC Training complete in {bc_train_results['training_time_seconds']:.2f}s.")
    print(f"Final Validation Accuracy: {bc_train_results['final_val_acc']*100:.2f}%")
    print(f"Fitted Optimal Temperature: {bc_train_results['optimal_temperature']:.3f}")

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/tiny_policy.pt")

    # Closed-loop evaluation
    closed_loop_eval = evaluate_policy_closed_loop(model, num_episodes=100, device=torch.device("cpu"))
    print(f"Closed-loop Rollout Success Rate: {closed_loop_eval['success_rate']*100:.1f}%")
    print(f"Average Steps: {closed_loop_eval['avg_steps']:.2f}, Average Reward: {closed_loop_eval['avg_reward']:+.1f}")
    print(f"Inference Latency: {closed_loop_eval['avg_latency_ms']:.3f} ms (P95: {closed_loop_eval['p95_latency_ms']:.3f} ms)")

    # 3. Calibration Analysis (Phase 6)
    print("\n[Phase 6] Evaluating Confidence Calibration & Reliability Diagrams...")
    # Evaluate calibration on holdout validation set
    val_split = int(len(X) * 0.2)
    X_val = X[-val_split:]
    y_val = y[-val_split:]

    # Uncalibrated (T=1.0)
    uncal_confs = []
    uncal_accs = []
    # Calibrated (T=optimal_t)
    cal_confs = []
    cal_accs = []

    model.eval()
    with torch.no_grad():
        for i in range(len(X_val)):
            d_uncal = model.get_action_distribution(X_val[i], temperature=1.0)
            d_cal = model.get_action_distribution(X_val[i], temperature=bc_train_results["optimal_temperature"])
            act_true = y_val[i]

            uncal_confs.append(d_uncal["confidence"])
            uncal_accs.append(1 if d_uncal["action_idx"] == act_true else 0)

            cal_confs.append(d_cal["confidence"])
            cal_accs.append(1 if d_cal["action_idx"] == act_true else 0)

    uncal_metrics = compute_calibration_metrics(np.array(uncal_confs), np.array(uncal_accs))
    cal_metrics = compute_calibration_metrics(np.array(cal_confs), np.array(cal_accs))

    print(f"Uncalibrated (T=1.0) -> ECE: {uncal_metrics['ece']:.4f}, Brier: {uncal_metrics['brier_score']:.4f}")
    print(f"Calibrated (T={bc_train_results['optimal_temperature']:.2f}) -> ECE: {cal_metrics['ece']:.4f}, Brier: {cal_metrics['brier_score']:.4f}")

    # 4. Uncertainty & Out-of-Distribution Benchmark (Phase 5)
    print("\n[Phase 5] Running Uncertainty Benchmark Across 5 Regimes...")
    uncertainty_results = run_uncertainty_benchmark(model, num_episodes_per_regime=50, seed=42)
    for reg, d in uncertainty_results.items():
        print(f"  [{reg:18s}] Success: {d['episode_success_rate']*100:5.1f}% | Conf: {d['mean_confidence']:.3f} | ECE: {d['ece']:.3f} | AUROC: {d['auroc_error_detection']:.3f}")

    # 5. Reinforcement Learning (PPO) comparison (Phase 7)
    print("\n[Phase 7] Training Reinforcement Learning (PPO) Baseline...")
    ppo_model, ppo_stats = train_ppo(total_episodes=400, seed=42)
    print(f"PPO Training Finished. Final Avg Reward: {ppo_stats['final_avg_reward']:+.1f}, Success: {ppo_stats['final_success_rate']*100:.1f}%")

    # 6. Stress Testing Suite (Phase 8)
    print("\n[Phase 8] Running Stress Testing Suite (Versions A through E)...")
    stress_results = run_stress_test_suite(model, num_episodes_per_variant=50, seed=42)
    for var, d in stress_results.items():
        print(f"  [{var:20s}] Success: {d['success_rate']*100:5.1f}% | Avg Rew: {d['avg_reward']:+6.1f} | Conf: {d['mean_confidence']:.3f}")

    # 7. Model Parameter Scaling Law (Phase 10)
    print("\n[Phase 10] Running Model Parameter Scaling Benchmark (3K, 50K, 1M, 5M)...")
    scaling_results = run_model_scaling_benchmark(
        features=X[:8000],  # representative subset for fast clean benchmarking
        actions=y[:8000],
        device=device,
        epochs=20,
        num_eval_episodes=40,
    )
    for sc in scaling_results:
        print(f"  [{sc['model_name']:18s}] Params: {sc['parameters']:>9,} | Val Acc: {sc['val_accuracy']*100:5.1f}% | Rollout Succ: {sc['success_rate']*100:5.1f}% | Latency: {sc['latency_ms']:.3f}ms")

    # 8. Hybrid Fallback Benchmark (Phase 14 & Final Benchmark)
    print("\n[Phase 14] Running Hybrid Fallback Architecture Benchmark...")
    hybrid_results = run_hybrid_fallback_benchmark(model, num_episodes=100, confidence_threshold=0.85, seed=42)
    for arch, d in hybrid_results.items():
        print(f"  [{arch:25s}] Success: {d['success_rate']*100:5.1f}% | Latency: {d['avg_latency_ms']:6.2f}ms | Cost/Ep: ${d['cost_per_episode_usd']:.4f} | Fallback: {d['fallback_frequency']*100:4.1f}%")

    # Save all raw results
    all_results = {
        "dataset_stats": data_stats,
        "bc_training": {
            "final_train_acc": bc_train_results["final_train_acc"],
            "final_val_acc": bc_train_results["final_val_acc"],
            "optimal_temperature": bc_train_results["optimal_temperature"],
            "training_time_seconds": bc_train_results["training_time_seconds"],
        },
        "closed_loop_eval": closed_loop_eval,
        "calibration": {
            "uncalibrated": uncal_metrics,
            "calibrated": cal_metrics,
        },
        "uncertainty_benchmark": uncertainty_results,
        "ppo_stats": {
            "final_avg_reward": ppo_stats["final_avg_reward"],
            "final_success_rate": ppo_stats["final_success_rate"],
            "total_episodes": ppo_stats["total_episodes"],
        },
        "stress_test": stress_results,
        "scaling_laws": scaling_results,
        "hybrid_fallback": hybrid_results,
    }

    result_json_path = os.path.join(OUTPUT_DIR, "all_research_metrics.json")
    with open(result_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experimental results saved to {result_json_path}")
    return all_results


if __name__ == "__main__":
    run_full_research()
