"""Hybrid Fallback Runtime Benchmark (Phase 14 & Final Benchmark).

Evaluates the central hypothesis of Pandu (pandu-jev):
Can a tiny, zero-cost local policy combined with confidence-gated fallback to an
expensive teacher/LLM achieve teacher-grade success rates at a fraction of latency and cost?

Compares:
- Policy A: Teacher Alone (Optimal oracle with simulated API latency & token cost)
- Policy B: Pandu Alone (Tiny local model, 0 cost, sub-millisecond)
- Policy C: Hybrid Fallback (Pandu when confidence >= tau, fallback to Teacher otherwise)
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from env.gridworld import GridWorld, Action, create_random_gridworld
from expert.astar import AStarExpert


class SimulatedTeacher:
    """Oracle Teacher modeling LLM API behavior with token cost and latency simulation."""

    COST_PER_CALL_USD = 0.0025  # Simulated cost per action query ($2.50 per 1k calls)
    TOKENS_PER_CALL = 350       # Average tokens per state reasoning prompt
    SIMULATED_LATENCY_MS = 380.0  # Network roundtrip + reasoning time (ms)

    def __init__(self):
        self.expert = AStarExpert()

    def get_action(self, env: GridWorld, simulate_sleep: bool = False) -> Tuple[Optional[Action], float, int, float]:
        """Returns (action, latency_ms, tokens, cost_usd)."""
        if simulate_sleep:
            time.sleep(self.SIMULATED_LATENCY_MS / 1000.0)

        action = self.expert.get_action(env)
        return action, self.SIMULATED_LATENCY_MS, self.TOKENS_PER_CALL, self.COST_PER_CALL_USD


def run_hybrid_fallback_benchmark(
    model: nn.Module,
    num_episodes: int = 100,
    confidence_threshold: float = 0.85,
    include_novel_maps: bool = True,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Execute head-to-head comparison across Teacher, Pandu (pandu-jev), and Hybrid architectures."""
    model.eval()
    teacher = SimulatedTeacher()
    rng = np.random.RandomState(seed)

    architectures = ["Teacher Alone", "Pandu Alone", f"Hybrid Fallback (τ={confidence_threshold:.2f})"]
    results: Dict[str, Dict[str, Any]] = {}

    for arch in architectures:
        episodes_success = 0
        total_steps = 0
        total_reward = 0.0
        total_wall_hits = 0
        total_latency_ms = 0.0
        total_tokens = 0
        total_cost_usd = 0.0
        fallback_count = 0
        total_decisions = 0

        for ep in range(num_episodes):
            ep_seed = int(rng.randint(0, 1_000_000))
            if include_novel_maps and (ep % 2 == 1):
                env = create_random_gridworld(width=12, height=7, wall_prob=0.20, seed=ep_seed)
            else:
                env = GridWorld(random_start_goal=True, seed=ep_seed)

            done = False
            ep_steps = 0
            ep_reward = 0.0

            while not done and ep_steps < env.max_steps:
                total_decisions += 1

                if arch == "Teacher Alone":
                    act_expert, lat, tok, cost = teacher.get_action(env, simulate_sleep=False)
                    action = int(act_expert) if act_expert is not None else 0
                    total_latency_ms += lat
                    total_tokens += tok
                    total_cost_usd += cost

                elif arch in ("Pandu Alone", "Mini-Jev Alone"):
                    feat = env.get_feature_vector()
                    t0 = time.perf_counter()
                    dist = model.get_action_distribution(feat)
                    t1 = time.perf_counter()
                    action = dist["action_idx"]
                    total_latency_ms += (t1 - t0) * 1000.0

                else:  # Hybrid Fallback
                    feat = env.get_feature_vector()
                    t0 = time.perf_counter()
                    dist = model.get_action_distribution(feat)
                    t1 = time.perf_counter()
                    local_lat = (t1 - t0) * 1000.0

                    if dist["confidence"] >= confidence_threshold:
                        # High confidence: execute locally
                        action = dist["action_idx"]
                        total_latency_ms += local_lat
                    else:
                        # Low confidence: fallback to Teacher
                        fallback_count += 1
                        act_expert, lat, tok, cost = teacher.get_action(env, simulate_sleep=False)
                        action = int(act_expert) if act_expert is not None else dist["action_idx"]
                        total_latency_ms += (local_lat + lat)
                        total_tokens += tok
                        total_cost_usd += cost

                obs, r, done, info = env.step(action)
                ep_reward += r
                ep_steps += 1
                if info.get("hit_wall", False):
                    total_wall_hits += 1
                if info.get("reached_goal", False):
                    episodes_success += 1

            total_steps += ep_steps
            total_reward += ep_reward

        results[arch] = {
            "architecture": arch,
            "episodes": num_episodes,
            "total_decisions": total_decisions,
            "success_rate": episodes_success / max(1, num_episodes),
            "avg_steps": total_steps / max(1, num_episodes),
            "avg_reward": total_reward / max(1, num_episodes),
            "wall_hits_per_episode": total_wall_hits / max(1, num_episodes),
            "avg_latency_ms": total_latency_ms / max(1, total_decisions),
            "total_cost_usd": total_cost_usd,
            "cost_per_episode_usd": total_cost_usd / max(1, num_episodes),
            "total_tokens": total_tokens,
            "fallback_frequency": fallback_count / max(1, total_decisions),
            "actions_per_second": 1000.0 / max(0.001, (total_latency_ms / max(1, total_decisions))),
        }

    return results
