"""Chess Curriculum Training Pipeline for Pandu Policy.

Trains PanduChessNet across cumulative curriculum tiers:
- Tier 0: Legal Move Mastery
- Tier 1: Material & Tactical Captures
- Tier 2: Checkmate Recognition & Check Defense
- Tier 3: Advanced Tactics (Pins, Forks, Promotions)
- Tier 4: Strategic Positional Play & Teacher Distillation

Maintains cumulative datasets across tiers to prevent catastrophic forgetting.
"""

from dataclasses import dataclass
import os
import time
from typing import Dict, List, Optional, Tuple
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from arena.chess_policy import PanduChessNet, PanduChessPolicy
from datasets.chess_curriculum import (
    ChessCurriculumSample,
    CurriculumTier,
    build_cumulative_curriculum,
    generate_curriculum_samples,
)


@dataclass
class TierTrainingMetrics:
    tier: CurriculumTier
    tier_name: str
    samples_count: int
    epochs: int
    final_loss: float
    target_move_accuracy: float
    mate_in_1_accuracy: float
    intent_accuracy: float
    training_time_sec: float


class ChessCurriculumTrainer:
    """Manages staged curriculum training for Pandu Chess policy."""

    def __init__(
        self,
        policy: Optional[PanduChessPolicy] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy = policy if policy is not None else PanduChessPolicy(device=device)
        self.model = self.policy.model
        self.optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train_single_tier(
        self,
        samples: List[ChessCurriculumSample],
        epochs: int = 5,
        batch_size: int = 32,
        val_samples: Optional[List[ChessCurriculumSample]] = None,
    ) -> Dict[str, float]:
        """Train model on a list of curriculum samples with multi-task loss."""
        self.model.train()
        n = len(samples)

        for epoch in range(epochs):
            indices = list(range(n))
            np.random.shuffle(indices)

            total_loss = 0.0
            total_correct = 0

            for i in range(0, n, batch_size):
                batch_idxs = indices[i : i + batch_size]
                self.optimizer.zero_grad()

                batch_loss = torch.tensor(0.0, device=self.device)
                count = 0

                for idx in batch_idxs:
                    s = samples[idx]
                    x = torch.tensor(s.feature_vector, dtype=torch.float32, device=self.device)
                    legal_logits, value, intent_logits = self.model.score_legal_moves(x, s.legal_moves)

                    if len(legal_logits) == 0:
                        continue

                    # 1. CrossEntropy over legal moves
                    target_idx = torch.tensor(s.target_move_idx, dtype=torch.long, device=self.device)
                    ce_loss = F.cross_entropy(legal_logits.unsqueeze(0), target_idx.unsqueeze(0))

                    # 2. Value prediction loss
                    val_target = torch.tensor([s.target_value], dtype=torch.float32, device=self.device)
                    val_loss = F.mse_loss(value.view(-1), val_target)

                    # 3. Strategic Intent prediction loss
                    intent_target = torch.tensor([s.target_intent], dtype=torch.long, device=self.device)
                    intent_loss = F.cross_entropy(intent_logits.unsqueeze(0), intent_target)

                    # Multi-task total loss
                    loss = ce_loss + 0.25 * val_loss + 0.35 * intent_loss
                    batch_loss = batch_loss + loss

                    if torch.argmax(legal_logits).item() == s.target_move_idx:
                        total_correct += 1
                    count += 1

                if count > 0:
                    (batch_loss / count).backward()
                    self.optimizer.step()
                    total_loss += float(batch_loss.item())

        eval_set = val_samples if val_samples is not None else samples
        metrics = self.evaluate_dataset(eval_set)
        metrics["final_loss"] = total_loss / max(1, n)
        return metrics

    def evaluate_dataset(
        self, samples: List[ChessCurriculumSample]
    ) -> Dict[str, float]:
        """Evaluate policy move accuracy, mate-in-1, and strategic intent classification."""
        self.model.eval()
        correct = 0
        intent_correct = 0
        total = len(samples)
        mate_total = 0
        mate_correct = 0

        with torch.no_grad():
            for s in samples:
                x = torch.tensor(s.feature_vector, dtype=torch.float32, device=self.device)
                legal_logits, _, intent_logits = self.model.score_legal_moves(x, s.legal_moves)
                if len(legal_logits) == 0:
                    continue

                pred_idx = int(torch.argmax(legal_logits).item())
                if pred_idx == s.target_move_idx:
                    correct += 1

                pred_intent = int(torch.argmax(intent_logits).item())
                if pred_intent == s.target_intent:
                    intent_correct += 1

                # Check if position has mate-in-1
                if s.tier == CurriculumTier.LEVEL_2_CHECKMATE and s.target_value >= 0.9:
                    mate_total += 1
                    if pred_idx == s.target_move_idx:
                        mate_correct += 1

        acc = (correct / total) if total > 0 else 0.0
        intent_acc = (intent_correct / total) if total > 0 else 0.0
        mate_acc = (mate_correct / mate_total) if mate_total > 0 else 1.0

        return {
            "target_move_accuracy": float(acc),
            "mate_in_1_accuracy": float(mate_acc),
            "intent_accuracy": float(intent_acc),
        }

    def run_curriculum(
        self,
        samples_per_tier: int = 250,
        epochs_per_tier: int = 5,
        max_tier: CurriculumTier = CurriculumTier.LEVEL_8_ENGINE_DISTILLATION,
        seed: int = 42,
        save_path: Optional[str] = "experiments/results/pandu_chess_curriculum.pt",
    ) -> List[TierTrainingMetrics]:
        """Run full 9-phase deep curriculum training pipeline."""
        print(f"=== Starting Pandu Chess Deep Curriculum Training (Up to {max_tier.name}) ===")
        all_metrics: List[TierTrainingMetrics] = []

        curriculum_datasets = build_cumulative_curriculum(
            max_tier=max_tier,
            samples_per_tier=samples_per_tier,
            seed=seed,
        )

        for tier_val in range(int(max_tier) + 1):
            tier = CurriculumTier(tier_val)
            tier_samples = curriculum_datasets[tier]
            print(f"\n--- Training Tier {tier_val}: {tier.name} ({len(tier_samples)} cumulative samples) ---")

            t0 = time.perf_counter()
            res = self.train_single_tier(tier_samples, epochs=epochs_per_tier)
            dt = time.perf_counter() - t0

            metric = TierTrainingMetrics(
                tier=tier,
                tier_name=tier.name,
                samples_count=len(tier_samples),
                epochs=epochs_per_tier,
                final_loss=res["final_loss"],
                target_move_accuracy=res["target_move_accuracy"],
                mate_in_1_accuracy=res["mate_in_1_accuracy"],
                intent_accuracy=res["intent_accuracy"],
                training_time_sec=dt,
            )
            all_metrics.append(metric)

            print(
                f"Completed {tier.name} in {dt:.2f}s | "
                f"Move Acc: {metric.target_move_accuracy*100:.1f}% | "
                f"Intent Acc: {metric.intent_accuracy*100:.1f}% | "
                f"Mate-in-1: {metric.mate_in_1_accuracy*100:.1f}% | "
                f"Loss: {metric.final_loss:.4f}"
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"\nSaved trained curriculum model weights to: {save_path}")

        return all_metrics
