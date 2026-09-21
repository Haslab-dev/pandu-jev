"""Behavioral Cloning (Imitation Learning) training pipeline for Pandu (pandu-jev).

Trains the TinyPolicy on expert A* demonstrations using cross-entropy loss,
evaluates closed-loop policy rollout performance, and optimizes temperature calibration.
"""

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from env.gridworld import GridWorld, Action
from models.policy import TinyPolicy


def train_behavioral_cloning(
    model: nn.Module,
    features: np.ndarray,
    actions: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train policy network via supervised behavioral cloning on expert actions."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(actions, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                v_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == batch_y).sum().item()
                v_total += len(batch_y)

        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    training_time = time.perf_counter() - start_time
    model.eval()

    # Optimize Temperature Scaling for post-hoc calibration
    optimal_t = fit_temperature(model, val_loader, device)
    if hasattr(model, "temperature"):
        model.temperature.data.fill_(optimal_t)

    return {
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "optimal_temperature": optimal_t,
        "training_time_seconds": training_time,
        "history": history,
    }


def fit_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Find scalar temperature T > 0 that minimizes NLL on validation set (Guo et al. 2017)."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            logits_list.append(logits.cpu())
            labels_list.append(by)

    if not logits_list:
        return 1.0

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Temperature parameter
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return float(temperature.item())


def evaluate_policy_closed_loop(
    model: nn.Module,
    num_episodes: int = 100,
    map_types: str = "default",
    device: Optional[torch.device] = None,
    seed: int = 123,
) -> Dict[str, Any]:
    """Evaluate trained policy in closed-loop environment rollout."""
    if device is None:
        device = torch.device("cpu")

    model.eval()
    model.to(device)

    rng = np.random.RandomState(seed)
    successes = 0
    total_steps = 0
    total_reward = 0.0
    wall_hits = 0
    confidences = []
    latencies = []

    for ep in range(num_episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        if map_types == "random":
            from env.gridworld import create_random_gridworld
            env = create_random_gridworld(width=12, height=7, wall_prob=0.18, seed=ep_seed)
        else:
            env = GridWorld(random_start_goal=True, seed=ep_seed)

        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and ep_steps < env.max_steps:
            feat = env.get_feature_vector()

            t0 = time.perf_counter()
            dist = model.get_action_distribution(feat)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

            action = dist["action_idx"]
            confidences.append(dist["confidence"])

            obs, r, done, info = env.step(action)
            ep_reward += r
            ep_steps += 1
            if info.get("hit_wall", False):
                wall_hits += 1
            if info.get("reached_goal", False):
                successes += 1

        total_steps += ep_steps
        total_reward += ep_reward

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "episodes": num_episodes,
        "success_rate": successes / max(1, num_episodes),
        "avg_steps": total_steps / max(1, num_episodes),
        "avg_reward": total_reward / max(1, num_episodes),
        "avg_wall_hits": wall_hits / max(1, num_episodes),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
    }
