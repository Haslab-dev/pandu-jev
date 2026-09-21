"""Proximal Policy Optimization (PPO) training pipeline (Phase 7).

Trains the policy directly through environment interaction without expert demonstrations,
allowing direct comparison between Imitation Learning (BC) and Reinforcement Learning (RL).
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env.gridworld import GridWorld


class ActorCritic(nn.Module):
    """Actor-Critic network matching the tiny parameter budget."""

    def __init__(self, input_dim: int = 16, num_actions: int = 4, hidden_dims: Tuple[int, int] = (32, 64)):
        super().__init__()
        # Shared trunk or separate
        self.actor_fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.actor_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.actor_out = nn.Linear(hidden_dims[1], num_actions)

        self.critic_fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.critic_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.critic_out = nn.Linear(hidden_dims[1], 1)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.actor_fc1(x))
        h = F.gelu(self.actor_fc2(h))
        return self.actor_out(h)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.critic_fc1(x))
        h = F.gelu(self.critic_fc2(h))
        return self.critic_out(h)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward_actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        value = self.critic_out(F.gelu(self.critic_fc2(F.gelu(self.critic_fc1(x)))))
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimator (GAE) and Returns."""
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_val = 0.0 if dones[t] or t == len(rewards) - 1 else values[t + 1]
        next_done = dones[t]
        delta = rewards[t] + gamma * next_val * (1.0 - float(next_done)) - values[t]
        gae = delta + gamma * lam * (1.0 - float(next_done)) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def train_ppo(
    total_episodes: int = 500,
    steps_per_batch: int = 1024,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    lr: float = 1e-3,
    train_iters: int = 4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = 42,
) -> Tuple[ActorCritic, Dict[str, Any]]:
    """Train policy on GridWorld via PPO."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = GridWorld(random_start_goal=True, seed=seed)
    model = ActorCritic(input_dim=env.feature_dim, num_actions=env.action_space_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {
        "episode_rewards": [],
        "success_rates": [],
        "policy_losses": [],
        "value_losses": [],
    }

    recent_rewards: List[float] = []
    recent_successes: List[int] = []

    steps_collected = 0
    episodes_completed = 0

    while episodes_completed < total_episodes:
        # Buffer for rollout batch
        b_obs: List[np.ndarray] = []
        b_actions: List[int] = []
        b_logprobs: List[float] = []
        b_rewards: List[float] = []
        b_values: List[float] = []
        b_dones: List[bool] = []

        batch_steps = 0

        while batch_steps < steps_per_batch and episodes_completed < total_episodes:
            obs = env.reset()
            ep_reward = 0.0
            done = False
            ep_steps = 0

            while not done and ep_steps < env.max_steps:
                feat = env.get_feature_vector()
                feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action, logprob, _, value = model.get_action_and_value(feat_t)

                act_int = int(action.item())
                next_obs, reward, done, info = env.step(act_int)

                b_obs.append(feat)
                b_actions.append(act_int)
                b_logprobs.append(float(logprob.item()))
                b_rewards.append(reward)
                b_values.append(float(value.item()))
                b_dones.append(done)

                ep_reward += reward
                ep_steps += 1
                batch_steps += 1

            episodes_completed += 1
            recent_rewards.append(ep_reward)
            recent_successes.append(1 if info.get("reached_goal", False) else 0)

        # Compute GAE on batch
        advantages, returns = compute_gae(b_rewards, b_values, b_dones, gamma=gamma, lam=lam)

        # Convert to tensors
        t_obs = torch.tensor(np.array(b_obs), dtype=torch.float32)
        t_actions = torch.tensor(b_actions, dtype=torch.long)
        t_old_logprobs = torch.tensor(b_logprobs, dtype=torch.float32)
        t_advantages = torch.tensor(advantages, dtype=torch.float32)
        t_returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        if len(t_advantages) > 1:
            t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

        # PPO update epochs
        batch_policy_loss = 0.0
        batch_value_loss = 0.0

        for _ in range(train_iters):
            _, logprob, entropy, values = model.get_action_and_value(t_obs, t_actions)
            ratio = torch.exp(logprob - t_old_logprobs)

            surr1 = ratio * t_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * t_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, t_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            batch_policy_loss += policy_loss.item()
            batch_value_loss += value_loss.item()

        window = 50
        avg_rew = float(np.mean(recent_rewards[-window:]))
        avg_succ = float(np.mean(recent_successes[-window:]))

        history["episode_rewards"].append(avg_rew)
        history["success_rates"].append(avg_succ)
        history["policy_losses"].append(batch_policy_loss / train_iters)
        history["value_losses"].append(batch_value_loss / train_iters)

    return model, {
        "final_avg_reward": history["episode_rewards"][-1] if history["episode_rewards"] else 0.0,
        "final_success_rate": history["success_rates"][-1] if history["success_rates"] else 0.0,
        "total_episodes": episodes_completed,
        "history": history,
    }
