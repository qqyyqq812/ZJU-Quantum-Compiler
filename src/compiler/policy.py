"""
PPO 策略网络 + 训练管线
========================
Proximal Policy Optimization (PPO) 用于学习量子电路路由策略。

简化但完整的 PPO 实现，可直接训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Actor-Critic 策略网络。

    Args:
        obs_dim: 观测空间维度
        n_actions: 动作空间大小
        hidden_dim: 隐藏层维度
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Critic: 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """前向传播。

        Returns:
            (action_distribution, state_value)
        """
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray | None = None) -> tuple[int, float, float]:
        """采样一个动作 (用于数据收集)。

        V3: 支持 action_mask —— 将无效动作 logit 设为 -inf。

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            features = self.shared(obs_t)
            logits = self.actor(features)
            value = self.critic(features).squeeze(-1)

            # V3: Action Masking
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
                logits = logits.masked_fill(mask_t == 0, -1e8)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor,
                 action_masks: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估一批 (obs, action) 对 (用于 PPO 更新)。

        V3: 支持 action_masks 批量处理。

        Returns:
            (log_probs, values, entropy)
        """
        features = self.shared(obs)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)

        if action_masks is not None:
            logits = logits.masked_fill(action_masks == 0, -1e8)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


@dataclass
class RolloutBuffer:
    """存储一个 rollout 的数据。"""
    observations: list
    actions: list
    rewards: list
    log_probs: list
    values: list
    dones: list

    @staticmethod
    def create() -> RolloutBuffer:
        return RolloutBuffer([], [], [], [], [], [])

    def add(self, obs, action, reward, log_prob, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.observations)

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[list, list]:
        """计算 GAE 优势和回报。"""
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0.0
                gae = 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]

        return advantages, returns


class PPOTrainer:
    """PPO 训练器。

    Args:
        policy: 策略网络
        lr: 学习率
        clip_epsilon: PPO clip 范围
        epochs_per_update: 每次数据收集后的训练轮数
        entropy_coef: 熵正则化系数
        value_coef: 价值损失系数
        max_grad_norm: 梯度裁剪
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        epochs_per_update: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """用收集的数据更新策略。

        Returns:
            训练指标 dict
        """
        advantages, returns = buffer.compute_returns()

        # 转为 tensor
        obs = torch.FloatTensor(np.array(buffer.observations))
        actions = torch.LongTensor(buffer.actions)
        old_log_probs = torch.FloatTensor(buffer.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # 标准化优势
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs_per_update):
            log_probs, values, entropy = self.policy.evaluate(obs, actions)

            # Policy loss (PPO clip)
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns_t)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n = self.epochs_per_update
        return {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
        }

    def save(self, path: str) -> None:
        """保存模型。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """加载模型。"""
        self.policy.load_state_dict(torch.load(path, weights_only=True))
