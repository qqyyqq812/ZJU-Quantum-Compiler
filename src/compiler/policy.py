"""
残差 Actor-Critic 与 Edge-Scoring Attention Head (V7)
=====================================================
彻底重构结构：
1. 双流输入：扁平张量 (obs) + 拓扑感知图 (gnn_input)
2. 边级打分独立 Attention 头 (解决维度灾难和无序性)
3. 独立的 Critic 头 (不共享 Actor 梯度)
4. 多层 LayerNorm 和残差连接
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch

from src.compiler.gnn_encoder import GraphSAGEEncoder


class ResidualMLP(nn.Module):
    """带残差和 LayerNorm 的 MLP 块."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + res)


class PolicyNetwork(nn.Module):
    """V7 芯片级 Actor-Critic 路由策略网络."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # ─── 1. 特征提取器 ───
        # 1.1 GNN 拓扑图特征编码 (提取物理比特特征和全局拓扑特征)
        self.gnn = GraphSAGEEncoder(in_channels=5, hidden_channels=128, out_channels=128)
        
        # 1.2 扁平标量状态特征编码 (obs_dim -> 128)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # ─── 2. 融合模块 ───
        # 全局上下文特征 = obs_feature (128) + graph_mean (128) + graph_max (128) = 384 → 256
        self.global_fusion = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            ResidualMLP(256)
        )
        
        # ─── 3. Action Critic (Value Network) ───
        # 独立网络，不被 Actor 的高频变化干扰
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # ─── 4. Attention Actor Heads ───
        # Edge Scorer: 评估给定的两条物理边的打分
        # 输入: target_node (128) + neighbor_node (128) + global_context (256) = 512 → 1
        self.edge_scorer = nn.Sequential(
            nn.Linear(128 + 128 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # PASS Scorer: 独立的 PASS 动作估值
        # 输入: global_context (256) → 1
        self.pass_scorer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs: torch.Tensor, gnn_batch: Batch, swap_edges: list[list[tuple]]) -> tuple[Categorical, torch.Tensor]:
        """批量前向提取。
        
        Args:
            obs: (B, obs_dim)
            gnn_batch: 包含物理图特征的 PyG Data Batch
            swap_edges: 列表，长度为 B，每项为当前环境内所有的候选 SWAP edges (u, v)
        """
        B = obs.size(0)
        
        # 1. GNN Embeddings
        # node_embed: (N_total, 128), global_graph_embed: (B, 256)
        node_embed, global_graph_embed = self.gnn(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch)
        
        # 2. Obs Embeddings
        obs_embed = self.obs_encoder(obs)  # (B, 128)
        
        # 3. Global Context
        global_context = self.global_fusion(torch.cat([obs_embed, global_graph_embed], dim=-1)) # (B, 256)
        
        # 4. Values (Critic)
        values = self.critic(global_context).squeeze(-1) # (B)
        
        # 5. Edge Action Scores (Actor)
        # 为每批次独立构建动作分布
        logits_list = []
        node_split = node_embed.split(gnn_batch.ptr.diff().tolist()) # 拆分出每个子图的节点特征 [N_nodes x 128] * B
        
        for i in range(B):
            n_features = node_split[i] # (N_phys, 128)
            ctx = global_context[i]    # (256)
            edges = swap_edges[i]      # [(u,v), ...]
            
            # PASS Score
            pass_score = self.pass_scorer(ctx).unsqueeze(0) # (1, 1)
            
            if not edges: # 异常情况，返回全零
                logits = torch.zeros(self.n_actions, device=obs.device)
                logits_list.append(logits)
                continue
                
            # Score each edge
            # edge_src/dst: (NumEdges, 128)
            src_idx = [e[0] for e in edges]
            dst_idx = [e[1] for e in edges]
            
            # 有时可能边索引超出实际图节点数(脏数据/未及时reset)
            num_nodes = n_features.size(0)
            src_idx = [min(idx, num_nodes-1) for idx in src_idx]
            dst_idx = [min(idx, num_nodes-1) for idx in dst_idx]
            
            src_feats = n_features[src_idx]
            dst_feats = n_features[dst_idx]
            
            ctx_expand = ctx.unsqueeze(0).expand(len(edges), -1) # (NumEdges, 256)
            edge_input = torch.cat([src_feats, dst_feats, ctx_expand], dim=-1) # (NumEdges, 512)
            
            scores = self.edge_scorer(edge_input).squeeze(-1) # (NumEdges)
            
            # Combine [scores, pass_score]
            final_scores = torch.cat([scores, pass_score.squeeze(-1)]) # (NumEdges + 1)
            
            # Pad to max n_actions dimension
            padded_scores = torch.full((self.n_actions,), -1e8, device=obs.device)
            padded_scores[:len(final_scores)] = final_scores
            
            logits_list.append(padded_scores)
            
        logits_stacked = torch.stack(logits_list) # (B, n_actions)
        dist = Categorical(logits=logits_stacked)
        
        return dist, values

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray | None = None, gnn_input: dict | None = None) -> tuple[int, float, float]:
        """单步推理 (Rollout使用)"""
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            if gnn_input is None or 'graph' not in gnn_input:
                # Fallback purely on random if not properly wired (safeguard)
                logits = torch.zeros((1, self.n_actions), device=device)
                values = torch.zeros(1, device=device)
            else:
                d_b = Batch.from_data_list([gnn_input['graph']]).to(device)
                swap_edges = [gnn_input['swap_edges']]
                dist, values = self.forward(obs_t, d_b, swap_edges)
                logits = dist.logits
                
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)
                logits = logits.masked_fill(mask_t == 0, -1e8)
                
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), values.item()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, 
                 action_masks: torch.Tensor | None = None, 
                 gnn_inputs: list[dict] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量评估 (PPO Update使用)"""
        if gnn_inputs is None or len(gnn_inputs) == 0 or gnn_inputs[0] is None:
            raise ValueError("V7 requires GNN inputs for training!")
            
        graphs = [gi['graph'] for gi in gnn_inputs]
        edges = [gi['swap_edges'] for gi in gnn_inputs]
        
        d_b = Batch.from_data_list(graphs).to(obs.device)
        
        dist, values = self.forward(obs, d_b, edges)
        logits = dist.logits
        
        if action_masks is not None:
            logits = logits.masked_fill(action_masks == 0, -1e8)
            dist = Categorical(logits=logits)
            
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


# ====================================================================
# PPO Trainer 等代码不需要大改，只需要把 update() 方法中的 gnn_inputs 透传。
# ====================================================================

from dataclasses import dataclass

@dataclass
class RolloutBuffer:
    observations: list
    actions: list
    rewards: list
    log_probs: list
    values: list
    dones: list
    gnn_inputs: list

    @staticmethod
    def create() -> 'RolloutBuffer':
        return RolloutBuffer([], [], [], [], [], [], [])

    def add(self, obs, action, reward, log_prob, value, done, gnn_input=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.gnn_inputs.append(gnn_input)

    def __len__(self):
        return len(self.observations)

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95):
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
    def __init__(self, policy: PolicyNetwork, lr: float = 3e-4, clip_epsilon: float = 0.2, 
                 epochs_per_update: int = 4, entropy_coef: float = 0.01, 
                 value_coef: float = 0.5, max_grad_norm: float = 0.5):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer: RolloutBuffer):
        advantages, returns = buffer.compute_returns()

        device = next(self.policy.parameters()).device
        obs = torch.tensor(np.array(buffer.observations), dtype=torch.float32, device=device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs_per_update):
            # 将 buffer.gnn_inputs 直接传给 evaluate
            log_probs, values, entropy = self.policy.evaluate(
                obs=obs, actions=actions, action_masks=None, gnn_inputs=buffer.gnn_inputs
            )

            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns_t)
            entropy_loss = -entropy.mean()

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

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        device = next(self.policy.parameters()).device
        self.policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
