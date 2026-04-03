"""
V11 图强化学习 DQN 策略核心
=============================
基于 Deep Q-Network 和图神经网络，具备经验回放与目标网络更新。
专治高维、强约束且死锁频繁的图表搜索问题。
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from collections import deque
import copy

from src.compiler.gnn_encoder import GraphSAGEEncoder
from src.compiler.policy import ResidualMLP


class DQNNetwork(nn.Module):
    """V11 DQN: 预估在当前拓扑状态下执行某个动作所需剩余代价(负向)"""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # 1. 图编码
        self.gnn = GraphSAGEEncoder(in_channels=5, hidden_channels=128, out_channels=128)
        
        # 2. 扁平状态编码
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 3. 维度融合
        self.global_fusion = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            ResidualMLP(256)
        )
        
        # 4. Q-Value Heads (与 Actor Head 相似，但输出 Q 值)
        self.edge_q_scorer = nn.Sequential(
            nn.Linear(128 + 128 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.pass_q_scorer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs: torch.Tensor, gnn_batch: Batch, swap_edges: list[list[tuple]]) -> torch.Tensor:
        """输出所有候选动作的 Q 值。屏蔽非合法边界为极小值。"""
        B = obs.size(0)
        
        node_embed, global_graph_embed = self.gnn(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch)
        obs_embed = self.obs_encoder(obs)
        global_context = self.global_fusion(torch.cat([obs_embed, global_graph_embed], dim=-1))
        
        q_list = []
        node_split = node_embed.split(gnn_batch.ptr.diff().tolist())
        
        for i in range(B):
            n_features = node_split[i]
            ctx = global_context[i]
            edges = swap_edges[i]
            
            pass_q = self.pass_q_scorer(ctx).unsqueeze(0)
            
            if not edges:
                logits = torch.zeros(self.n_actions, device=obs.device)
                q_list.append(logits)
                continue
                
            src_idx = [min(e[0], n_features.size(0)-1) for e in edges]
            dst_idx = [min(e[1], n_features.size(0)-1) for e in edges]
            
            src_feats = n_features[src_idx]
            dst_feats = n_features[dst_idx]
            
            ctx_expand = ctx.unsqueeze(0).expand(len(edges), -1)
            edge_input = torch.cat([src_feats, dst_feats, ctx_expand], dim=-1)
            edge_q = self.edge_q_scorer(edge_input).squeeze(-1)
            
            final_q = torch.cat([edge_q, pass_q.squeeze(-1)])
            
            padded_q = torch.full((self.n_actions,), -1e8, device=obs.device)
            padded_q[:len(final_q)] = final_q
            
            q_list.append(padded_q)
            
        return torch.stack(q_list)

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray | None, gnn_input: dict, epsilon: float = 0.0) -> tuple[int, float]:
        """Epsilon-Greedy 获取最佳动作"""
        with torch.no_grad():
            device = next(self.parameters()).device
            if random.random() < epsilon:
                # 随机探索
                valid_actions = np.where(action_mask > 0)[0] if action_mask is not None else np.arange(self.n_actions)
                if len(valid_actions) == 0:
                    return 0, 0.0
                return int(np.random.choice(valid_actions)), 0.0
            
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            d_b = Batch.from_data_list([gnn_input['graph']]).to(device)
            edges = [gnn_input['swap_edges']]
            
            q_values = self.forward(obs_t, d_b, edges).squeeze(0)  # (n_actions,)
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                q_values = q_values.masked_fill(mask_t == 0, -1e8)
                
            action = torch.argmax(q_values).item()
            return action, q_values[action].item()


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, next_obs, done, action_mask, next_action_mask, gnn_input, next_gnn_input):
        self.buffer.append((obs, action, float(reward), next_obs, bool(done), 
                           action_mask, next_action_mask, gnn_input, next_gnn_input))
                           
    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, q_net: DQNNetwork, lr: float = 1e-4, gamma: float = 0.99, tau: float = 0.005, max_grad_norm: float = 1.0):
        self.q_net = q_net
        self.target_net = copy.deepcopy(q_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

    def update(self, buffer: ReplayBuffer, batch_size: int) -> float:
        if len(buffer) < batch_size:
            return 0.0
            
        device = next(self.q_net.parameters()).device
        transitions = buffer.sample(batch_size)
        
        obs_b = []
        action_b = []
        reward_b = []
        next_obs_b = []
        done_b = []
        next_mask_b = []
        gnn_b = []
        next_gnn_b = []
        
        for obs, action, reward, next_obs, done, _, next_mask, gnn, next_gnn in transitions:
            obs_b.append(obs)
            action_b.append(action)
            reward_b.append(reward)
            next_obs_b.append(next_obs)
            done_b.append(done)
            next_mask_b.append(next_mask)
            gnn_b.append(gnn)
            next_gnn_b.append(next_gnn)
            
        # 转换为张量
        obs_t = torch.tensor(np.array(obs_b), dtype=torch.float32, device=device)
        action_t = torch.tensor(action_b, dtype=torch.long, device=device).unsqueeze(-1)
        reward_t = torch.tensor(reward_b, dtype=torch.float32, device=device).unsqueeze(-1)
        next_obs_t = torch.tensor(np.array(next_obs_b), dtype=torch.float32, device=device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=device).unsqueeze(-1)
        
        # 构建图 Batch
        curr_graphs = [g['graph'] for g in gnn_b]
        curr_edges = [g['swap_edges'] for g in gnn_b]
        curr_batch = Batch.from_data_list(curr_graphs).to(device)
        
        next_graphs = [g['graph'] for g in next_gnn_b]
        next_edges = [g['swap_edges'] for g in next_gnn_b]
        next_batch = Batch.from_data_list(next_graphs).to(device)
        
        # 计算当前 Q 值
        curr_q_all = self.q_net(obs_t, curr_batch, curr_edges)
        curr_q = curr_q_all.gather(1, action_t)
        
        # 计算目标 Q 值 (Double DQN)
        with torch.no_grad():
            next_mask_t = torch.tensor(np.array(next_mask_b), dtype=torch.float32, device=device)
            # 使用主网络选择最佳下一动作
            next_q_main_all = self.q_net(next_obs_t, next_batch, next_edges)
            next_q_main_all = next_q_main_all.masked_fill(next_mask_t == 0, -1e8)
            best_next_actions = next_q_main_all.argmax(dim=1, keepdim=True)
            
            # 使用目标网络评估该动作的 Q 值
            next_q_target_all = self.target_net(next_obs_t, next_batch, next_edges)
            next_q = next_q_target_all.gather(1, best_next_actions)
            
            target_q = reward_t + (1 - done_t) * self.gamma * next_q
            
        # 损失与更新
        loss = F.smooth_l1_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Soft update Target Net
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()
        
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'q_net': self.q_net.state_dict()}, path)
        
    def load(self, path: str):
        device = next(self.q_net.parameters()).device
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['q_net'])
