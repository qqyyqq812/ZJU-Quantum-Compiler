"""Neural policy network used by the public NPQR runtime."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.compiler.gnn_extractor import GraphBatch

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
    """Graph-aware routing policy for SWAP action scoring."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 512):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        self.gnn = GraphSAGEEncoder(in_channels=9, hidden_channels=256, out_channels=256)
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        self.global_fusion = nn.Sequential(
            nn.Linear(256 + 512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            ResidualMLP(512),
            ResidualMLP(512)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(256 + 256 + 512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.pass_scorer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor, gnn_batch: GraphBatch, swap_edges: list[list[tuple]]) -> tuple[Categorical, torch.Tensor]:
        """Return an action distribution and scalar state value."""
        B = obs.size(0)
        
        node_embed, global_graph_embed = self.gnn(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch)
        obs_embed = self.obs_encoder(obs)
        global_context = self.global_fusion(torch.cat([obs_embed, global_graph_embed], dim=-1))
        values = self.critic(global_context).squeeze(-1)
        logits_list = []
        node_split = node_embed.split(gnn_batch.ptr.diff().tolist())
        
        for i in range(B):
            n_features = node_split[i]
            ctx = global_context[i]
            edges = swap_edges[i]
            
            pass_score = self.pass_scorer(ctx).unsqueeze(0)
            
            if not edges:
                logits = torch.zeros(self.n_actions, device=obs.device)
                logits_list.append(logits)
                continue
                
            src_idx = [e[0] for e in edges]
            dst_idx = [e[1] for e in edges]
            
            num_nodes = n_features.size(0)
            src_idx = [min(idx, num_nodes-1) for idx in src_idx]
            dst_idx = [min(idx, num_nodes-1) for idx in dst_idx]
            
            src_feats = n_features[src_idx]
            dst_feats = n_features[dst_idx]
            
            ctx_expand = ctx.unsqueeze(0).expand(len(edges), -1)
            edge_input = torch.cat([src_feats, dst_feats, ctx_expand], dim=-1)
            
            scores = self.edge_scorer(edge_input).squeeze(-1)
            
            final_scores = torch.cat([scores, pass_score.squeeze(-1)])
            
            padded_scores = torch.full((self.n_actions,), -1e8, device=obs.device)
            padded_scores[:len(final_scores)] = final_scores
            
            logits_list.append(padded_scores)
            
        logits_stacked = torch.stack(logits_list)
        dist = Categorical(logits=logits_stacked)
        
        return dist, values

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray | None = None,
        gnn_input: dict | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Return one sampled or greedy action for a single state."""
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            if gnn_input is None or 'graph' not in gnn_input:
                logits = torch.zeros((1, self.n_actions), device=device)
                values = torch.zeros(1, device=device)
            else:
                d_b = GraphBatch.from_data_list([gnn_input['graph']]).to(device)
                swap_edges = [gnn_input['swap_edges']]
                dist, values = self.forward(obs_t, d_b, swap_edges)
                logits = dist.logits
                
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)
                logits = logits.masked_fill(mask_t == 0, -1e8)
                
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), values.item()
