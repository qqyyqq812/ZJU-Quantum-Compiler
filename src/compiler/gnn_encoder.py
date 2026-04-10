"""
Topology-Aware GraphSAGE 编码器 (V13)
====================================
V13: 节点特征从 5维 → 9维 (增加映射距离、DAG深度、前沿目标距离、参与度)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data


class GraphSAGEEncoder(nn.Module):
    """3层 GraphSAGE，专为拓扑感知设计。
    
    Args:
        in_channels: 节点特征的原始维度 (5)
        hidden_channels: 隐藏层维度 (128)
        out_channels: 每个物理节点的输出嵌入维度 (128)
    """
    
    def __init__(self, in_channels: int = 9, hidden_channels: int = 128, out_channels: int = 128):
        super().__init__()
        
        # 将基础特征升维
        self.node_embedder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )
        
        # SAGE 层：能够捕捉拓扑的邻居连接结构
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.norm3 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。
        
        返回:
            node_embed: (N, out_channels) 所有物理节点的嵌入
            graph_embed: (B, out_channels*2) 整个图的全局上下文嵌入 (Mean + Max 池化)
        """
        # Node embedding
        h = self.node_embedder(x)
        
        # Layer 1
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Layer 3
        h = self.conv3(h, edge_index)
        h = self.norm3(h)
        node_embed = F.relu(h)
        
        # Graph Level Readout
        if batch is None:
            # 单图情况，默认全节点同属 batch 0
            batch = torch.zeros(node_embed.size(0), dtype=torch.long, device=node_embed.device)
            
        h_mean = global_mean_pool(node_embed, batch)
        h_max = global_max_pool(node_embed, batch)
        graph_embed = torch.cat([h_mean, h_max], dim=-1)  # (B, out_channels * 2)
        
        return node_embed, graph_embed
