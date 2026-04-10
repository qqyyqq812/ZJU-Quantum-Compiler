"""
Topology-Aware GraphSAGE 编码器 (V13 — 纯 PyTorch 版)
====================================
V13: 节点特征 9维, 完全去除 torch_geometric 依赖
用纯 PyTorch 实现 SAGEConv + global pooling
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureSAGEConv(nn.Module):
    """纯 PyTorch 实现的 GraphSAGE 卷积层。

    SAGE(v) = W · CONCAT(h_v, MEAN(h_u for u in N(v)))
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) 节点特征
            edge_index: (2, E) 边索引 [src, dst]
        Returns:
            (N, out_channels)
        """
        n_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # 聚合邻居特征 (mean aggregation) — 避免 in-place 操作
        src_feats = x[src]  # (E, in_channels)
        dst_expanded = dst.unsqueeze(-1).expand_as(src_feats)  # (E, in_channels)
        neighbor_sum = torch.zeros(n_nodes, x.size(1), device=x.device, dtype=x.dtype).scatter_add(0, dst_expanded, src_feats)
        ones = torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype)
        neighbor_count = torch.zeros(n_nodes, 1, device=x.device, dtype=x.dtype).scatter_add(0, dst.unsqueeze(-1), ones)
        neighbor_count = neighbor_count.clamp(min=1)
        neighbor_mean = neighbor_sum / neighbor_count

        # CONCAT(self, neighbor_mean) → linear
        out = self.linear(torch.cat([x, neighbor_mean], dim=-1))
        return out


class GraphSAGEEncoder(nn.Module):
    """3层 GraphSAGE，纯 PyTorch 实现。

    Args:
        in_channels: 节点特征的原始维度 (9)
        hidden_channels: 隐藏层维度 (128)
        out_channels: 每个物理节点的输出嵌入维度 (128)
    """

    def __init__(self, in_channels: int = 9, hidden_channels: int = 128, out_channels: int = 128):
        super().__init__()

        self.node_embedder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )

        self.conv1 = PureSAGEConv(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.conv2 = PureSAGEConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.conv3 = PureSAGEConv(hidden_channels, out_channels)
        self.norm3 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        返回:
            node_embed: (N, out_channels)
            graph_embed: (B, out_channels*2) — Mean + Max 池化
        """
        h = self.node_embedder(x)

        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv3(h, edge_index)
        h = self.norm3(h)
        node_embed = F.relu(h)

        if batch is None:
            batch = torch.zeros(node_embed.size(0), dtype=torch.long, device=node_embed.device)

        # Global pooling (mean + max) — 纯 torch 实现，避免 in-place
        num_graphs = int(batch.max().item()) + 1
        batch_expanded = batch.unsqueeze(-1).expand(-1, node_embed.size(1))

        # Mean pooling
        h_sum = torch.zeros(num_graphs, node_embed.size(1), device=node_embed.device, dtype=node_embed.dtype).scatter_add(0, batch_expanded, node_embed)
        count = torch.zeros(num_graphs, 1, device=node_embed.device, dtype=node_embed.dtype).scatter_add(0, batch.unsqueeze(-1), torch.ones(node_embed.size(0), 1, device=node_embed.device, dtype=node_embed.dtype))
        h_mean = h_sum / count.clamp(min=1)

        # Max pooling via scatter_reduce
        h_max = torch.full((num_graphs, node_embed.size(1)), float('-inf'), device=node_embed.device, dtype=node_embed.dtype)
        h_max = h_max.scatter_reduce(0, batch_expanded, node_embed, reduce='amax')

        graph_embed = torch.cat([h_mean, h_max], dim=-1)

        return node_embed, graph_embed
