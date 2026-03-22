"""
GNN 量子电路编码器
==================
用图注意力网络 (GAT) 对量子电路 DAG 和 Coupling Map 进行编码。

输出: 固定维度的嵌入向量，供 RL 策略网络使用。

架构:
  电路 DAG → GAT → 图级嵌入 (mean pooling)
  Coupling Map → GAT → 图级嵌入 (mean pooling)
  拼接 → MLP → 融合嵌入
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch


class GATEncoder(nn.Module):
    """GAT 图编码器: 将图数据编码为固定维度嵌入。

    Args:
        in_channels: 节点特征维度
        hidden_channels: 隐藏层维度
        out_channels: 输出嵌入维度
        num_layers: GAT 层数
        heads: 注意力头数
        dropout: Dropout 比率
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # 最后一层 (单头)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.norms.append(nn.LayerNorm(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: 节点特征 (N, in_channels)
            edge_index: 边索引 (2, E)
            batch: 批次索引 (N,)，用于多图批处理
        Returns:
            图级嵌入 (B, out_channels) 或节点嵌入 (N, out_channels)
        """
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)

        # 图级池化
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x


class CircuitEncoder(nn.Module):
    """量子电路编译器的双图编码器。

    将电路 DAG 和 Coupling Map 分别编码后融合。

    Args:
        dag_node_features: DAG 节点特征维度 (from dag.get_node_features())
        coupling_node_features: CouplingMap 节点特征维度
        hidden_dim: 隐藏层维度
        embed_dim: 最终嵌入维度
    """

    def __init__(
        self,
        dag_node_features: int = 20,
        coupling_node_features: int = 4,
        hidden_dim: int = 64,
        embed_dim: int = 64,
    ):
        super().__init__()

        self.dag_encoder = GATEncoder(
            in_channels=dag_node_features,
            hidden_channels=hidden_dim,
            out_channels=embed_dim // 2,
        )

        self.coupling_encoder = GATEncoder(
            in_channels=coupling_node_features,
            hidden_channels=hidden_dim // 2,
            out_channels=embed_dim // 2,
            num_layers=2,
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, dag_data: Data, coupling_data: Data) -> torch.Tensor:
        """编码电路 DAG + Coupling Map → 融合嵌入。

        Args:
            dag_data: DAG 的 PyG Data (x, edge_index, batch)
            coupling_data: CouplingMap 的 PyG Data
        Returns:
            融合嵌入 (B, embed_dim)
        """
        dag_embed = self.dag_encoder(dag_data.x, dag_data.edge_index,
                                     dag_data.batch if hasattr(dag_data, 'batch') else None)
        coupling_embed = self.coupling_encoder(coupling_data.x, coupling_data.edge_index,
                                               coupling_data.batch if hasattr(coupling_data, 'batch') else None)

        # 确保维度匹配
        if dag_embed.dim() == 1:
            dag_embed = dag_embed.unsqueeze(0)
        if coupling_embed.dim() == 1:
            coupling_embed = coupling_embed.unsqueeze(0)

        fused = torch.cat([dag_embed, coupling_embed], dim=-1)
        return self.fusion(fused)
