import torch
from torch_geometric.data import Data
import numpy as np
from src.compiler.dag import CircuitDAG
from qiskit.transpiler import CouplingMap

def extract_coupling_data(coupling_map: CouplingMap) -> Data:
    """提取拓扑图数据供 GNN。"""
    edges = list(coupling_map.get_edges())
    if not edges:
        return Data(x=torch.zeros((coupling_map.size(), 4)), edge_index=torch.empty((2, 0), dtype=torch.long))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # 简单的物理节点特征: [degree, 0, 0, 0]
    degrees = [0] * coupling_map.size()
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    x = torch.zeros((coupling_map.size(), 4), dtype=torch.float)
    for i, d in enumerate(degrees):
        x[i, 0] = d
        
    return Data(x=x, edge_index=edge_index)

def extract_dag_data(dag: CircuitDAG) -> Data:
    """提取 DAG 数据供 GNN。"""
    x_np = dag.get_node_features()
    if x_np.shape[0] == 0:
        return Data(x=torch.zeros((1, 20)), edge_index=torch.empty((2, 0), dtype=torch.long))
        
    x = torch.tensor(x_np, dtype=torch.float)
    # x 维度需要匹配 CircuitEncoder 默认值 (20)
    # 目前 dag.get_node_features() 可能是不同的维度，自动 pad 到 20
    if x.shape[1] < 20:
        pad = torch.zeros((x.shape[0], 20 - x.shape[1]))
        x = torch.cat([x, pad], dim=1)
    elif x.shape[1] > 20:
        x = x[:, :20]
        
    edges = []
    # 重构未执行节点的边
    active_gids = [gid for gid, node in dag._gates.items() if not node.executed]
    active_nodes = set(active_gids)
    gid_to_idx = {gid: i for i, gid in enumerate(active_gids)}
    
    for u, v in dag._graph.edges():
        if u in active_nodes and v in active_nodes:
            edges.append([gid_to_idx[u], gid_to_idx[v]])
            
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
    return Data(x=x, edge_index=edge_index)
