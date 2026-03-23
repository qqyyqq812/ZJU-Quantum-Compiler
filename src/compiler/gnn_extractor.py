"""
统一的拓扑感知图特征提取器 (V7)
===============================
将当前电路路由状态编码为以物理量子比特为节点的单一图结构。
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from qiskit.transpiler import CouplingMap
from src.compiler.dag import CircuitDAG

def extract_physical_graph(
    coupling_map: CouplingMap,
    mapping: dict[int, int],
    dag: CircuitDAG | None
) -> Data:
    """提取基于物理拓扑的统一图表征。
    
    图定义:
        节点: 物理量子比特 (0 to N-1)
        边: CouplingMap 的物理连接
        节点特征 (5维):
            [
                0: 物理节点的度数 (局部连接维度)
                1: 是否被占用 (1 或 0)
                2: 映射的逻辑比特索引 (-1 表示未映射)
                3: 是否参与当前前沿双比特门 (1 或 0)
                4: 是否参与 Look-ahead 扩展前沿门 (1 或 0)
            ]
    """
    n_phys = coupling_map.size()
    
    # 构建边
    edges = list(coupling_map.get_edges())
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
    # 构建节点特征
    x = torch.zeros((n_phys, 5), dtype=torch.float)
    
    # 特征 0: Degree
    for u, v in edges:
        x[u, 0] += 1
        x[v, 0] += 1
        
    # 特征 1, 2: Occupancy & Logical ID
    phys_to_logical = {p: l for l, p in mapping.items()}
    for p in range(n_phys):
        if p in phys_to_logical:
            x[p, 1] = 1.0
            x[p, 2] = float(phys_to_logical[p])
        else:
            x[p, 1] = 0.0
            x[p, 2] = -1.0
            
    # 特征 3, 4: Front Gate & Extended Front Gate
    if dag is not None and not dag.is_done():
        # Front
        for gate in dag.get_two_qubit_front():
            p0 = mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < n_phys: x[p0, 3] = 1.0
            if p1 < n_phys: x[p1, 3] = 1.0
            
        # Extended Front (Look-ahead)
        front_ids = {g.gate_id for g in dag.get_two_qubit_front()}
        for gate in dag.get_extended_front(depth=2):
            if gate.gate_id not in front_ids:
                p0 = mapping.get(gate.qubits[0], gate.qubits[0])
                p1 = mapping.get(gate.qubits[1], gate.qubits[1])
                if p0 < n_phys: x[p0, 4] = 1.0
                if p1 < n_phys: x[p1, 4] = 1.0
                
    return Data(x=x, edge_index=edge_index)
