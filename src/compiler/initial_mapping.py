"""
初始映射优化 (V3)
=================
用交互图 × 距离矩阵的贪心匹配替代 identity 映射。

思路:
1. 从 DAG 提取逻辑比特交互图 (频繁交互的比特对)
2. 用拓扑距离矩阵评估映射质量
3. 贪心: 交互最频繁的比特对 → 物理距离最短的位置
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.dag import CircuitDAG


def compute_initial_mapping(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
) -> dict[int, int]:
    """用交互图贪心匹配计算初始映射。

    Returns:
        逻辑比特 → 物理比特的映射
    """
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()

    if n_logical <= 1:
        return {i: i for i in range(n_logical)}

    dag = CircuitDAG(circuit)
    interaction = dag.get_interaction_graph()

    # 距离矩阵
    dist = np.zeros((n_physical, n_physical))
    for i in range(n_physical):
        for j in range(n_physical):
            try:
                dist[i][j] = coupling_map.distance(i, j)
            except Exception:
                dist[i][j] = n_physical

    # 按交互频率排序逻辑比特对
    pairs = []
    for u, v, data in interaction.edges(data=True):
        pairs.append((u, v, data.get('weight', 1)))
    pairs.sort(key=lambda x: -x[2])  # 高交互优先

    # 贪心分配: 找物理距离最短的位置
    mapping: dict[int, int] = {}
    used_physical: set[int] = set()

    if pairs:
        # 第一对: 放在距离最短的物理对上
        best_pair = None
        best_dist = float('inf')
        edges = list(set(tuple(sorted(e)) for e in coupling_map.get_edges()))
        for p1, p2 in edges:
            if dist[p1][p2] < best_dist:
                best_dist = dist[p1][p2]
                best_pair = (p1, p2)

        if best_pair:
            u, v = pairs[0][0], pairs[0][1]
            mapping[u] = best_pair[0]
            mapping[v] = best_pair[1]
            used_physical.update(best_pair)

    # 剩余逻辑比特: 按交互顺序贪心放置
    for pair in pairs:
        for q in [pair[0], pair[1]]:
            if q in mapping:
                continue
            # 找离已放置比特最近的物理位置
            best_p = None
            best_cost = float('inf')
            for p in range(n_physical):
                if p in used_physical:
                    continue
                cost = sum(dist[p][mapping[mapped_q]]
                           for mapped_q in mapping if mapped_q != q)
                if cost < best_cost:
                    best_cost = cost
                    best_p = p
            if best_p is not None:
                mapping[q] = best_p
                used_physical.add(best_p)

    # 放置剩余未分配的逻辑比特
    for q in range(n_logical):
        if q not in mapping:
            for p in range(n_physical):
                if p not in used_physical:
                    mapping[q] = p
                    used_physical.add(p)
                    break

    return mapping
