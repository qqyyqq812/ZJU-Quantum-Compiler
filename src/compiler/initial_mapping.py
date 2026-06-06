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


def _distance_matrix(coupling_map: CouplingMap) -> np.ndarray:
    n_physical = coupling_map.size()
    dist = np.zeros((n_physical, n_physical), dtype=np.float32)
    for i in range(n_physical):
        for j in range(n_physical):
            try:
                dist[i][j] = coupling_map.distance(i, j)
            except Exception:
                dist[i][j] = n_physical
    return dist


def _interaction_pairs(circuit: QuantumCircuit) -> list[tuple[int, int, float]]:
    dag = CircuitDAG(circuit)
    pairs: list[tuple[int, int, float]] = []
    for u, v, data in dag.get_interaction_graph().edges(data=True):
        if u == v:
            continue
        pairs.append((int(u), int(v), float(data.get("weight", 1.0))))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs


def qap_mapping_cost(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    mapping: dict[int, int],
) -> float:
    """Score a mapping by interaction weight times physical distance.

    This is a small QAP-style objective for initial mapping selection only. It
    is not a full QAP solver.
    """
    dist = _distance_matrix(coupling_map)
    cost = 0.0
    for u, v, weight in _interaction_pairs(circuit):
        if u not in mapping or v not in mapping:
            continue
        pu = mapping[u]
        pv = mapping[v]
        if pu >= dist.shape[0] or pv >= dist.shape[1]:
            continue
        cost += weight * float(dist[pu][pv])
    return float(cost)


def selector_mapping_score(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    mapping: dict[int, int],
) -> float:
    """Score an initial mapping for Stage10 selector ranking.

    Lower is better. This uses only circuit/topology structure, not route
    completion labels, SWAP counts, or trace replay outcomes.
    """
    dist = _distance_matrix(coupling_map)
    weighted_distance = 0.0
    max_distance = 0.0
    far_interactions = 0
    for u, v, weight in _interaction_pairs(circuit):
        if u not in mapping or v not in mapping:
            continue
        pu = mapping[u]
        pv = mapping[v]
        if pu >= dist.shape[0] or pv >= dist.shape[1]:
            continue
        distance = float(dist[pu][pv])
        weighted_distance += weight * distance
        max_distance = max(max_distance, distance)
        far_interactions += int(distance > 2)
    return float(weighted_distance + 10.0 * max_distance + 2.0 * far_interactions)


def _is_valid_mapping(mapping: dict[int, int], n_logical: int, n_physical: int) -> bool:
    if set(mapping) != set(range(n_logical)):
        return False
    values = list(mapping.values())
    if len(values) != len(set(values)):
        return False
    return all(0 <= value < n_physical for value in values)


def improve_mapping_qap_local_search(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    mapping: dict[int, int],
    rounds: int = 2,
) -> dict[int, int]:
    """Improve a mapping with a small deterministic logical pair-swap search."""
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()
    best = dict(mapping)
    if rounds <= 0 or not _is_valid_mapping(best, n_logical, n_physical):
        return best

    best_cost = qap_mapping_cost(circuit, coupling_map, best)
    logicals = list(range(n_logical))
    for _ in range(rounds):
        improved = False
        best_round_mapping = best
        best_round_cost = best_cost
        for i, q1 in enumerate(logicals):
            for q2 in logicals[i + 1 :]:
                candidate = dict(best)
                candidate[q1], candidate[q2] = candidate[q2], candidate[q1]
                cost = qap_mapping_cost(circuit, coupling_map, candidate)
                if cost < best_round_cost:
                    best_round_cost = cost
                    best_round_mapping = candidate
                    improved = True
        if not improved:
            break
        best = dict(best_round_mapping)
        best_cost = best_round_cost
    return best


def compute_qap_initial_mapping(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    local_search_rounds: int = 2,
) -> dict[int, int]:
    """Return the best lightweight QAP-style mapping among simple seeds."""
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()
    seeds = [
        {i: i for i in range(n_logical)},
        compute_initial_mapping(circuit, coupling_map),
    ]
    best = seeds[0]
    best_cost = float("inf")
    seen: set[tuple[tuple[int, int], ...]] = set()
    for seed in seeds:
        if not _is_valid_mapping(seed, n_logical, n_physical):
            continue
        candidate = improve_mapping_qap_local_search(
            circuit,
            coupling_map,
            seed,
            rounds=local_search_rounds,
        )
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        cost = qap_mapping_cost(circuit, coupling_map, candidate)
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return dict(best)
