"""
量子电路 DAG 操作模块
=====================
量子电路的有向无环图 (DAG) 表示与操作。

这是整个编译器的核心数据结构，M3 的 GNN 编码器和 RL 环境都依赖此模块。

EDA 类比: DAG ≈ 逻辑网表 (Netlist)
- 节点 = 量子门
- 边 = 数据依赖 (同一比特上的门执行顺序)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


# 门类型到整数的编码 (供 GNN 节点特征使用)
GATE_TYPE_MAP = {
    'h': 0, 'x': 1, 'y': 2, 'z': 3,
    'rx': 4, 'ry': 5, 'rz': 6,
    'sx': 7, 's': 8, 't': 9, 'sdg': 10, 'tdg': 11,
    'cx': 12, 'cz': 13, 'swap': 14,
    'ccx': 15,
    'id': 16, 'barrier': 17,
}


@dataclass
class GateNode:
    """DAG 中的门节点"""
    gate_id: int          # 唯一 ID
    gate_type: str        # 门类型名称 (如 'cx', 'h')
    qubits: tuple[int, ...]  # 操作的逻辑比特
    params: tuple = ()    # 门参数 (如旋转角度)
    executed: bool = False

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) >= 2


class CircuitDAG:
    """量子电路的 DAG 表示。

    核心功能:
    1. 从 QuantumCircuit 构建 DAG
    2. 提取前沿层 (front layer)
    3. 执行门并更新 DAG
    4. 应用 SWAP 并更新映射
    5. 导出为 NetworkX 图 (供 GNN 使用)

    Usage:
        dag = CircuitDAG(circuit)
        while not dag.is_done():
            front = dag.get_front_layer()
            # 选择可执行的门或 SWAP
            dag.execute_gate(gate_id)
    """

    def __init__(self, circuit: QuantumCircuit):
        self._gates: dict[int, GateNode] = {}
        self._graph = nx.DiGraph()
        self._n_qubits = circuit.num_qubits
        self._build_from_circuit(circuit)

    def _build_from_circuit(self, circuit: QuantumCircuit) -> None:
        """从 QuantumCircuit 构建 DAG。"""
        # 跟踪每个比特上最后一个门 (用于建立依赖边)
        last_gate_on_qubit: dict[int, Optional[int]] = {
            q: None for q in range(circuit.num_qubits)
        }

        gate_id = 0
        for instruction in circuit.data:
            op = instruction.operation
            if op.name in ('barrier', 'measure'):
                continue

            qubits = tuple(circuit.qubits.index(q) for q in instruction.qubits)
            params = tuple(float(p) for p in op.params) if op.params else ()

            node = GateNode(
                gate_id=gate_id,
                gate_type=op.name,
                qubits=qubits,
                params=params,
            )
            self._gates[gate_id] = node
            self._graph.add_node(gate_id)

            # 添加依赖边: 同一比特上的前一个门 → 当前门
            for q in qubits:
                prev = last_gate_on_qubit[q]
                if prev is not None:
                    self._graph.add_edge(prev, gate_id)
                last_gate_on_qubit[q] = gate_id

            gate_id += 1

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_gates(self) -> int:
        return len(self._gates)

    def get_front_layer(self) -> list[GateNode]:
        """获取前沿层：所有前驱已执行的门。

        前沿层是 DAG 中入度为 0 (在剩余图中) 的门集合。
        这些门可以立即执行 (如果满足拓扑约束)。

        Returns:
            可执行的 GateNode 列表
        """
        front = []
        for gid, node in self._gates.items():
            if node.executed:
                continue
            # 检查所有前驱是否已执行
            predecessors = list(self._graph.predecessors(gid))
            if all(self._gates[p].executed for p in predecessors):
                front.append(node)
        return front

    def get_two_qubit_front(self) -> list[GateNode]:
        """获取前沿层中的双比特门 (路由的核心关注对象)。"""
        return [g for g in self.get_front_layer() if g.is_two_qubit]


    def get_extended_front(self, depth: int = 2) -> list[GateNode]:
        """V3 Look-Ahead: 前沿 + 后续 depth 层的双比特门。"""
        result = list(self.get_two_qubit_front())
        seen = {g.gate_id for g in result}
        sim_done = {gid for gid, g in self._gates.items() if g.executed}
        for g in self.get_front_layer():
            if not g.is_two_qubit:
                sim_done.add(g.gate_id)
        for _ in range(depth):
            nxt = []
            for gid, node in self._gates.items():
                if gid in sim_done or gid in seen:
                    continue
                if all(p in sim_done or p in seen
                       for p in self._graph.predecessors(gid)):
                    nxt.append(node)
            for g in nxt:
                if g.is_two_qubit:
                    result.append(g)
                    seen.add(g.gate_id)
                else:
                    sim_done.add(g.gate_id)
        return result

    def execute_gate(self, gate_id: int) -> None:
        """标记一个门为已执行。

        Args:
            gate_id: 门的唯一 ID
        Raises:
            ValueError: 门不在前沿层中
        """
        if gate_id not in self._gates:
            raise ValueError(f"门 {gate_id} 不存在")
        if self._gates[gate_id].executed:
            raise ValueError(f"门 {gate_id} 已被执行")

        front_ids = {g.gate_id for g in self.get_front_layer()}
        if gate_id not in front_ids:
            raise ValueError(f"门 {gate_id} 不在前沿层中，无法执行")

        self._gates[gate_id].executed = True

    def execute_executable(self, mapping: dict[int, int],
                           coupling_map: CouplingMap) -> int:
        """执行前沿层中所有满足拓扑约束的门。

        Args:
            mapping: 逻辑比特 → 物理比特 的映射
            coupling_map: 物理拓扑
        Returns:
            执行的门数
        """
        edges = set(tuple(e) for e in coupling_map.get_edges())
        executed_count = 0

        for gate in self.get_front_layer():
            if not gate.is_two_qubit:
                # 单比特门无拓扑约束
                self.execute_gate(gate.gate_id)
                executed_count += 1
            else:
                # 双比特门需要检查物理连通性
                p0 = mapping[gate.qubits[0]]
                p1 = mapping[gate.qubits[1]]
                if (p0, p1) in edges or (p1, p0) in edges:
                    self.execute_gate(gate.gate_id)
                    executed_count += 1

        return executed_count

    @staticmethod
    def apply_swap(q1: int, q2: int, mapping: dict[int, int]) -> dict[int, int]:
        """应用 SWAP 操作，更新逻辑→物理映射。

        SWAP(q1, q2) 交换两个物理比特上的逻辑比特。

        Args:
            q1, q2: 要交换的两个物理比特
            mapping: 当前逻辑→物理映射 (会被修改)
        Returns:
            更新后的映射
        """
        new_mapping = dict(mapping)
        # 找到映射到 q1 和 q2 的逻辑比特
        log1 = None
        log2 = None
        for log, phys in new_mapping.items():
            if phys == q1:
                log1 = log
            if phys == q2:
                log2 = log

        if log1 is not None and log2 is not None:
            new_mapping[log1] = q2
            new_mapping[log2] = q1
        elif log1 is not None:
            new_mapping[log1] = q2
        elif log2 is not None:
            new_mapping[log2] = q1

        return new_mapping

    def remaining_gates(self) -> int:
        """剩余未执行的门数。"""
        return sum(1 for g in self._gates.values() if not g.executed)

    def remaining_two_qubit_gates(self) -> int:
        """剩余未执行的双比特门数。"""
        return sum(1 for g in self._gates.values()
                   if not g.executed and g.is_two_qubit)

    def is_done(self) -> bool:
        """所有门是否已执行完毕。"""
        return self.remaining_gates() == 0

    def to_networkx(self) -> nx.DiGraph:
        """导出为 NetworkX 有向图 (供 GNN 使用)。

        节点属性包含门信息，适合转换为 PyTorch Geometric 的 Data 对象。

        Returns:
            带节点属性的 NetworkX DiGraph
        """
        G = nx.DiGraph()
        for gid, node in self._gates.items():
            if not node.executed:
                G.add_node(gid, **{
                    'gate_type': GATE_TYPE_MAP.get(node.gate_type, len(GATE_TYPE_MAP)),
                    'n_qubits': len(node.qubits),
                    'qubits': node.qubits,
                    'is_two_qubit': int(node.is_two_qubit),
                })

        for u, v in self._graph.edges():
            if not self._gates[u].executed and not self._gates[v].executed:
                G.add_edge(u, v)

        return G

    def get_node_features(self) -> np.ndarray:
        """获取节点特征矩阵 (供 GNN 使用)。

        特征维度: [gate_type_onehot, n_qubits, is_two_qubit]

        Returns:
            shape (n_remaining_gates, feature_dim) 的 numpy 数组
        """
        n_types = len(GATE_TYPE_MAP) + 1  # +1 for unknown
        features = []

        for gid, node in self._gates.items():
            if node.executed:
                continue
            # One-hot 门类型
            onehot = np.zeros(n_types, dtype=np.float32)
            type_idx = GATE_TYPE_MAP.get(node.gate_type, n_types - 1)
            onehot[type_idx] = 1.0
            # 额外特征
            extra = np.array([len(node.qubits), int(node.is_two_qubit)],
                             dtype=np.float32)
            features.append(np.concatenate([onehot, extra]))

        if not features:
            n_types_total = n_types + 2
            return np.zeros((0, n_types_total), dtype=np.float32)

        return np.stack(features)

    def get_interaction_graph(self) -> nx.Graph:
        """提取逻辑比特交互图。

        节点 = 逻辑比特, 边 = 两个比特之间存在双比特门交互。
        边权重 = 交互次数。

        Returns:
            无向加权图
        """
        G = nx.Graph()
        G.add_nodes_from(range(self._n_qubits))

        for node in self._gates.values():
            if not node.executed and node.is_two_qubit:
                q0, q1 = node.qubits[0], node.qubits[1]
                if G.has_edge(q0, q1):
                    G[q0][q1]['weight'] += 1
                else:
                    G.add_edge(q0, q1, weight=1)

        return G
