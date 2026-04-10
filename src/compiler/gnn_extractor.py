"""
统一的拓扑感知图特征提取器 (V13)
===============================
将当前电路路由状态编码为以物理量子比特为节点的单一图结构。

V13 升级: 5维 → 9维节点特征
- 新增: 映射对目标距离、DAG 剩余深度、前沿门目标物理距离、逻辑比特归一化 ID
"""
from __future__ import annotations

import torch
from qiskit.transpiler import CouplingMap
from src.compiler.dag import CircuitDAG


class GraphData:
    """轻量级图数据容器，替代 torch_geometric.data.Data。"""
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.x = x
        self.edge_index = edge_index

    def to(self, device):
        return GraphData(self.x.to(device), self.edge_index.to(device))


class GraphBatch:
    """轻量级批量图容器，替代 torch_geometric.data.Batch。

    将多个 GraphData 合并为单个大图：
    - x: 所有节点特征拼接
    - edge_index: 所有边索引拼接（偏移后）
    - batch: 每个节点属于哪个子图
    - ptr: 每个子图的节点起始偏移
    """
    def __init__(self, x, edge_index, batch, ptr):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.ptr = ptr

    def to(self, device):
        return GraphBatch(
            self.x.to(device), self.edge_index.to(device),
            self.batch.to(device), self.ptr.to(device)
        )

    @staticmethod
    def from_data_list(data_list: list[GraphData]) -> 'GraphBatch':
        xs, edge_indices, batches = [], [], []
        ptr = [0]
        offset = 0
        for i, data in enumerate(data_list):
            n = data.x.size(0)
            xs.append(data.x)
            edge_indices.append(data.edge_index + offset)
            batches.append(torch.full((n,), i, dtype=torch.long))
            offset += n
            ptr.append(offset)
        return GraphBatch(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long),
            batch=torch.cat(batches, dim=0),
            ptr=torch.tensor(ptr, dtype=torch.long),
        )

def extract_physical_graph(
    coupling_map: CouplingMap,
    mapping: dict[int, int],
    dag: CircuitDAG | None
) -> Data:
    """提取基于物理拓扑的统一图表征。

    图定义:
        节点: 物理量子比特 (0 to N-1)
        边: CouplingMap 的物理连接
        节点特征 (9维):
            [
                0: 物理节点的度数 (归一化)
                1: 是否被逻辑比特占用 (1 或 0)
                2: 映射的逻辑比特索引 (归一化到 [0,1]，未映射=-1)
                3: 是否参与当前前沿双比特门 (1 或 0)
                4: 是否参与 Look-ahead 扩展前沿门 (1 或 0)
                5: 前沿门目标距离 (该节点参与的前沿门，其 partner 的物理距离，归一化)
                6: DAG 剩余双比特门数 (归一化)
                7: 该节点上逻辑比特在 DAG 中的剩余参与度 (归一化)
                8: 节点到所有前沿门 partner 的最小距离 (归一化)
            ]
    """
    n_phys = coupling_map.size()

    # 预计算距离矩阵
    dist_matrix = {}
    for i in range(n_phys):
        for j in range(n_phys):
            try:
                dist_matrix[(i, j)] = coupling_map.distance(i, j)
            except Exception:
                dist_matrix[(i, j)] = n_phys
    max_dist = max(dist_matrix.values()) if dist_matrix else 1.0

    # 构建边
    edges = list(coupling_map.get_edges())
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 构建节点特征 (9维)
    x = torch.zeros((n_phys, 9), dtype=torch.float)

    # 特征 0: Degree (归一化)
    max_degree = 1.0
    for u, v in edges:
        x[u, 0] += 1
        x[v, 0] += 1
    max_degree = max(float(x[:, 0].max()), 1.0)
    x[:, 0] /= max_degree

    # 特征 1, 2: Occupancy & Logical ID (归一化)
    phys_to_logical = {p: l for l, p in mapping.items()}
    n_logical = max(len(mapping), 1)
    for p in range(n_phys):
        if p in phys_to_logical:
            x[p, 1] = 1.0
            x[p, 2] = float(phys_to_logical[p]) / n_logical  # 归一化
        else:
            x[p, 1] = 0.0
            x[p, 2] = -1.0

    # 特征 3, 4, 5, 6, 7, 8: 需要 DAG 信息
    if dag is not None and not dag.is_done():
        front_gates = dag.get_two_qubit_front()
        extended_gates = dag.get_extended_front(depth=2)
        front_ids = {g.gate_id for g in front_gates}
        remaining_2q = dag.remaining_two_qubit_gates()

        # 特征 6: DAG 剩余双比特门数 (归一化)
        total_2q = max(dag.n_two_qubit_gates if hasattr(dag, 'n_two_qubit_gates') else remaining_2q, 1)
        x[:, 6] = float(remaining_2q) / total_2q

        # 特征 3, 5: Front Gate + 前沿门目标距离
        for gate in front_gates:
            p0 = mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < n_phys:
                x[p0, 3] = 1.0
                d = dist_matrix.get((p0, p1), max_dist) if p1 < n_phys else max_dist
                x[p0, 5] = max(x[p0, 5].item(), d / max_dist)
            if p1 < n_phys:
                x[p1, 3] = 1.0
                d = dist_matrix.get((p1, p0), max_dist) if p0 < n_phys else max_dist
                x[p1, 5] = max(x[p1, 5].item(), d / max_dist)

        # 特征 4: Extended Front Gate
        for gate in extended_gates:
            if gate.gate_id not in front_ids:
                p0 = mapping.get(gate.qubits[0], gate.qubits[0])
                p1 = mapping.get(gate.qubits[1], gate.qubits[1])
                if p0 < n_phys: x[p0, 4] = 1.0
                if p1 < n_phys: x[p1, 4] = 1.0

        # 特征 7: 节点上逻辑比特在 DAG 中的剩余参与度
        for p in range(n_phys):
            if p in phys_to_logical:
                logical = phys_to_logical[p]
                if hasattr(dag, 'qubit_remaining_gates'):
                    x[p, 7] = float(dag.qubit_remaining_gates(logical)) / max(total_2q, 1)

        # 特征 8: 节点到所有前沿门 partner 的最小距离
        if front_gates:
            front_physicals = set()
            for gate in front_gates:
                p0 = mapping.get(gate.qubits[0], gate.qubits[0])
                p1 = mapping.get(gate.qubits[1], gate.qubits[1])
                if p0 < n_phys: front_physicals.add(p0)
                if p1 < n_phys: front_physicals.add(p1)
            for p in range(n_phys):
                if front_physicals:
                    min_d = min(dist_matrix.get((p, fp), max_dist) for fp in front_physicals)
                    x[p, 8] = min_d / max_dist

    return GraphData(x=x, edge_index=edge_index)
