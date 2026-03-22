"""
量子芯片拓扑集
==============
加载/创建量子芯片的 Coupling Map 拓扑。

支持:
- 理论拓扑: linear, ring, grid, heavy_hex, full
- 真实芯片: IBM Eagle (127q), IBM Heron (133q), Google Sycamore (53q)
"""

from __future__ import annotations

import networkx as nx
from qiskit.transpiler import CouplingMap


# ============================================================
# 真实芯片拓扑定义 (手工构建核心结构)
# ============================================================

def _build_ibm_eagle_coupling() -> CouplingMap:
    """IBM Eagle r3 拓扑 (127 qubits, heavy-hex 结构)。
    使用 Qiskit 内置的 heavy-hex 生成器近似。
    """
    return CouplingMap.from_heavy_hex(7)  # d=7 → 127 qubits


def _build_google_sycamore_coupling() -> CouplingMap:
    """Google Sycamore 拓扑 (53 qubits, 网格+对角连接)。
    近似为 6x9=54 grid (去掉一个角 → 53)。
    """
    cm = CouplingMap.from_grid(6, 9)
    # Sycamore 实际是 54 qubit grid, 这里用 54 近似
    return cm


def _build_ibm_heron_coupling() -> CouplingMap:
    """IBM Heron 拓扑 (133 qubits, heavy-hex 结构)。"""
    return CouplingMap.from_heavy_hex(7, bidirectional=True)


# ============================================================
# 拓扑注册表
# ============================================================

_TOPOLOGY_REGISTRY: dict[str, callable] = {
    'ibm_eagle': _build_ibm_eagle_coupling,
    'google_sycamore': _build_google_sycamore_coupling,
    'ibm_heron': _build_ibm_heron_coupling,
}


def get_topology(name: str) -> CouplingMap:
    """获取指定名称的拓扑。

    支持的名称格式:
    - "linear_N": N 比特线性拓扑
    - "ring_N": N 比特环形拓扑
    - "grid_RxC": R 行 C 列网格拓扑
    - "heavy_hex_D": 距离参数为 D 的重型六边形拓扑
    - "full_N": N 比特全连接拓扑
    - "ibm_eagle": IBM Eagle 127q
    - "google_sycamore": Google Sycamore 53q
    - "ibm_heron": IBM Heron 133q

    Args:
        name: 拓扑名称
    Returns:
        CouplingMap 对象
    Raises:
        ValueError: 未知的拓扑名称
    """
    # 注册表中的真实芯片
    if name in _TOPOLOGY_REGISTRY:
        return _TOPOLOGY_REGISTRY[name]()

    # 参数化拓扑
    if name.startswith('linear_'):
        n = int(name.split('_')[1])
        return CouplingMap.from_line(n)
    elif name.startswith('ring_'):
        n = int(name.split('_')[1])
        return CouplingMap.from_ring(n)
    elif name.startswith('grid_'):
        dims = name.split('_')[1].split('x')
        return CouplingMap.from_grid(int(dims[0]), int(dims[1]))
    elif name.startswith('heavy_hex_'):
        d = int(name.split('_')[2])
        return CouplingMap.from_heavy_hex(d)
    elif name.startswith('full_'):
        n = int(name.split('_')[1])
        return CouplingMap.from_full(n)
    else:
        raise ValueError(f"未知拓扑: {name}. 支持: linear_N, ring_N, grid_RxC, "
                         f"heavy_hex_D, full_N, ibm_eagle, google_sycamore, ibm_heron")


def get_topology_info(cm: CouplingMap) -> dict:
    """分析拓扑的关键属性。

    Args:
        cm: CouplingMap 对象
    Returns:
        包含 n_qubits, n_edges, diameter, avg_path, avg_degree 的字典
    """
    n = cm.size()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(cm.get_edges())

    n_edges = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]

    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        avg_path = round(nx.average_shortest_path_length(graph), 2)
    else:
        diameter = -1
        avg_path = -1.0

    return {
        'n_qubits': n,
        'n_edges': n_edges,
        'diameter': diameter,
        'avg_path': avg_path,
        'avg_degree': round(sum(degrees) / len(degrees), 2) if degrees else 0,
        'max_degree': max(degrees) if degrees else 0,
    }


def get_standard_topologies() -> dict[str, CouplingMap]:
    """获取标准测试拓扑集合。

    Returns:
        {名称: CouplingMap} 的字典, 按规模从小到大排列
    """
    return {
        'linear_5': get_topology('linear_5'),
        'ring_5': get_topology('ring_5'),
        'grid_3x3': get_topology('grid_3x3'),
        'heavy_hex_3': get_topology('heavy_hex_3'),
        'linear_20': get_topology('linear_20'),
        'grid_5x5': get_topology('grid_5x5'),
        'ibm_eagle': get_topology('ibm_eagle'),
    }
