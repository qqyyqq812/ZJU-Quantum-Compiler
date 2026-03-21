"""
M1.2 Coupling Map 可视化：理解量子芯片的物理拓扑约束
=====================================================
学习目标：
1. 理解 Coupling Map 是什么（哪些比特之间可以做 CNOT）
2. 加载真实芯片拓扑
3. 可视化不同拓扑的差异
4. 理解为什么需要 SWAP 门

EDA 类比：Coupling Map ≈ FPGA 互连矩阵 / 芯片布线资源
"""

from qiskit.transpiler import CouplingMap
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 无头渲染
import matplotlib.pyplot as plt
import json
from pathlib import Path


# ============================================================
# 常见量子芯片拓扑定义
# ============================================================

def create_topologies() -> dict[str, CouplingMap]:
    """创建常见量子芯片拓扑"""
    topologies = {}

    # 1. 线性拓扑 (5 qubit) — 最简单
    topologies['linear_5q'] = CouplingMap.from_line(5)

    # 2. 环形拓扑 (5 qubit)
    topologies['ring_5q'] = CouplingMap.from_ring(5)

    # 3. 网格拓扑 (3x3 = 9 qubit) — 类似 Google Sycamore 的基础结构
    topologies['grid_3x3'] = CouplingMap.from_grid(3, 3)

    # 4. 重型六边形拓扑 (heavy-hex) — IBM Eagle/Heron 的基础结构
    topologies['heavy_hex_15q'] = CouplingMap.from_heavy_hex(3)  # d=3 → 15 qubits

    # 5. 全连接拓扑 (5 qubit) — 理想情况，不需要 SWAP
    topologies['full_5q'] = CouplingMap.from_full(5)

    return topologies


def analyze_topology(name: str, cm: CouplingMap) -> dict:
    """分析一个拓扑的关键属性"""
    # Qiskit 2.x 使用 rustworkx，需要转换为 networkx 图
    n = cm.size()
    edges = cm.get_edges()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    num_edges = graph.number_of_edges()

    # 最大/平均最短路径
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        avg_path = nx.average_shortest_path_length(graph)
    else:
        diameter = -1
        avg_path = -1

    # 连通度
    degrees = [d for _, d in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0

    info = {
        'name': name,
        'num_qubits': n,
        'num_edges': num_edges,
        'diameter': diameter,
        'avg_shortest_path': round(avg_path, 2),
        'avg_degree': round(avg_degree, 2),
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0,
    }
    return info


def visualize_topology(name: str, cm: CouplingMap, ax) -> None:
    """可视化一个 Coupling Map"""
    n = cm.size()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(cm.get_edges())
    pos = nx.spring_layout(graph, seed=42, k=2)

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='#4fc3f7',
                           node_size=400, edgecolors='#0277bd', linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#90a4ae', width=1.5)

    info = analyze_topology(name, cm)
    ax.set_title(f"{name}\n{info['num_qubits']}q, {info['num_edges']}edges, "
                 f"diam={info['diameter']}, avg_path={info['avg_shortest_path']}",
                 fontsize=9)
    ax.axis('off')


def main():
    topologies = create_topologies()
    output_dir = Path(__file__).parent

    # 1. 打印分析表
    print("=" * 80)
    print("量子芯片拓扑分析")
    print("=" * 80)
    print(f"{'名称':<20} {'比特数':>6} {'边数':>6} {'直径':>6} {'平均路径':>8} {'平均度':>6}")
    print("-" * 60)

    all_info = []
    for name, cm in topologies.items():
        info = analyze_topology(name, cm)
        all_info.append(info)
        print(f"{info['name']:<20} {info['num_qubits']:>6} {info['num_edges']:>6} "
              f"{info['diameter']:>6} {info['avg_shortest_path']:>8} {info['avg_degree']:>6}")

    # 2. 生成可视化图
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('量子芯片 Coupling Map 拓扑对比', fontsize=14, fontweight='bold')

    for ax, (name, cm) in zip(axes, topologies.items()):
        visualize_topology(name, cm, ax)

    plt.tight_layout()
    fig_path = output_dir / 'coupling_map_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: {fig_path}")

    # 3. 保存分析数据
    data_path = output_dir / 'topology_analysis.json'
    with open(data_path, 'w') as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 分析数据已保存: {data_path}")

    # 4. SWAP 距离示例
    print("\n" + "=" * 80)
    print("SWAP 需求示例（grid_3x3 拓扑）")
    print("=" * 80)
    grid = topologies['grid_3x3']
    grid_graph = nx.Graph()
    grid_graph.add_nodes_from(range(grid.size()))
    grid_graph.add_edges_from(grid.get_edges())
    pairs = [(0, 8), (0, 4), (2, 6), (0, 1)]
    for q1, q2 in pairs:
        dist = nx.shortest_path_length(grid_graph, q1, q2)
        swaps = max(0, dist - 1)
        print(f"  CNOT(q{q1}, q{q2}): 物理距离={dist}, 需要 {swaps} 个 SWAP (= {swaps*3} 个额外 CNOT)")

    return all_info


if __name__ == '__main__':
    main()
