"""
拓扑可视化引擎 — V1
====================
核心功能:
1. 渲染任意 CouplingMap 为精美的 2D 物理拓扑图
2. 在拓扑图上叠加"逻辑→物理"的当前映射状态
3. 捕获路由过程中每一步的映射快照 → 逐帧渲染 → 合成 GIF 动画

用法:
    # 1. 静态拓扑图
    python -m src.visualization.topology_visualizer static ibm_tokyo

    # 2. 路由动画 Demo（使用已有模型）
    python -m src.visualization.topology_visualizer animate ibm_tokyo models/v9_tokyo20/v7_ibm_tokyo_best.pt
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # 无头模式，服务器安全

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from PIL import Image

# ============================================================
# IBM Tokyo 物理布局坐标 (手工精调)
# ============================================================

# IBM Tokyo 20Q 是一个 4 行 5 列的近似网格，带有额外的对角连边
# 物理布局：按照 IBM 官方芯片排布图
# Row 0: Q0  Q1  Q2  Q3  Q4
# Row 1: Q5  Q6  Q7  Q8  Q9
# Row 2: Q10 Q11 Q12 Q13 Q14
# Row 3: Q15 Q16 Q17 Q18 Q19

IBM_TOKYO_POS = {
    0: (0, 3), 1: (1, 3), 2: (2, 3), 3: (3, 3), 4: (4, 3),
    5: (0, 2), 6: (1, 2), 7: (2, 2), 8: (3, 2), 9: (4, 2),
    10: (0, 1), 11: (1, 1), 12: (2, 1), 13: (3, 1), 14: (4, 1),
    15: (0, 0), 16: (1, 0), 17: (2, 0), 18: (3, 0), 19: (4, 0),
}


def _get_layout(coupling_map, topology_name: str = "auto") -> dict:
    """为给定拓扑生成节点坐标。"""
    n = coupling_map.size()

    if topology_name == "ibm_tokyo" and n == 20:
        return IBM_TOKYO_POS

    # 通用 fallback：用 networkx 的 spring 布局
    G = nx.Graph()
    G.add_nodes_from(range(n))
    seen = set()
    for u, v in coupling_map.get_edges():
        key = (min(u, v), max(u, v))
        if key not in seen:
            G.add_edge(u, v)
            seen.add(key)
    return nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n))


# ============================================================
# 渲染引擎
# ============================================================

# 颜色方案 (科研论文暗色调)
COLORS = {
    'bg': '#1a1a2e',
    'edge': '#555577',
    'edge_active': '#FFD700',     # 金色：刚发生 SWAP 的边
    'node_idle': '#334466',       # 深蓝灰：无逻辑比特
    'node_mapped': '#00C9A7',     # 翡翠绿：有逻辑比特映射
    'node_front': '#FF6B6B',      # 珊瑚红：前沿门涉及的比特
    'text_phys': '#AAAACC',       # 物理比特编号
    'text_logic': '#FFFFFF',      # 逻辑比特编号
    'title': '#E0E0FF',
}


def render_topology(
    coupling_map,
    topology_name: str = "auto",
    mapping: Optional[dict] = None,
    front_qubits: Optional[set] = None,
    active_swap_edge: Optional[tuple] = None,
    step_info: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    ax=None,
) -> plt.Figure:
    """
    渲染单帧拓扑图。

    Args:
        coupling_map: Qiskit CouplingMap
        topology_name: 拓扑名（用于选择布局）
        mapping: {逻辑比特 → 物理比特} 的映射字典
        front_qubits: 前沿门涉及的物理比特集合（高亮红色）
        active_swap_edge: 当前步骤刚执行的 SWAP 边 (p1, p2)（金色高亮）
        step_info: 标题栏的附加信息（如 "Step 12 / SWAP(3,8)"）
        save_path: 保存路径
        figsize: 图片尺寸
        ax: 可选的外部 axes (用于嵌入子图)
    """
    pos = _get_layout(coupling_map, topology_name)
    n = coupling_map.size()

    # 构建 networkx 图
    G = nx.Graph()
    G.add_nodes_from(range(n))
    seen = set()
    for u, v in coupling_map.get_edges():
        key = (min(u, v), max(u, v))
        if key not in seen:
            G.add_edge(u, v)
            seen.add(key)

    # 反转映射: 物理 → 逻辑
    phys_to_logic = {}
    if mapping:
        for log_q, phys_q in mapping.items():
            phys_to_logic[phys_q] = log_q

    # 节点颜色
    node_colors = []
    for i in range(n):
        if front_qubits and i in front_qubits:
            node_colors.append(COLORS['node_front'])
        elif i in phys_to_logic:
            node_colors.append(COLORS['node_mapped'])
        else:
            node_colors.append(COLORS['node_idle'])

    # 边颜色
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if active_swap_edge and set(active_swap_edge) == {u, v}:
            edge_colors.append(COLORS['edge_active'])
            edge_widths.append(3.5)
        else:
            edge_colors.append(COLORS['edge'])
            edge_widths.append(1.2)

    # 绘图
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=COLORS['bg'])
    else:
        fig = ax.figure
    ax.set_facecolor(COLORS['bg'])

    # 画边
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.8,
    )

    # 画节点
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=700,
        edgecolors='#FFFFFF33',
        linewidths=1.5,
    )

    # 物理比特编号 (小字, 偏移)
    for i in range(n):
        x, y = pos[i]
        ax.text(x, y + 0.18, f'P{i}', fontsize=7, color=COLORS['text_phys'],
                ha='center', va='bottom', fontweight='bold')

    # 逻辑比特映射标注 (节点内大字)
    for phys_q, log_q in phys_to_logic.items():
        x, y = pos[phys_q]
        ax.text(x, y, f'L{log_q}', fontsize=11, color=COLORS['text_logic'],
                ha='center', va='center', fontweight='bold')

    # 标题
    title = f'Quantum Chip Topology — {topology_name.upper()} ({n}Q)'
    if step_info:
        title += f'\n{step_info}'
    ax.set_title(title, fontsize=14, color=COLORS['title'], fontweight='bold', pad=15)

    # 图例
    legend_elements = [
        mpatches.Patch(color=COLORS['node_mapped'], label='Mapped Qubit'),
        mpatches.Patch(color=COLORS['node_front'], label='Front Gate Qubit'),
        mpatches.Patch(color=COLORS['node_idle'], label='Idle Physical Qubit'),
        mpatches.Patch(color=COLORS['edge_active'], label='Active SWAP'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              facecolor='#0a0a1a', edgecolor='#333355', labelcolor='#CCCCEE')

    ax.axis('off')
    if own_fig:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"  💾 拓扑图已保存: {save_path}")

    return fig


# ============================================================
# 路由捕获器 — 记录 AI 路由过程中每一步的映射快照
# ============================================================

class RouteCapturer:
    """在 AI 路由过程中逐步捕获映射状态，用于事后生成动画。"""

    def __init__(self):
        self.snapshots: list[dict] = []

    def capture(
        self,
        mapping: dict,
        action: int,
        swap_edges: list,
        front_qubits: set | None = None,
        reward: float = 0.0,
    ):
        """记录一帧快照。"""
        snap = {
            'mapping': dict(mapping),
            'action': action,
            'front_qubits': set(front_qubits) if front_qubits else set(),
            'reward': reward,
            'active_swap': None,
        }
        if action < len(swap_edges):
            snap['active_swap'] = swap_edges[action]
        self.snapshots.append(snap)

    def clear(self):
        self.snapshots.clear()


def run_captured_route(circuit, policy, coupling_map, topology_name="ibm_tokyo"):
    """执行一次 AI 路由，同时捕获全过程快照。"""
    import torch
    from src.compiler.env import QuantumRoutingEnv

    env = QuantumRoutingEnv(coupling_map=coupling_map, max_steps=500,
                            soft_mask=True, tabu_size=4)
    env.set_circuit(circuit)
    obs, info = env.reset()

    capturer = RouteCapturer()
    done = False

    with torch.no_grad():
        while not done:
            mask = env.get_action_mask()
            action, _, _ = policy.get_action(
                obs, action_mask=mask, gnn_input=info.get("gnn_input")
            )

            # 捕获当前帧 (在 step 之前)
            front_phys = set()
            if env._dag:
                for gate in env._dag.get_two_qubit_front():
                    p0 = env._mapping.get(gate.qubits[0], gate.qubits[0])
                    p1 = env._mapping.get(gate.qubits[1], gate.qubits[1])
                    front_phys.update({p0, p1})

            capturer.capture(
                mapping=env._mapping,
                action=action,
                swap_edges=env.swap_edges,
                front_qubits=front_phys,
            )

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    return capturer, env._total_swaps, env._dag.is_done()


# ============================================================
# GIF 动画合成器
# ============================================================

def generate_route_gif(
    capturer: RouteCapturer,
    coupling_map,
    topology_name: str = "ibm_tokyo",
    output_path: str = "results/route_animation.gif",
    max_frames: int = 60,
    frame_duration: int = 500,  # ms
):
    """
    将捕获的路由快照合成为 GIF 动画。

    Args:
        capturer: 路由捕获器
        coupling_map: CouplingMap
        topology_name: 拓扑名
        output_path: GIF 输出路径
        max_frames: 最大帧数（太长的路由会均匀采样）
        frame_duration: 每帧持续时间 (ms)
    """
    snaps = capturer.snapshots
    if not snaps:
        print("  ⚠️ 没有捕获到任何快照")
        return

    # 如果帧数太多，均匀采样
    if len(snaps) > max_frames:
        indices = np.linspace(0, len(snaps) - 1, max_frames, dtype=int)
        snaps = [snaps[i] for i in indices]
    else:
        indices = list(range(len(snaps)))

    print(f"  🎬 正在渲染 {len(snaps)} 帧动画...")

    frames = []
    for frame_idx, snap in enumerate(snaps):
        action_str = "PASS" if snap['active_swap'] is None else f"SWAP{snap['active_swap']}"
        step_info = f"Step {indices[frame_idx] if frame_idx < len(indices) else frame_idx} | Action: {action_str}"

        fig = render_topology(
            coupling_map,
            topology_name=topology_name,
            mapping=snap['mapping'],
            front_qubits=snap['front_qubits'],
            active_swap_edge=snap['active_swap'],
            step_info=step_info,
            figsize=(10, 8),
        )

        # fig → PIL Image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes('RGBA', (w, h), fig.canvas.buffer_rgba())
        frames.append(img.convert('RGB'))
        plt.close(fig)

    # 保存 GIF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )
    print(f"  ✅ 路由动画已保存: {output_path} ({len(frames)} 帧, {len(frames)*frame_duration/1000:.1f}s)")


# ============================================================
# CLI 入口
# ============================================================

def main():
    import sys
    from src.benchmarks.topologies import get_topology

    if len(sys.argv) < 3:
        print("用法:")
        print("  python -m src.visualization.topology_visualizer static <topology>")
        print("  python -m src.visualization.topology_visualizer animate <topology> <model_path>")
        sys.exit(1)

    mode = sys.argv[1]
    topo_name = sys.argv[2]
    cm = get_topology(topo_name)

    if mode == "static":
        # 纯静态拓扑图
        out = f"results/{topo_name}_topology.png"
        render_topology(cm, topology_name=topo_name, save_path=out)
        print(f"完成! {out}")

    elif mode == "animate":
        if len(sys.argv) < 4:
            print("需要模型路径! 用法: ... animate <topology> <model_path>")
            sys.exit(1)

        model_path = sys.argv[3]
        from src.compiler.inference_v8 import load_policy
        from src.benchmarks.circuits import generate_qft

        policy, _ = load_policy(model_path, topo_name)
        circuit = generate_qft(5)  # 使用 5Q QFT 作为 Demo

        print(f"🎬 执行 AI 路由并捕获过程...")
        capturer, total_swaps, completed = run_captured_route(
            circuit, policy, cm, topology_name=topo_name
        )
        print(f"   路由完成: SWAP={total_swaps}, 成功={completed}, 步数={len(capturer.snapshots)}")

        out = f"results/{topo_name}_route_animation.gif"
        generate_route_gif(capturer, cm, topology_name=topo_name, output_path=out)

    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    main()
