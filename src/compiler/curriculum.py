"""
课程学习调度器
==============
从简单电路渐进到复杂电路，帮助 PPO 收敛。

阶段1: 2-3 qubit, 1-2 个 CX → 学会基本 SWAP
阶段2: 3-4 qubit, 多个 CX → 学会多步规划
阶段3: 5 qubit, 全套基准 → 精细优化
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa


def get_curriculum_circuits(stage: int, seed: int = 42) -> list[QuantumCircuit]:
    """根据训练阶段返回对应难度的电路集。

    Args:
        stage: 训练阶段 (1, 2, 3)
        seed: 随机种子
    """
    if stage == 1:
        # 极简电路: 2-3 qubit, 1-2 个 CX
        circuits = []
        for i in range(10):
            qc = QuantumCircuit(3)
            qc.cx(0, 2)  # 需要 1 个 SWAP
            circuits.append(qc)
        for i in range(10):
            qc = QuantumCircuit(3)
            qc.cx(0, 1)
            qc.cx(1, 2)
            circuits.append(qc)
        return circuits

    elif stage == 2:
        # 中等电路: 4 qubit, 多个 CX
        circuits = []
        for i in range(10):
            circuits.append(generate_random(4, depth=3, seed=seed + i))
        circuits.append(generate_qft(4))
        circuits.append(generate_qaoa(4, p=1))
        return circuits

    else:
        # 完整电路: 5 qubit, 全套
        circuits = []
        for i in range(15):
            circuits.append(generate_random(5, depth=5, seed=seed + i))
        circuits.append(generate_qft(5))
        circuits.append(generate_qaoa(5, p=1))
        circuits.append(generate_qaoa(5, p=2))
        return circuits


def get_stage(episode: int, total_episodes: int) -> int:
    """根据当前 episode 确定训练阶段。"""
    if episode < total_episodes * 0.2:
        return 1
    elif episode < total_episodes * 0.5:
        return 2
    else:
        return 3
