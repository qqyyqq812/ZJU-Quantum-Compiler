"""
课程学习调度器 V7
==================
5 阶段渐进式训练课程 + 自适应升级条件。

阶段 0: Warm-up    — 3Q, 1-2 CX, 学基本 SWAP 意识
阶段 1: Elementary — 5Q, depth 3, 学多步规划
阶段 2: Standard   — 5Q, depth 5, 全套 QFT/QAOA/Grover
阶段 3: Challenge  — 10Q, depth 8, 大规模电路
阶段 4: Master     — 20Q, depth 15, 芯片级真实挑战
"""

from __future__ import annotations

from dataclasses import dataclass
from qiskit import QuantumCircuit

from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa, generate_grover


@dataclass
class StageConfig:
    """单个课程阶段的配置"""
    name: str
    n_qubits: int
    n_circuits: int
    depth_range: tuple[int, int]
    include_structured: bool
    promotion_threshold: float  # 平均 SWAP 数低于此值时升级


STAGES = [
    StageConfig("warm-up",     3,  20, (1, 2), False, 1.5),
    StageConfig("elementary",  5,  30, (2, 4), True,  4.0),
    StageConfig("standard",    5,  40, (4, 6), True,  6.0),
    StageConfig("challenge",  10,  50, (6, 10), True, 12.0),
    StageConfig("master",     20,  60, (10, 20), True, 25.0),
]


def build_stage_circuits(stage: int, seed: int = 42) -> list[QuantumCircuit]:
    """根据阶段编号构建该阶段的训练电路集。"""
    cfg = STAGES[min(stage, len(STAGES) - 1)]
    circuits = []

    # 随机电路
    for i in range(cfg.n_circuits):
        depth = cfg.depth_range[0] + (i % (cfg.depth_range[1] - cfg.depth_range[0] + 1))
        circuits.append(generate_random(cfg.n_qubits, depth=depth, seed=seed + i))

    # 结构化电路
    if cfg.include_structured:
        circuits.append(generate_qft(cfg.n_qubits))
        circuits.append(generate_qaoa(cfg.n_qubits, p=1))
        if cfg.n_qubits >= 5:
            circuits.append(generate_qaoa(cfg.n_qubits, p=2))
            circuits.append(generate_grover(cfg.n_qubits))

    return circuits


class CurriculumScheduler:
    """自适应课程调度器。

    基于滑动窗口平均 SWAP 数来决定是否升级到下一阶段。
    """

    def __init__(self, window_size: int = 50, min_episodes_per_stage: int = 500):
        self.current_stage = 0
        self.window_size = window_size
        self.min_episodes_per_stage = min_episodes_per_stage
        self._swap_history: list[float] = []
        self._stage_episode_count = 0
        self._circuits = build_stage_circuits(0)

    @property
    def stage_config(self) -> StageConfig:
        return STAGES[min(self.current_stage, len(STAGES) - 1)]

    @property
    def circuits(self) -> list[QuantumCircuit]:
        return self._circuits

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage >= len(STAGES) - 1

    def report_episode(self, n_swaps: int) -> bool:
        """报告一个 episode 的 SWAP 数，返回是否刚发生了阶段升级。"""
        self._swap_history.append(n_swaps)
        self._stage_episode_count += 1

        if self.is_final_stage:
            return False

        # 检查是否满足升级条件
        if self._stage_episode_count < self.min_episodes_per_stage:
            return False

        if len(self._swap_history) < self.window_size:
            return False

        recent_avg = sum(self._swap_history[-self.window_size:]) / self.window_size
        threshold = self.stage_config.promotion_threshold

        if recent_avg <= threshold:
            return self._promote()

        return False

    def _promote(self) -> bool:
        """升级到下一阶段。"""
        self.current_stage += 1
        self._swap_history.clear()
        self._stage_episode_count = 0
        self._circuits = build_stage_circuits(self.current_stage)
        return True
