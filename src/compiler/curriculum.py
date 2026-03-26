"""
课程学习调度器 V7.1
==================
5 阶段渐进式训练课程 + 自适应升级条件。

V7.1 修订：
- 阈值基于 SABRE 在同规模电路上的实际 SWAP 数 (~1.2x SABRE 水平)
- 降低 min_episodes_per_stage 以加速早期课程迭代
- 增加 promotion_patience: 如果长时间卡住则自动放宽阈值

阶段 0: Warm-up    — 3Q, 1-2 CX, 学基本 SWAP 意识
阶段 1: Elementary — 5Q, depth 2-4, 学多步规划
阶段 2: Standard   — 5Q, depth 4-6, 全套 QFT/QAOA/Grover
阶段 3: Challenge  — 10Q, depth 6-10, 大规模电路
阶段 4: Master     — 20Q, depth 10-20, 芯片级真实挑战
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


# 阈值设计原理:
#   SABRE 在 linear_5 拓扑上的实际表现:
#     3Q depth 1-2: ~0-1 SWAP   →  阈值 2.0 (宽松，让 warm-up 快速通过)
#     5Q depth 2-4: ~3-8 SWAP   →  阈值 8.0 (AI 只需达到 SABRE 水平)
#     5Q depth 4-6: ~5-12 SWAP  →  阈值 12.0
#    10Q depth 6-10: ~15-25 SWAP → 阈值 25.0
#    20Q depth 10-20: ~30-60 SWAP → 阈值 50.0
STAGES = [
    StageConfig("warm-up",     3,  20, (1, 2), False, 2.0),
    StageConfig("elementary",  5,  30, (2, 4), True,  8.0),
    StageConfig("standard",    5,  40, (4, 6), True, 12.0),
    StageConfig("challenge",  10,  50, (6, 10), True, 25.0),
    StageConfig("master",     20,  60, (10, 20), True, 50.0),
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
    """自适应课程调度器 V7.2。

    V7.2 修复：
    - 拓扑感知：根据物理比特数自动限制最大课程阶段
    - 防止 5Q 拓扑上升级到 10Q/20Q 课程（会导致 truncated 假升级）
    """

    def __init__(self, max_n_qubits: int = 5, window_size: int = 50,
                 min_episodes_per_stage: int = 300,
                 promotion_patience: int = 5000):
        self.current_stage = 0
        self.window_size = window_size
        self.min_episodes_per_stage = min_episodes_per_stage
        self.promotion_patience = promotion_patience
        self._swap_history: list[float] = []
        self._stage_episode_count = 0
        self._circuits = build_stage_circuits(0)
        self._relaxation_factor = 1.0
        
        # 拓扑感知：计算最大可用阶段
        self._max_stage = 0
        for i, stage in enumerate(STAGES):
            if stage.n_qubits <= max_n_qubits:
                self._max_stage = i
        print(f"   拓扑限制: {max_n_qubits}Q → 最大课程阶段 {self._max_stage} ({STAGES[self._max_stage].name})")

    @property
    def stage_config(self) -> StageConfig:
        return STAGES[min(self.current_stage, len(STAGES) - 1)]

    @property
    def circuits(self) -> list[QuantumCircuit]:
        return self._circuits

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage >= self._max_stage

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

        # 自动宽限：如果卡住太久，逐步放宽阈值
        if self._stage_episode_count > self.promotion_patience:
            self._relaxation_factor = 1.0 + 0.2 * (
                (self._stage_episode_count - self.promotion_patience) / self.promotion_patience
            )
            self._relaxation_factor = min(self._relaxation_factor, 2.0)  # 最多放宽 2 倍

        recent_avg = sum(self._swap_history[-self.window_size:]) / self.window_size
        effective_threshold = self.stage_config.promotion_threshold * self._relaxation_factor

        if recent_avg <= effective_threshold:
            return self._promote()

        return False

    def _promote(self) -> bool:
        """升级到下一阶段。"""
        self.current_stage += 1
        self._swap_history.clear()
        self._stage_episode_count = 0
        self._relaxation_factor = 1.0  # 重置宽限
        self._circuits = build_stage_circuits(self.current_stage)
        return True
