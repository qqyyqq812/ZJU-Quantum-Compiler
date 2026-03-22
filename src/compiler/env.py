"""
量子电路路由 RL 环境
====================
遵循 Gymnasium 标准接口，将量子电路路由建模为 MDP。

State: GNN 嵌入 + 映射矩阵 + 前沿 mask
Action: 选择 Coupling Map 上的一条边执行 SWAP
Reward: 执行的前沿门数 - SWAP 惩罚
Done: DAG 清空

EDA 类比: RL 做路由 ≈ 强化学习寻找最优布线方案
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.dag import CircuitDAG


class QuantumRoutingEnv(gym.Env):
    """量子电路路由 RL 环境。

    - 每个 episode = 编译一个量子电路
    - Agent 通过选择 SWAP 操作来满足拓扑约束
    - 目标: 最小化总 SWAP 数

    Args:
        coupling_map: 目标芯片拓扑
        reward_gate: 每执行一个前沿门的奖励
        penalty_swap: 每执行一个 SWAP 的惩罚
        reward_done: 完成编译的额外奖励
        max_steps: 最大步数 (防止无限循环)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        coupling_map: CouplingMap,
        reward_gate: float = 1.0,
        penalty_swap: float = -0.3,
        reward_done: float = 10.0,
        max_steps: int = 500,
    ):
        super().__init__()

        self.coupling_map = coupling_map
        self.n_physical = coupling_map.size()
        self.reward_gate = reward_gate
        self.penalty_swap = penalty_swap
        self.reward_done = reward_done
        self.max_steps = max_steps

        # 可用的 SWAP 边 (每条物理边 = 一个可能的 SWAP action)
        self.swap_edges = list(set(
            tuple(sorted(e)) for e in coupling_map.get_edges()
        ))
        self.n_actions = len(self.swap_edges)

        # Action space: 选择一条边执行 SWAP
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: 扁平化的状态向量
        # [映射 one-hot (n_logical * n_physical) + 前沿门 mask (n_physical * n_physical)]
        obs_dim = self.n_physical * self.n_physical + self.n_physical * self.n_physical
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 内部状态
        self._dag: CircuitDAG | None = None
        self._mapping: dict[int, int] = {}
        self._n_logical: int = 0
        self._step_count: int = 0
        self._total_swaps: int = 0
        self._total_gates_executed: int = 0
        self._circuit: QuantumCircuit | None = None

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        """设置要编译的电路 (在 reset 之前调用)。"""
        self._circuit = circuit

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """重置环境，开始新的编译 episode。"""
        super().reset(seed=seed)

        if self._circuit is None:
            raise ValueError("必须先调用 set_circuit() 设置电路")

        self._dag = CircuitDAG(self._circuit)
        self._n_logical = self._circuit.num_qubits
        self._step_count = 0
        self._total_swaps = 0
        self._total_gates_executed = 0

        # 初始映射: 逻辑比特 i → 物理比特 i
        self._mapping = {i: i for i in range(self._n_logical)}

        # 先执行所有可直接执行的门
        executed = self._dag.execute_executable(self._mapping, self.coupling_map)
        self._total_gates_executed += executed

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步: 选择一条边执行 SWAP。

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        reward = 0.0

        # 1. 执行 SWAP
        p1, p2 = self.swap_edges[action]
        self._mapping = CircuitDAG.apply_swap(p1, p2, self._mapping)
        self._total_swaps += 1
        reward += self.penalty_swap

        # 2. 执行所有变得可执行的门
        executed = self._dag.execute_executable(self._mapping, self.coupling_map)
        self._total_gates_executed += executed
        reward += executed * self.reward_gate

        # 3. 检查终止条件
        terminated = self._dag.is_done()
        truncated = self._step_count >= self.max_steps

        if terminated:
            reward += self.reward_done

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """构建观测向量。"""
        n = self.n_physical

        # 1. 映射矩阵 (n_physical × n_physical): entry[i][j] = 1 if logical i → physical j
        mapping_matrix = np.zeros((n, n), dtype=np.float32)
        for log, phys in self._mapping.items():
            if log < n and phys < n:
                mapping_matrix[log][phys] = 1.0

        # 2. 前沿门 mask (n_physical × n_physical): entry[i][j] = 1 if 前沿有门(logical_i, logical_j)
        front_matrix = np.zeros((n, n), dtype=np.float32)
        if self._dag is not None:
            for gate in self._dag.get_two_qubit_front():
                q0, q1 = gate.qubits[0], gate.qubits[1]
                p0 = self._mapping.get(q0, q0)
                p1 = self._mapping.get(q1, q1)
                if p0 < n and p1 < n:
                    front_matrix[p0][p1] = 1.0
                    front_matrix[p1][p0] = 1.0

        obs = np.concatenate([mapping_matrix.flatten(), front_matrix.flatten()])

        # Pad or truncate to match observation_space
        expected_size = self.observation_space.shape[0]
        if obs.shape[0] < expected_size:
            obs = np.pad(obs, (0, expected_size - obs.shape[0]))
        elif obs.shape[0] > expected_size:
            obs = obs[:expected_size]

        return obs

    def _get_info(self) -> dict[str, Any]:
        """返回额外信息。"""
        return {
            'total_swaps': self._total_swaps,
            'total_gates_executed': self._total_gates_executed,
            'remaining_gates': self._dag.remaining_gates() if self._dag else 0,
            'step_count': self._step_count,
        }
