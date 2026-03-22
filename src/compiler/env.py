"""
量子电路路由 RL 环境 V2
========================
V2 改进:
1. PASS 动作: Agent 可以选择不做 SWAP (解决 SABRE=0 但 AI=40 的问题)
2. 距离特征: 观测包含前沿门的物理距离 (告诉 Agent 迷宫方向)
3. 距离缩减奖励: 鼓励"为未来投资"的 SWAP
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.dag import CircuitDAG


class QuantumRoutingEnv(gym.Env):
    """V2 量子电路路由 RL 环境。

    关键改进:
    - Action Space: N_edges + 1 (最后一个 = PASS)
    - Observation: 映射矩阵 + 前沿mask + 距离向量 + 进度
    - Reward: 门执行 + SWAP惩罚 + 距离缩减奖励
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        coupling_map: CouplingMap,
        reward_gate: float = 1.0,
        penalty_swap: float = -0.5,
        penalty_useless_pass: float = -1.0,
        reward_done: float = 10.0,
        distance_reward_coef: float = 0.5,
        max_steps: int = 200,
    ):
        super().__init__()

        self.coupling_map = coupling_map
        self.n_physical = coupling_map.size()
        self.reward_gate = reward_gate
        self.penalty_swap = penalty_swap
        self.penalty_useless_pass = penalty_useless_pass
        self.reward_done = reward_done
        self.distance_reward_coef = distance_reward_coef
        self.max_steps = max_steps

        # 预计算距离矩阵 (物理比特间最短路径)
        self._dist_matrix = np.zeros((self.n_physical, self.n_physical), dtype=np.float32)
        for i in range(self.n_physical):
            for j in range(self.n_physical):
                try:
                    self._dist_matrix[i][j] = coupling_map.distance(i, j)
                except Exception:
                    self._dist_matrix[i][j] = self.n_physical  # unreachable

        # SWAP 边
        self.swap_edges = list(set(
            tuple(sorted(e)) for e in coupling_map.get_edges()
        ))

        # Action: N_edges 个 SWAP + 1 个 PASS
        self.n_swap_actions = len(self.swap_edges)
        self.PASS_ACTION = self.n_swap_actions  # 最后一个 action = PASS
        self.n_actions = self.n_swap_actions + 1
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation: 映射(n²) + 前沿mask(n²) + 距离向量(max_front) + 进度(1)
        self._max_front_gates = 20  # 最多跟踪的前沿门数
        obs_dim = (self.n_physical * self.n_physical +  # 映射矩阵
                   self.n_physical * self.n_physical +  # 前沿 mask
                   self._max_front_gates +               # 前沿门距离
                   1)                                    # 进度比例
        self.observation_space = spaces.Box(
            low=-1.0, high=float(self.n_physical), shape=(obs_dim,), dtype=np.float32
        )

        # 内部状态
        self._dag: CircuitDAG | None = None
        self._mapping: dict[int, int] = {}
        self._n_logical: int = 0
        self._step_count: int = 0
        self._total_swaps: int = 0
        self._total_gates_executed: int = 0
        self._total_gates: int = 0
        self._circuit: QuantumCircuit | None = None

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        """设置要编译的电路 (在 reset 之前调用)。"""
        self._circuit = circuit

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self._circuit is None:
            raise ValueError("必须先调用 set_circuit() 设置电路")

        self._dag = CircuitDAG(self._circuit)
        self._n_logical = self._circuit.num_qubits
        self._step_count = 0
        self._total_swaps = 0
        self._total_gates_executed = 0
        self._total_gates = self._dag.n_gates

        # 初始映射: 逻辑比特 i → 物理比特 i
        self._mapping = {i: i for i in range(self._n_logical)}

        # 先执行所有可直接执行的门
        executed = self._dag.execute_executable(self._mapping, self.coupling_map)
        self._total_gates_executed += executed

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        reward = 0.0

        # --- V2: 计算 SWAP/PASS 前的前沿距离 ---
        dist_before = self._compute_front_distance()

        if action == self.PASS_ACTION:
            # PASS: 不做 SWAP，尝试执行当前可执行的门
            executed = self._dag.execute_executable(self._mapping, self.coupling_map)
            if executed > 0:
                reward += executed * self.reward_gate
            else:
                # 做了 PASS 但没有门可执行 → 惩罚 (防止死循环)
                reward += self.penalty_useless_pass
            self._total_gates_executed += executed
        else:
            # SWAP
            p1, p2 = self.swap_edges[action]
            self._mapping = CircuitDAG.apply_swap(p1, p2, self._mapping)
            self._total_swaps += 1
            reward += self.penalty_swap

            # 执行新解锁的门
            executed = self._dag.execute_executable(self._mapping, self.coupling_map)
            self._total_gates_executed += executed
            reward += executed * self.reward_gate

        # --- V2: 距离缩减奖励 ---
        dist_after = self._compute_front_distance()
        if dist_before > 0:
            distance_delta = dist_before - dist_after
            reward += distance_delta * self.distance_reward_coef

        # 终止条件
        terminated = self._dag.is_done()
        truncated = self._step_count >= self.max_steps

        if terminated:
            reward += self.reward_done

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _compute_front_distance(self) -> float:
        """计算当前前沿双比特门的总物理距离。"""
        if self._dag is None:
            return 0.0
        total_dist = 0.0
        for gate in self._dag.get_two_qubit_front():
            p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < self.n_physical and p1 < self.n_physical:
                total_dist += self._dist_matrix[p0][p1]
        return total_dist

    def _get_obs(self) -> np.ndarray:
        n = self.n_physical

        # 1. 映射矩阵 (n × n)
        mapping_matrix = np.zeros((n, n), dtype=np.float32)
        for log, phys in self._mapping.items():
            if log < n and phys < n:
                mapping_matrix[log][phys] = 1.0

        # 2. 前沿 mask (n × n)
        front_matrix = np.zeros((n, n), dtype=np.float32)
        if self._dag is not None:
            for gate in self._dag.get_two_qubit_front():
                p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
                p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
                if p0 < n and p1 < n:
                    front_matrix[p0][p1] = 1.0
                    front_matrix[p1][p0] = 1.0

        # 3. V2: 前沿门距离向量
        distances = np.zeros(self._max_front_gates, dtype=np.float32)
        if self._dag is not None:
            for i, gate in enumerate(self._dag.get_two_qubit_front()):
                if i >= self._max_front_gates:
                    break
                p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
                p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
                if p0 < n and p1 < n:
                    distances[i] = self._dist_matrix[p0][p1]

        # 4. V2: 进度
        progress = np.array([
            self._dag.remaining_gates() / max(self._total_gates, 1)
        ], dtype=np.float32) if self._dag else np.zeros(1, dtype=np.float32)

        obs = np.concatenate([
            mapping_matrix.flatten(),
            front_matrix.flatten(),
            distances,
            progress,
        ])

        # Pad/truncate
        expected = self.observation_space.shape[0]
        if obs.shape[0] < expected:
            obs = np.pad(obs, (0, expected - obs.shape[0]))
        elif obs.shape[0] > expected:
            obs = obs[:expected]

        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            'total_swaps': self._total_swaps,
            'total_gates_executed': self._total_gates_executed,
            'remaining_gates': self._dag.remaining_gates() if self._dag else 0,
            'step_count': self._step_count,
            'front_distance': self._compute_front_distance(),
        }
