"""State environment used by the NPQR neural routing runtime."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.dag import CircuitDAG


class QuantumRoutingEnv(gym.Env):
    """Quantum routing state machine for NPQR inference.

    The environment tracks the logical-to-physical mapping, executable front
    gates, valid SWAP actions, and fixed-size neural observation features.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        coupling_map: CouplingMap,
        reward_gate: float = 0.3,
        penalty_swap: float = -0.5,
        penalty_useless_pass: float = -1.0,
        reward_done: float = 0.0,
        distance_reward_coef: float = 0.3,
        lookahead_coef: float = 0.1,
        max_steps: int = 2000,
        initial_mapping_fn=None,
        soft_mask: bool = True,
        tabu_size: int = 4,
        penalty_tabu: float = -5.0,
        early_stage_reward_floor: float = 5.0,
    ):
        super().__init__()

        self.coupling_map = coupling_map
        self.n_physical = coupling_map.size()
        self.reward_gate = reward_gate
        self.penalty_swap = penalty_swap
        self.penalty_useless_pass = penalty_useless_pass
        self.reward_done = reward_done
        self.distance_reward_coef = distance_reward_coef
        self.lookahead_coef = lookahead_coef
        self.max_steps = max_steps
        self.initial_mapping_fn = initial_mapping_fn
        self.soft_mask = soft_mask
        self.tabu_size = tabu_size
        self.penalty_tabu = penalty_tabu
        self.early_stage_reward_floor = early_stage_reward_floor
        self._sabre_swaps: int = 0
        self.tabu_list = []

        # 预计算距离矩阵
        self._dist_matrix = np.zeros((self.n_physical, self.n_physical), dtype=np.float32)
        for i in range(self.n_physical):
            for j in range(self.n_physical):
                try:
                    self._dist_matrix[i][j] = coupling_map.distance(i, j)
                except Exception:
                    self._dist_matrix[i][j] = self.n_physical

        # SWAP 边
        self.swap_edges = list(set(
            tuple(sorted(e)) for e in coupling_map.get_edges()
        ))

        # Action: N_edges + 1 (PASS)
        self.n_swap_actions = len(self.swap_edges)
        self.PASS_ACTION = self.n_swap_actions
        self.n_actions = self.n_swap_actions + 1
        self.action_space = spaces.Discrete(self.n_actions)

        # Fixed-size observation encoding shared by all supported topologies.
        self._max_front_gates = 20
        self._max_swap_edges = 60  # 20Q 拓扑约有 36 条 SWAP 边，余量至 60
        # obs = [per_edge_features(MAX_EDGES*4) + front_dist(MAX_FRONT) + ext_dist(MAX_FRONT) + stats(10) + progress(1)]
        obs_dim = (self._max_swap_edges * 4 +   # 每条 SWAP 边 4 维特征
                   self._max_front_gates +        # 前沿门距离
                   self._max_front_gates +        # look-ahead 距离
                   10 +                            # 聚合统计量
                   1)                              # 进度
        self.observation_space = spaces.Box(
            low=-10.0, high=100.0, shape=(obs_dim,), dtype=np.float32
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
        self._topology_name: str = "unknown"

    def set_circuit(self, circuit: QuantumCircuit, topology_name: str | None = None) -> None:
        self._circuit = circuit
        if topology_name:
            self._topology_name = topology_name

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
        self.tabu_list.clear()

        self._sabre_swaps = 0

        if self.initial_mapping_fn is not None:
            self._mapping = self.initial_mapping_fn(self._circuit, self.coupling_map)
        else:
            self._mapping = {i: i for i in range(self._n_logical)}

        executed_gates = self._dag.execute_executable(self._mapping, self.coupling_map)
        self._total_gates_executed += len(executed_gates)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        reward = 0.0

        # 前沿 + look-ahead 距离
        dist_before = self._compute_front_distance()
        ext_dist_before = self._compute_extended_distance()

        if action == self.PASS_ACTION:
            executed_gates = self._dag.execute_executable(self._mapping, self.coupling_map)
            executed = len(executed_gates)
            if executed > 0:
                reward += executed * self.reward_gate
            else:
                reward += self.penalty_useless_pass
            self._total_gates_executed += executed
        else:
            p1, p2 = self.swap_edges[action]
            self._mapping = CircuitDAG.apply_swap(p1, p2, self._mapping)
            self._total_swaps += 1
            reward += self.penalty_swap
            # 更新禁忌表 (FIFO)
            if self.tabu_size > 0:
                self.tabu_list.append(action)
                if len(self.tabu_list) > self.tabu_size:
                    self.tabu_list.pop(0)

            executed_gates = self._dag.execute_executable(self._mapping, self.coupling_map)
            executed = len(executed_gates)
            self._total_gates_executed += executed
            reward += executed * self.reward_gate

        # 距离缩减奖励 (前沿 + look-ahead)
        dist_after = self._compute_front_distance()
        ext_dist_after = self._compute_extended_distance()

        if dist_before > 0:
            reward += (dist_before - dist_after) * self.distance_reward_coef
        if ext_dist_before > 0:
            reward += (ext_dist_before - ext_dist_after) * self.lookahead_coef

        terminated = self._dag.is_done()
        truncated = self._step_count >= self.max_steps
        if terminated:
            reward += max(self.reward_done, self.early_stage_reward_floor)
        elif truncated:
            remaining = self._dag.remaining_gates() if hasattr(self._dag, 'remaining_gates') else 0
            reward -= float(remaining) * 1.0  # 每个剩余门 -1

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def get_action_mask(self) -> np.ndarray:
        """Return the valid action mask for the current front layer.

        SWAP 只保留能缩减某个前沿/扩展前沿门物理距离的。
        PASS 始终可用。
        """
        mask = np.zeros(self.n_actions, dtype=np.float32)
        mask[self.PASS_ACTION] = 1.0  # PASS 始终可用

        if self._dag is None or self._dag.is_done():
            return mask

        front = self._dag.get_extended_front(depth=2)
        if not front:
            return mask

        # 收集前沿门涉及的物理比特对
        front_pairs = []
        for gate in front:
            p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < self.n_physical and p1 < self.n_physical:
                front_pairs.append((p0, p1))

        if not front_pairs:
            return mask

        # 对每条 SWAP 边: 检查是否能缩短任何前沿门距离
        for i, (s1, s2) in enumerate(self.swap_edges):
            best_delta = float('inf')  # 最佳距离变化
            for p0, p1 in front_pairs:
                d_now = self._dist_matrix[p0][p1]
                # 模拟 SWAP(s1, s2) 后的新位置
                new_p0 = s2 if p0 == s1 else (s1 if p0 == s2 else p0)
                new_p1 = s2 if p1 == s1 else (s1 if p1 == s2 else p1)
                d_after = self._dist_matrix[new_p0][new_p1]
                delta = d_after - d_now
                best_delta = min(best_delta, delta)
            
            if not self.soft_mask:
                # Hard mask: 严格缩短
                if best_delta < 0:
                    mask[i] = 1.0
            else:
                # Soft mask: keep mildly neutral actions for beam expansion.
                if best_delta <= 1:
                    mask[i] = 1.0

        # 如果没有有用的 SWAP（所有前沿门已可执行），只留 PASS
        if mask[:self.n_swap_actions].sum() == 0:
            return mask
            
        # Remove recently reversed SWAPs from the local action set.
        for tabu_action in self.tabu_list:
            if tabu_action < self.n_swap_actions:
                mask[tabu_action] = 0.0

        return mask

    def _compute_front_distance(self) -> float:
        if self._dag is None:
            return 0.0
        total = 0.0
        for gate in self._dag.get_two_qubit_front():
            p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < self.n_physical and p1 < self.n_physical:
                total += self._dist_matrix[p0][p1]
        return total

    def _compute_extended_distance(self) -> float:
        """Look-ahead 距离: 前沿 + 后续层的双比特门距离。"""
        if self._dag is None:
            return 0.0
        total = 0.0
        for gate in self._dag.get_extended_front(depth=2):
            p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < self.n_physical and p1 < self.n_physical:
                total += self._dist_matrix[p0][p1]
        return total

    def _get_obs(self) -> np.ndarray:
        """Return the fixed-size NPQR neural observation vector.
        
        核心思想：用 SWAP 边为锚点编码局部距离差分特征，实现观测维度与 N 无关。
        """
        n = self.n_physical

        # --- 1. 前沿门距离 ---
        two_q_front = self._dag.get_two_qubit_front() if self._dag else []
        distances = np.zeros(self._max_front_gates, dtype=np.float32)
        front_pairs = []  # (p0, p1) 物理比特对
        for i, gate in enumerate(two_q_front):
            if i >= self._max_front_gates:
                break
            p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
            p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
            if p0 < n and p1 < n:
                distances[i] = self._dist_matrix[p0][p1]
                front_pairs.append((p0, p1))

        # --- 2. Look-ahead 距离 ---
        ext_distances = np.zeros(self._max_front_gates, dtype=np.float32)
        if self._dag is not None:
            extended = self._dag.get_extended_front(depth=2)
            front_ids = {g.gate_id for g in two_q_front}
            j = 0
            for gate in extended:
                if gate.gate_id in front_ids:
                    continue
                if j >= self._max_front_gates:
                    break
                p0 = self._mapping.get(gate.qubits[0], gate.qubits[0])
                p1 = self._mapping.get(gate.qubits[1], gate.qubits[1])
                if p0 < n and p1 < n:
                    ext_distances[j] = self._dist_matrix[p0][p1]
                j += 1

        # --- 3. 每条 SWAP 边的特征 (4维/边) ---
        # [best_delta, worst_delta, mean_delta, n_improved]
        edge_features = np.zeros(self._max_swap_edges * 4, dtype=np.float32)
        for idx, (s1, s2) in enumerate(self.swap_edges):
            if idx >= self._max_swap_edges:
                break
            if not front_pairs:
                continue
            deltas = []
            for p0, p1 in front_pairs:
                d_now = self._dist_matrix[p0][p1]
                new_p0 = s2 if p0 == s1 else (s1 if p0 == s2 else p0)
                new_p1 = s2 if p1 == s1 else (s1 if p1 == s2 else p1)
                d_after = self._dist_matrix[new_p0][new_p1]
                deltas.append(d_after - d_now)
            base = idx * 4
            edge_features[base + 0] = min(deltas)                        # best delta
            edge_features[base + 1] = max(deltas)                        # worst delta
            edge_features[base + 2] = sum(deltas) / len(deltas)          # mean delta
            edge_features[base + 3] = sum(1 for d in deltas if d < 0)    # n_improved

        # --- 4. 聚合统计量 (10维固定) ---
        stats = np.zeros(10, dtype=np.float32)
        valid_dists = distances[distances > 0]
        stats[0] = len(front_pairs)                               # num_front_gates
        stats[1] = float(np.sum(valid_dists)) if len(valid_dists) > 0 else 0  # sum_dist
        stats[2] = float(np.mean(valid_dists)) if len(valid_dists) > 0 else 0 # mean_dist
        stats[3] = float(np.max(valid_dists)) if len(valid_dists) > 0 else 0  # max_dist
        stats[4] = float(np.min(valid_dists)) if len(valid_dists) > 0 else 0  # min_dist
        stats[5] = len(self.swap_edges)                            # num_swap_edges
        stats[6] = self.n_physical                                  # n_physical
        stats[7] = self._total_swaps                                # total_swaps_so_far
        stats[8] = self._step_count                                 # steps_taken
        stats[9] = self._dag.remaining_two_qubit_gates() if self._dag else 0  # remaining_2q

        # --- 5. 进度 ---
        progress = np.array([
            self._dag.remaining_gates() / max(self._total_gates, 1)
        ], dtype=np.float32) if self._dag else np.zeros(1, dtype=np.float32)

        obs = np.concatenate([
            edge_features, distances, ext_distances, stats, progress,
        ])

        expected = self.observation_space.shape[0]
        if obs.shape[0] < expected:
            obs = np.pad(obs, (0, expected - obs.shape[0]))
        elif obs.shape[0] > expected:
            obs = obs[:expected]

        return obs

    def _get_info(self) -> dict[str, Any]:
        from src.compiler.gnn_extractor import extract_physical_graph
        graph_data = extract_physical_graph(self.coupling_map, self._mapping, self._dag)
        return {
            'total_swaps': self._total_swaps,
            'sabre_swaps': self._sabre_swaps,
            'total_gates_executed': self._total_gates_executed,
            'remaining_gates': self._dag.remaining_gates() if self._dag else 0,
            'step_count': self._step_count,
            'front_distance': self._compute_front_distance(),
            'gnn_input': {
                'graph': graph_data,
                'swap_edges': self.swap_edges
            }
        }
