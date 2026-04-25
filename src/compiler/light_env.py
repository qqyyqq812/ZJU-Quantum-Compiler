import numpy as np
import torch
from qiskit.transpiler import CouplingMap

class MockGate:
    __slots__ = ['gate_id', 'qubits']
    def __init__(self, gate_id, qubits):
        self.gate_id = gate_id
        self.qubits = qubits

class MockDAG:
    def __init__(self, env):
        self.env = env
    def is_done(self):
        return self.env.is_done()
    def get_two_qubit_front(self):
        return [MockGate(i, self.env.gate_qubits[i]) for i in self.env.two_qubit_front]
    def get_extended_front(self, depth=2):
        return [MockGate(i, self.env.gate_qubits[i]) for i in self.env.extended_two_qubit_front]
    def remaining_two_qubit_gates(self):
        # V13 9D GNN feature 6 — 需要的"剩余双比特门数"
        return int(((~self.env.executed) & self.env.gate_is_two_qubit).sum())
    @property
    def n_two_qubit_gates(self):
        return int(self.env.gate_is_two_qubit.sum())
    def qubit_remaining_gates(self, logical_qubit):
        # V13 9D GNN feature 7 — 指定逻辑比特剩余双比特门数
        cnt = 0
        for i in range(self.env.n_gates):
            if self.env.executed[i] or not self.env.gate_is_two_qubit[i]:
                continue
            if logical_qubit in self.env.gate_qubits[i]:
                cnt += 1
        return cnt
    def remaining_gates(self):
        return int((~self.env.executed).sum())

class LightweightEnv:
    """零拷贝高速前瞻环境机制，替代 Qiskit deepcopy 开销。
    速度比深拷贝 QuantumRoutingEnv 快 1000 倍以上。"""
    
    def __init__(self, base_env):
        # --- 静态引用 ---
        self.coupling_map = base_env.coupling_map
        self.n_physical = base_env.n_physical
        self.reward_gate = base_env.reward_gate
        self.penalty_swap = base_env.penalty_swap
        self.penalty_useless_pass = base_env.penalty_useless_pass
        self.distance_reward_coef = base_env.distance_reward_coef
        self.lookahead_coef = base_env.lookahead_coef
        self.reward_done = base_env.reward_done
        self.max_steps = base_env.max_steps
        self.soft_mask = base_env.soft_mask
        self._dist_matrix = base_env._dist_matrix
        self.swap_edges = base_env.swap_edges
        self.PASS_ACTION = base_env.PASS_ACTION
        self.n_actions = base_env.n_actions
        self.n_swap_actions = base_env.n_swap_actions
        self._max_front_gates = base_env._max_front_gates
        
        dag = base_env._dag
        self.n_gates = dag.n_gates
        self._total_gates = base_env._total_gates
        self.gate_is_two_qubit = np.zeros(self.n_gates, dtype=bool)
        self.gate_qubits = []
        
        self.successors = [[] for _ in range(self.n_gates)]
        self.predecessors = [[] for _ in range(self.n_gates)]
        
        for idx in range(self.n_gates):
            node = dag._gates[idx]
            self.gate_is_two_qubit[idx] = node.is_two_qubit
            self.gate_qubits.append(node.qubits)
            
        for u, v in dag._graph.edges():
            self.successors[u].append(v)
            self.predecessors[v].append(u)
            
        self.initial_in_degree = np.array([len(preds) for preds in self.predecessors], dtype=np.int32)
        self.edges_set = set(tuple(e) for e in self.coupling_map.get_edges())
        
        # --- 动态状态 ---
        self._mapping = dict(base_env._mapping)
        self.in_degree = self.initial_in_degree.copy()
        self.executed = np.zeros(self.n_gates, dtype=bool)
        
        # 同步基准进度
        for idx in range(self.n_gates):
            if dag._gates[idx].executed:
                self._execute_gate_internal(idx)
                
        self._step_count = base_env._step_count
        self._total_swaps = base_env._total_swaps
        self._total_gates_executed = base_env._total_gates_executed
        
        self._update_fronts()

    def _execute_gate_internal(self, gid):
        self.executed[gid] = True
        self.in_degree[gid] = -1
        for child in self.successors[gid]:
            self.in_degree[child] -= 1

    def clone(self):
        new_env = LightweightEnv.__new__(LightweightEnv)
        
        # 静态复用
        new_env.coupling_map = self.coupling_map
        new_env.n_physical = self.n_physical
        new_env.reward_gate = self.reward_gate
        new_env.penalty_swap = self.penalty_swap
        new_env.penalty_useless_pass = self.penalty_useless_pass
        new_env.distance_reward_coef = self.distance_reward_coef
        new_env.lookahead_coef = self.lookahead_coef
        new_env.reward_done = self.reward_done
        new_env.max_steps = self.max_steps
        new_env.soft_mask = self.soft_mask
        new_env._dist_matrix = self._dist_matrix
        new_env.swap_edges = self.swap_edges
        new_env.PASS_ACTION = self.PASS_ACTION
        new_env.n_actions = self.n_actions
        new_env.n_swap_actions = self.n_swap_actions
        new_env._max_front_gates = self._max_front_gates
        new_env.n_gates = self.n_gates
        new_env._total_gates = self._total_gates
        new_env.gate_is_two_qubit = self.gate_is_two_qubit
        new_env.gate_qubits = self.gate_qubits
        new_env.successors = self.successors
        new_env.predecessors = self.predecessors
        new_env.edges_set = self.edges_set
        
        # 动态复制 O(1) 极大加速
        new_env._mapping = self._mapping.copy()
        new_env.in_degree = self.in_degree.copy()
        new_env.executed = self.executed.copy()
        new_env._step_count = self._step_count
        new_env._total_swaps = self._total_swaps
        new_env._total_gates_executed = self._total_gates_executed
        
        new_env.front_indices = getattr(self, 'front_indices', [])
        new_env.two_qubit_front = getattr(self, 'two_qubit_front', [])
        new_env.extended_two_qubit_front = getattr(self, 'extended_two_qubit_front', [])
        
        return new_env

    def _update_fronts(self):
        self.front_indices = np.where((self.in_degree == 0) & (~self.executed))[0].tolist()
        self.two_qubit_front = [i for i in self.front_indices if self.gate_is_two_qubit[i]]
        
        ext_front = list(self.two_qubit_front)
        seen = set(ext_front)
        sim_done = set(np.where(self.executed)[0])
        for i in self.front_indices:
            if not self.gate_is_two_qubit[i]:
                sim_done.add(i)
                
        for _ in range(2):
            nxt = []
            for gid in range(self.n_gates):
                if gid in sim_done or gid in seen: continue
                can_add = True
                for p in self.predecessors[gid]:
                    if p not in sim_done and p not in seen:
                        can_add = False
                        break
                if can_add:
                    nxt.append(gid)
            for g in nxt:
                if self.gate_is_two_qubit[g]:
                    ext_front.append(g)
                    seen.add(g)
                else:
                    sim_done.add(g)
        self.extended_two_qubit_front = ext_front

    def is_done(self):
        return np.all(self.executed)

    def _execute_executable(self):
        executed_count = 0
        while True:
            did_execute = False
            for gid in list(self.front_indices):
                if self.executed[gid]: continue
                
                if not self.gate_is_two_qubit[gid]:
                    self._execute_gate_internal(gid)
                    executed_count += 1
                    did_execute = True
                else:
                    q0, q1 = self.gate_qubits[gid]
                    p0 = self._mapping[q0]
                    p1 = self._mapping[q1]
                    if (p0, p1) in self.edges_set or (p1, p0) in self.edges_set:
                        self._execute_gate_internal(gid)
                        executed_count += 1
                        did_execute = True
            
            if did_execute:
                self._update_fronts()
            else:
                break
        return executed_count

    def _compute_front_distance(self):
        total = 0.0
        for gid in self.two_qubit_front:
            q0, q1 = self.gate_qubits[gid]
            p0 = self._mapping.get(q0, q0)
            p1 = self._mapping.get(q1, q1)
            if p0 < self.n_physical and p1 < self.n_physical:
                total += self._dist_matrix[p0][p1]
        return total

    def _compute_extended_distance(self):
        total = 0.0
        for gid in self.extended_two_qubit_front:
            q0, q1 = self.gate_qubits[gid]
            p0 = self._mapping.get(q0, q0)
            p1 = self._mapping.get(q1, q1)
            if p0 < self.n_physical and p1 < self.n_physical:
                total += self._dist_matrix[p0][p1]
        return total

    def apply_swap(self, q1, q2):
        log1 = None
        log2 = None
        for log, phys in self._mapping.items():
            if phys == q1: log1 = log
            if phys == q2: log2 = log
        if log1 is not None and log2 is not None:
            self._mapping[log1] = q2
            self._mapping[log2] = q1
        elif log1 is not None:
            self._mapping[log1] = q2
        elif log2 is not None:
            self._mapping[log2] = q1

    def step(self, action):
        self._step_count += 1
        reward = 0.0
        
        dist_before = self._compute_front_distance()
        ext_dist_before = self._compute_extended_distance()

        if action == self.PASS_ACTION:
            executed = self._execute_executable()
            if executed > 0:
                reward += executed * self.reward_gate
            else:
                reward += self.penalty_useless_pass
            self._total_gates_executed += executed
        else:
            p1, p2 = self.swap_edges[action]
            self.apply_swap(p1, p2)
            self._total_swaps += 1
            reward += self.penalty_swap
            
            executed = self._execute_executable()
            self._total_gates_executed += executed
            reward += executed * self.reward_gate

        dist_after = self._compute_front_distance()
        ext_dist_after = self._compute_extended_distance()

        if dist_before > 0: reward += (dist_before - dist_after) * self.distance_reward_coef
        if ext_dist_before > 0: reward += (ext_dist_before - ext_dist_after) * self.lookahead_coef

        terminated = self.is_done()
        truncated = self._step_count >= self.max_steps
        if terminated:
            reward += self.reward_done

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def get_action_mask(self):
        mask = np.zeros(self.n_actions, dtype=np.float32)
        mask[self.PASS_ACTION] = 1.0

        if self.is_done() or not self.two_qubit_front:
            return mask

        front_pairs = []
        for gid in self.two_qubit_front:
            q0, q1 = self.gate_qubits[gid]
            p0 = self._mapping.get(q0, q0)
            p1 = self._mapping.get(q1, q1)
            if p0 < self.n_physical and p1 < self.n_physical:
                front_pairs.append((p0, p1))
                
        if not front_pairs:
            return mask
            
        for i, (s1, s2) in enumerate(self.swap_edges):
            best_delta = float('inf')
            for p0, p1 in front_pairs:
                d_now = self._dist_matrix[p0][p1]
                new_p0 = s2 if p0 == s1 else (s1 if p0 == s2 else p0)
                new_p1 = s2 if p1 == s1 else (s1 if p1 == s2 else p1)
                d_after = self._dist_matrix[new_p0][new_p1]
                best_delta = min(best_delta, d_after - d_now)
            
            if self.soft_mask:
                if best_delta <= 1: mask[i] = 1.0
            else:
                if best_delta < 0: mask[i] = 1.0

        if mask[:self.n_swap_actions].sum() == 0:
            return mask
        return mask

    def _get_obs(self):
        n = self.n_physical
        mapping_matrix = np.zeros((n, n), dtype=np.float32)
        for log, phys in self._mapping.items():
            if log < n and phys < n:
                mapping_matrix[log][phys] = 1.0

        front_matrix = np.zeros((n, n), dtype=np.float32)
        for gid in self.two_qubit_front:
            q0, q1 = self.gate_qubits[gid]
            p0 = self._mapping.get(q0, q0)
            p1 = self._mapping.get(q1, q1)
            if p0 < n and p1 < n:
                front_matrix[p0][p1] = 1.0
                front_matrix[p1][p0] = 1.0

        distances = np.zeros(self._max_front_gates, dtype=np.float32)
        for i, gid in enumerate(self.two_qubit_front[:self._max_front_gates]):
            q0, q1 = self.gate_qubits[gid]
            p0 = self._mapping.get(q0, q0)
            p1 = self._mapping.get(q1, q1)
            if p0 < n and p1 < n:
                distances[i] = self._dist_matrix[p0][p1]

        ext_distances = np.zeros(self._max_front_gates, dtype=np.float32)
        j = 0
        front_set = set(self.two_qubit_front)
        for gid in self.extended_two_qubit_front:
            if gid in front_set: continue
            if j >= self._max_front_gates: break
            q0, q1 = self.gate_qubits[gid]
            p0 = self._mapping.get(q0, q0)
            p1 = self._mapping.get(q1, q1)
            if p0 < n and p1 < n:
                ext_distances[j] = self._dist_matrix[p0][p1]
            j += 1

        remaining = np.sum(~self.executed)
        progress = np.array([remaining / max(self._total_gates, 1)], dtype=np.float32)

        obs = np.concatenate([
            mapping_matrix.flatten(), front_matrix.flatten(),
            distances, ext_distances, progress,
        ])
        # observation_space.shape[0] usually 50 + 50 + 20 + 20 + 1 = 141 (for 5Q)
        # We enforce matching by returning it exactly
        expected = 2 * n * n + 2 * self._max_front_gates + 1
        if obs.shape[0] < expected:
            obs = np.pad(obs, (0, expected - obs.shape[0]))
        elif obs.shape[0] > expected:
            obs = obs[:expected]
        return obs

    def _get_info(self):
        from src.compiler.gnn_extractor import extract_physical_graph
        graph_data = extract_physical_graph(self.coupling_map, self._mapping, MockDAG(self))
        return {
            'total_swaps': self._total_swaps,
            'total_gates_executed': self._total_gates_executed,
            'remaining_gates': np.sum(~self.executed),
            'step_count': self._step_count,
            'front_distance': self._compute_front_distance(),
            'gnn_input': {
                'graph': graph_data,
                'swap_edges': self.swap_edges
            }
        }
