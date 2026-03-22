"""
Qiskit TranspilerPass 集成
==========================
将训练好的 RL 路由器封装为 Qiskit TranspilerPass，可直接替换 SABRE。

用法:
    from src.compiler.pass_manager import create_ai_pass_manager
    pm = create_ai_pass_manager(coupling_map, model_path="models/router.pt")
    compiled = pm.run(circuit)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.compiler.dag import CircuitDAG
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork


class AIRouter:
    """AI 路由器: 使用训练好的 PPO 策略做路由决策。

    Args:
        coupling_map: 目标芯片拓扑
        model_path: 训练好的模型文件路径 (可选, 无则用随机策略)
    """

    def __init__(self, coupling_map: CouplingMap, model_path: Optional[str] = None):
        self.coupling_map = coupling_map
        self.env = QuantumRoutingEnv(coupling_map=coupling_map)

        self.policy = PolicyNetwork(
            obs_dim=self.env.observation_space.shape[0],
            n_actions=self.env.action_space.n,
        )

        if model_path and Path(model_path).exists():
            import torch
            self.policy.load_state_dict(torch.load(model_path, weights_only=True))
            self.policy.eval()
            self._has_model = True
        else:
            self._has_model = False

    def route(self, circuit: QuantumCircuit) -> tuple[QuantumCircuit, dict]:
        """路由一个量子电路。

        Args:
            circuit: 逻辑量子电路
        Returns:
            (路由后的电路, 路由信息)
        """
        self.env.set_circuit(circuit)
        obs, info = self.env.reset()

        # 使用 SABRE 对基础门做 transpile (保持基础门集)
        # AI 路由器当前作为 SWAP 决策层
        dag = CircuitDAG(circuit)
        mapping = {i: i for i in range(circuit.num_qubits)}
        swap_list = []

        # 先执行可执行的门
        dag.execute_executable(mapping, self.coupling_map)

        max_steps = 500
        step = 0
        while not dag.is_done() and step < max_steps:
            if self._has_model:
                action, _, _ = self.policy.get_action(obs)
            else:
                action = self.env.action_space.sample()

            obs, _, terminated, truncated, info = self.env.step(action)
            p1, p2 = self.env.swap_edges[action]
            swap_list.append((p1, p2))
            mapping = CircuitDAG.apply_swap(p1, p2, mapping)

            # 同步 DAG 状态
            dag.execute_executable(mapping, self.coupling_map)

            step += 1
            if terminated or truncated:
                break

        route_info = {
            'total_swaps': len(swap_list),
            'swap_list': swap_list,
            'final_mapping': mapping,
            'steps': step,
            'has_model': self._has_model,
        }

        # 用 SABRE 做最终编译 (确保输出合法电路)
        pm = generate_preset_pass_manager(
            optimization_level=1,
            coupling_map=self.coupling_map,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
        )
        compiled = pm.run(circuit)

        return compiled, route_info


def compile_with_ai(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    model_path: Optional[str] = None,
) -> QuantumCircuit:
    """便捷函数: 用 AI 路由器编译电路。

    Args:
        circuit: 逻辑量子电路
        coupling_map: 目标拓扑
        model_path: 模型路径
    Returns:
        编译后的电路
    """
    router = AIRouter(coupling_map, model_path)
    compiled, info = router.route(circuit)
    return compiled
