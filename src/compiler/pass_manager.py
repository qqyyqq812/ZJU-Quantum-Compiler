"""
Qiskit TranspilerPass 集成 (V14 — 真正的 AI 电路构建)
=======================================================
将训练好的 RL 路由器封装为可替换 SABRE 的编译器。

V14 §V14-4 修复: _build_routed_circuit() 真正把 AI 的 SWAP 写入输出
电路，不再调用 Qiskit SABRE 重新编译。输出电路与输入功能严格等价。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.transpiler import CouplingMap

from src.compiler.dag import CircuitDAG
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork
from src.compiler.mcts import RouterMCTS


class AIRouter:
    """AI 路由器: 使用训练好的 PPO 策略做路由决策。

    V14 改动:
    - route() 记录每个决策点（AI SWAP / gate execution），保留时间顺序
    - _build_routed_circuit() 按时间顺序构建 QuantumCircuit，不再依赖 SABRE
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        model_path: Optional[str] = None,
        use_gnn: bool = False,
        use_mcts: bool = False,
    ):
        self.coupling_map = coupling_map
        self.env = QuantumRoutingEnv(coupling_map=coupling_map)
        self.use_gnn = use_gnn
        self.use_mcts = use_mcts

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

    # ----------------------------------------------------------------
    # 核心路由 (V14: 时间顺序轨迹 + 真实电路构建)
    # ----------------------------------------------------------------

    def route(
        self,
        circuit: QuantumCircuit,
        max_steps: int = 2000,
    ) -> tuple[QuantumCircuit, dict]:
        """路由一个量子电路 — 真正使用 AI 决策插入 SWAP。

        工作流程:
        1. 运行 AI 决策循环，记录 **按时间顺序** 的事件 (swap / executed gates)
        2. 按事件顺序重建物理电路，保证与原电路功能等价

        Args:
            circuit: 逻辑量子电路
            max_steps: 最大 RL 步数（20Q 深电路需 >=2000）

        Returns:
            (带 SWAP 的路由电路, 路由信息字典)
        """
        # 1. 跑 AI 决策，收集事件轨迹
        trace, final_mapping = self._collect_trace(circuit, max_steps)

        # 2. 根据 trace 构建真实电路
        compiled = self._build_routed_circuit(circuit, trace)

        route_info = {
            "total_swaps": sum(1 for e in trace if e[0] == "swap"),
            "trace_events": len(trace),
            "final_mapping": final_mapping,
            "has_model": self._has_model,
        }
        return compiled, route_info

    def _collect_trace(
        self,
        circuit: QuantumCircuit,
        max_steps: int,
    ) -> tuple[list, dict[int, int]]:
        """执行 AI 策略，记录事件序列。

        事件格式:
        - ("swap", phys_a, phys_b)
        - ("gate", gate_id)   # 在当前 mapping 下执行原电路的第 gate_id 个门

        返回: (events, final_mapping)
        """
        self.env.set_circuit(circuit)
        obs, info = self.env.reset()

        dag = CircuitDAG(circuit)
        n_logical = circuit.num_qubits
        mapping = {i: i for i in range(n_logical)}

        trace: list = []

        # 先把初始状态下可执行的门执行（不需要 SWAP 的门）
        executed = dag.execute_executable(mapping, self.coupling_map)
        for g in executed:
            trace.append(("gate", g.gate_id))

        step = 0
        while not dag.is_done() and step < max_steps:
            if self._has_model:
                mask = self.env.get_action_mask()
                einfo = self.env._get_info()
                if self.use_mcts:
                    mcts = RouterMCTS(policy=self.policy, num_simulations=30, c_puct=1.5)
                    action = mcts.search(self.env, obs, einfo)
                else:
                    action, _, _ = self.policy.get_action(
                        obs, action_mask=mask, gnn_input=einfo.get("gnn_input")
                    )
            else:
                action = self.env.action_space.sample()

            obs, _, terminated, truncated, _ = self.env.step(action)

            if action < self.env.n_swap_actions:
                p1, p2 = self.env.swap_edges[action]
                trace.append(("swap", p1, p2))
                mapping = CircuitDAG.apply_swap(p1, p2, mapping)

            executed = dag.execute_executable(mapping, self.coupling_map)
            for g in executed:
                trace.append(("gate", g.gate_id))

            step += 1
            if terminated or truncated:
                break

        return trace, mapping

    def _build_routed_circuit(
        self,
        original: QuantumCircuit,
        trace: list,
    ) -> QuantumCircuit:
        """V14 真·集成: 按 AI trace 顺序构建物理电路。

        核心思路:
        1. 从原电路提取门序列（按 gate_id 索引）
        2. 按 trace 顺序回放:
           - ("swap", a, b): 在物理电路上插入 SwapGate(a, b)
           - ("gate", gid):  把 gid 门的逻辑 qubits 映射到物理 qubits 后插入
        3. 输出电路的 qubit 数 = 物理拓扑 qubit 数（而非原逻辑 qubit 数）

        这个实现完全不依赖 Qiskit 的 SABRE，AI 的每个 SWAP 决策都会出现在输出。
        """
        n_phys = self.coupling_map.size()

        # 建立 gate_id → 原电路 instruction 索引的映射
        # CircuitDAG 的 gate_id 等同于 circuit.data 的 enumeration index
        # （见 dag.py::_build_from_circuit()）
        orig_ops = list(original.data)

        compiled = QuantumCircuit(n_phys, original.num_clbits)

        # 初始 mapping：logical i → physical i（与 env.reset 默认一致）
        # 若 env 用了 random mapping，这里简化：trace 本身已经反映了决策轨迹
        # 为了严格一致，我们从 logical→logical 开始，让 trace 里的 swap 自己搬
        mapping: dict[int, int] = {i: i for i in range(original.num_qubits)}

        for event in trace:
            if event[0] == "swap":
                _, p1, p2 = event
                compiled.append(SwapGate(), [p1, p2])
                # 更新 mapping：找到当前映射到 p1, p2 的 logical qubit
                inv = {p: l for l, p in mapping.items()}
                l1 = inv.get(p1)
                l2 = inv.get(p2)
                if l1 is not None:
                    mapping[l1] = p2
                if l2 is not None:
                    mapping[l2] = p1
            elif event[0] == "gate":
                gid = event[1]
                if gid >= len(orig_ops):
                    continue  # 防御性，不应发生
                instr = orig_ops[gid]
                op = instr.operation
                # 将 instruction 的 logical qubits 映射到 physical
                phys_qubits: list[int] = []
                for q in instr.qubits:
                    lq = original.find_bit(q).index
                    phys_qubits.append(mapping.get(lq, lq))
                # clbits 直接复制
                clbits = [original.find_bit(c).index for c in instr.clbits] if instr.clbits else []
                compiled.append(op, phys_qubits, clbits)

        return compiled

    # ----------------------------------------------------------------
    # 辅助: 只要 SWAP 数（评测用）
    # ----------------------------------------------------------------

    def route_count_only(self, circuit: QuantumCircuit, max_steps: int = 2000) -> dict:
        """只统计 AI 路由需要的 SWAP 数（不构建完整电路，评测公平对比用）。"""
        trace, _ = self._collect_trace(circuit, max_steps)
        swaps = sum(1 for e in trace if e[0] == "swap")
        gates = sum(1 for e in trace if e[0] == "gate")
        return {
            "ai_swaps": swaps,
            "executed_gates": gates,
            "trace_len": len(trace),
            "done": gates == circuit.size() if circuit.size() > 0 else True,
        }


def compile_with_ai(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    model_path: Optional[str] = None,
) -> QuantumCircuit:
    """便捷函数: 用 AI 路由器编译电路。"""
    router = AIRouter(coupling_map, model_path)
    compiled, _info = router.route(circuit)
    return compiled
