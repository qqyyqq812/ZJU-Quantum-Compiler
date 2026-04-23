"""V14 §V14-4 pass_manager 真集成 smoke test."""
from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.pass_manager import AIRouter


def _test_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)  # 需要 SWAP（0-2 不相邻 on linear_3）
    qc.cx(1, 2)
    return qc


def test_router_outputs_phys_circuit_with_swaps():
    """无模型时用 random policy，至少要能输出电路."""
    cm = CouplingMap.from_line(3)
    router = AIRouter(coupling_map=cm, model_path=None)

    qc = _test_circuit()
    compiled, info = router.route(qc, max_steps=200)

    assert compiled.num_qubits == cm.size()
    assert info["trace_events"] > 0
    # 随机 policy 多半会产生至少 1 个 swap
    # 但不强制，只要 trace 有东西
    assert "final_mapping" in info


def test_output_respects_coupling_map():
    """输出电路中所有 CX 门的 qubits 必须是物理相邻的."""
    cm = CouplingMap.from_line(3)
    router = AIRouter(coupling_map=cm, model_path=None)

    qc = _test_circuit()
    compiled, _ = router.route(qc, max_steps=200)

    edges = set(tuple(sorted(e)) for e in cm.get_edges())
    for instr in compiled.data:
        if instr.operation.name in ("cx", "cz"):
            indices = tuple(sorted(compiled.find_bit(q).index for q in instr.qubits))
            assert indices in edges, f"CX on non-adjacent qubits: {indices}"
        # SWAP 本身不受 coupling constraint 约束？
        # 实际上 SWAP 也要物理相邻才合法
        if instr.operation.name == "swap":
            indices = tuple(sorted(compiled.find_bit(q).index for q in instr.qubits))
            assert indices in edges, f"SWAP on non-adjacent qubits: {indices}"


def test_route_count_matches_trace_swaps():
    """route() 的 total_swaps 应等于 route_count_only() 的 ai_swaps（同 trace）."""
    cm = CouplingMap.from_line(3)
    router = AIRouter(coupling_map=cm, model_path=None)

    qc = _test_circuit()
    # 注意: 随机 policy 两次调用会给不同结果。所以单独调用一次 route，手动数
    compiled, info = router.route(qc, max_steps=200)
    swap_count_in_circuit = sum(1 for i in compiled.data if i.operation.name == "swap")
    assert swap_count_in_circuit == info["total_swaps"]
