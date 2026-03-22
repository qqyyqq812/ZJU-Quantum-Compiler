"""
编译质量评估框架
================
统一的量子电路编译结果评估模块。

核心数据类: CompileResult
核心函数: evaluate_compiler, compare_compilers
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


@dataclass
class CompileResult:
    """编译结果记录"""
    circuit_name: str
    topology_name: str
    compiler_name: str
    original_depth: int
    compiled_depth: int
    original_cx: int
    compiled_cx: int
    extra_swap: int
    compile_time_ms: float
    n_qubits: int

    @property
    def depth_overhead(self) -> float:
        """深度开销比 (越小越好)"""
        if self.original_depth == 0:
            return 0.0
        return (self.compiled_depth - self.original_depth) / self.original_depth

    @property
    def cx_overhead(self) -> float:
        """CNOT 开销比 (越小越好)"""
        if self.original_cx == 0:
            return 0.0
        return (self.compiled_cx - self.original_cx) / self.original_cx


def _count_cx(circuit: QuantumCircuit) -> int:
    """统计电路中 CNOT (cx) 门的数量"""
    ops = dict(circuit.count_ops())
    return ops.get('cx', 0)


def evaluate_compiler(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    circuit_name: str = "",
    topology_name: str = "",
    compiler_name: str = "sabre",
    optimization_level: int = 1,
    compile_fn: Callable | None = None,
) -> CompileResult:
    """评估编译器在给定电路和拓扑上的表现。

    Args:
        circuit: 逻辑量子电路
        coupling_map: 目标拓扑
        circuit_name: 电路名称标识
        topology_name: 拓扑名称标识
        compiler_name: 编译器名称
        optimization_level: Qiskit 优化级别 (0-3)
        compile_fn: 自定义编译函数 (如果提供则忽略 optimization_level)
    Returns:
        CompileResult 编译结果
    """
    original_depth = circuit.depth()
    original_cx = _count_cx(circuit)

    t0 = time.perf_counter()

    if compile_fn is not None:
        compiled = compile_fn(circuit, coupling_map)
    else:
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            coupling_map=coupling_map,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x']
        )
        compiled = pm.run(circuit)

    compile_time_ms = (time.perf_counter() - t0) * 1000

    compiled_cx = _count_cx(compiled)
    extra_swap = max(0, (compiled_cx - original_cx)) // 3  # 每个 SWAP = 3 CNOT

    return CompileResult(
        circuit_name=circuit_name or circuit.name,
        topology_name=topology_name,
        compiler_name=compiler_name,
        original_depth=original_depth,
        compiled_depth=compiled.depth(),
        original_cx=original_cx,
        compiled_cx=compiled_cx,
        extra_swap=extra_swap,
        compile_time_ms=round(compile_time_ms, 2),
        n_qubits=circuit.num_qubits,
    )


def compare_compilers(results: list[CompileResult]) -> str:
    """生成 Markdown 格式的对比表。

    Args:
        results: CompileResult 列表
    Returns:
        Markdown 格式的对比表字符串
    """
    header = ("| 电路 | 拓扑 | 编译器 | 原始CX | 编译CX | 额外SWAP | "
              "原始深度 | 编译深度 | 耗时(ms) |")
    sep = "|" + "|".join(["---"] * 9) + "|"
    rows = [header, sep]

    for r in results:
        row = (f"| {r.circuit_name} | {r.topology_name} | {r.compiler_name} | "
               f"{r.original_cx} | {r.compiled_cx} | {r.extra_swap} | "
               f"{r.original_depth} | {r.compiled_depth} | {r.compile_time_ms} |")
        rows.append(row)

    return "\n".join(rows)


def save_results(results: list[CompileResult], path: str) -> None:
    """保存结果到 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_results(path: str) -> list[CompileResult]:
    """从 JSON 文件加载结果。"""
    with open(path) as f:
        data = json.load(f)
    return [CompileResult(**d) for d in data]
