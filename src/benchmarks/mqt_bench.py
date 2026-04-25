"""
MQT-Bench 电路加载器 (V14 评测管线)
=====================================
封装 Munich Quantum Toolkit Benchmark Library (mqt.bench v2.x)，
返回 Qiskit QuantumCircuit。

设计要点（遵守 .claude/rules/code-and-config.md）：
- 失败时 fallback 到 src/benchmarks/circuits.py（项目自己的电路生成器），
  不静默吞掉异常 — 必须 log warning。
- 评测公平起见：默认拉 INDEP level（已分解到 1Q+2Q 门，独立于硬件映射），
  并把所有非 measure/barrier 的复合门 transpile 到 basis_gates，
  确保 qiskit transpiler 与 AIRouter 看到同样的 2Q gate 数。
- 至少覆盖 4 种电路类型 × 3 种 qubit 数 = 12 条标准电路。

参考：
- https://github.com/munich-quantum-toolkit/bench
- mqt.bench v2.2.2 API: get_benchmark(name, level, qubits)
"""
from __future__ import annotations

import logging
from typing import Callable

from qiskit import QuantumCircuit, transpile

logger = logging.getLogger(__name__)

# 标准 IBM basis gates — 评测时所有电路统一到这套门，公平比对 SWAP 数。
DEFAULT_BASIS_GATES = ["cx", "id", "rz", "sx", "x"]

# 5 种代表性电路类型 × 3 种规模（5/10/20 qubit）= 15 条
# 注: mqt.bench 不直接收 "vqe"，会落到 "vqe_real_amp"。fallback 时退化到 _vqe_fallback。
DEFAULT_BENCHMARKS = [
    ("qft", "QFT — 量子傅里叶变换 (密集 CNOT)"),
    ("qaoa", "QAOA — 变分量子优化 (rzz 交互)"),
    ("grover", "Grover — 量子搜索 (重复 Oracle+Diffusion)"),
    ("ghz", "GHZ — 多体纠缠态制备 (链式 CNOT)"),
    ("vqe", "VQE — 变分量子求解器 (RealAmplitudes ansatz)"),
]

# 用户友好别名 → mqt.bench 内部 benchmark 名（v2.x catalog）
MQT_NAME_ALIASES = {
    "vqe": "vqe_real_amp",
}


def _try_mqt(
    name: str,
    n_qubits: int,
    basis_gates: list[str],
) -> QuantumCircuit | None:
    """尝试用 mqt.bench 拉取指定电路；失败时返回 None（调用方决定 fallback）。"""
    try:
        from mqt.bench import BenchmarkLevel, get_benchmark
    except ImportError as exc:
        logger.warning(
            "mqt.bench 未安装 (%s)；将 fallback 到项目自带 circuits.py。"
            " 安装命令：pip install mqt.bench",
            exc,
        )
        return None

    # 把别名映射到 mqt.bench v2.x 内部名（如 vqe → vqe_real_amp）
    mqt_name = MQT_NAME_ALIASES.get(name, name)
    try:
        qc = get_benchmark(mqt_name, BenchmarkLevel.INDEP, n_qubits)
    except Exception as exc:  # pragma: no cover - 取决于 mqt.bench 内部状态
        logger.warning(
            "mqt.bench get_benchmark(%s, INDEP, %d) 失败: %s", mqt_name, n_qubits, exc
        )
        return None

    # mqt.bench 返回的 INDEP 电路里可能含 measure/barrier 和未分解复合门 (如 grover 的 'Q')
    # 我们要 routing 视角的 2Q 门数，所以剥掉 measure 并 transpile 到 basis_gates。
    qc_no_measure = qc.remove_final_measurements(inplace=False)
    if qc_no_measure is None:
        # remove_final_measurements 在 inplace=False 下返回新电路；某些版本对 measure-free
        # 电路返回 None。退而求其次：手动 copy。
        qc_no_measure = qc.copy()

    try:
        qc_basis = transpile(
            qc_no_measure,
            basis_gates=basis_gates,
            optimization_level=0,
            seed_transpiler=42,
        )
    except Exception as exc:
        logger.warning(
            "transpile %s_%d 到 basis_gates 失败: %s — 返回未分解电路",
            name,
            n_qubits,
            exc,
        )
        qc_basis = qc_no_measure

    qc_basis.name = f"{name}_{n_qubits}"
    return qc_basis


def _fallback_circuit(name: str, n_qubits: int) -> QuantumCircuit | None:
    """fallback 到项目自带 circuits.py 的电路生成器。"""
    try:
        from src.benchmarks.circuits import (
            generate_grover,
            generate_qaoa,
            generate_qft,
            generate_random,
        )
    except ImportError as exc:
        logger.error("fallback 失败：无法 import src.benchmarks.circuits — %s", exc)
        return None

    builders: dict[str, Callable[[int], QuantumCircuit]] = {
        "qft": lambda n: generate_qft(n),
        "qaoa": lambda n: generate_qaoa(n, p=1),
        "grover": lambda n: generate_grover(n, num_iterations=1),
        "ghz": lambda n: _ghz_fallback(n),
        "vqe": lambda n: _vqe_fallback(n),
        "random": lambda n: generate_random(n, depth=max(n, 10), seed=42),
    }

    if name not in builders:
        logger.warning("fallback 不识别电路类型 %s — 返回 None", name)
        return None

    try:
        qc = builders[name](n_qubits)
        qc.name = f"{name}_{n_qubits}"
        return qc
    except Exception as exc:
        logger.warning("fallback 生成 %s_%d 失败: %s", name, n_qubits, exc)
        return None


def _ghz_fallback(n_qubits: int) -> QuantumCircuit:
    """简单 GHZ 制备：H + 链式 CNOT。"""
    qc = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def _vqe_fallback(n_qubits: int) -> QuantumCircuit:
    """RealAmplitudes-style VQE ansatz fallback (1 layer): RY + 全连接 CNOT 链。

    与 mqt.bench 的 vqe_real_amp 形状一致（2-local entanglement, linear），
    用于 mqt.bench 不可用时仍能产出可比对的 VQE 电路。
    """
    import numpy as np

    qc = QuantumCircuit(n_qubits, name=f"vqe_{n_qubits}")
    rng = np.random.default_rng(seed=42)
    # 第一层 RY
    for q in range(n_qubits):
        qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
    # entanglement: 链式 CX
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    # 第二层 RY
    for q in range(n_qubits):
        qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
    return qc


def get_mqt_circuits(
    n_qubits_list: list[int] | None = None,
    benchmark_names: list[str] | None = None,
    basis_gates: list[str] | None = None,
) -> dict[str, QuantumCircuit]:
    """加载 MQT-Bench 标准电路集合。

    Args:
        n_qubits_list: 要测试的比特数列表，默认 [5, 10, 20]
        benchmark_names: 电路类型列表，默认 ["qft", "qaoa", "ghz", "grover"]
        basis_gates: transpile 目标基础门集，默认 IBM ["cx","id","rz","sx","x"]

    Returns:
        {"qft_5": QuantumCircuit, "qaoa_5": QuantumCircuit, ...}
        失败的电路（mqt.bench + fallback 都失败）会被 skip 并 log warning，
        不会出现在返回的 dict 中。
    """
    if n_qubits_list is None:
        n_qubits_list = [5, 10, 20]
    if benchmark_names is None:
        benchmark_names = [name for name, _ in DEFAULT_BENCHMARKS]
    if basis_gates is None:
        basis_gates = DEFAULT_BASIS_GATES

    circuits: dict[str, QuantumCircuit] = {}

    for name in benchmark_names:
        for nq in n_qubits_list:
            key = f"{name}_{nq}"
            qc = _try_mqt(name, nq, basis_gates)
            if qc is None:
                logger.info("尝试 fallback 生成 %s", key)
                qc = _fallback_circuit(name, nq)
            if qc is None:
                logger.warning("跳过 %s（mqt + fallback 均失败）", key)
                continue
            circuits[key] = qc

    logger.info("MQT-Bench 加载完成：%d/%d 条电路成功",
                len(circuits), len(benchmark_names) * len(n_qubits_list))
    return circuits


def describe_circuits(circuits: dict[str, QuantumCircuit]) -> list[dict]:
    """汇总每条电路的 metadata（qubits / gates / 2Q 门数）— 用于报告表头。"""
    rows = []
    for key, qc in circuits.items():
        ops = dict(qc.count_ops())
        n_2q = sum(c for op, c in ops.items() if op in ("cx", "cz", "cp", "cy", "swap"))
        rows.append({
            "name": key,
            "n_qubits": qc.num_qubits,
            "size": qc.size(),
            "depth": qc.depth(),
            "n_2q": n_2q,
            "ops": ops,
        })
    return rows


def is_mqt_available() -> bool:
    """检查 mqt.bench 是否可用（用于 CLI 报告 metadata）。"""
    try:
        import mqt.bench  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_mqt_circuits(
    circuit_names: list[str] | None = None,
    num_qubits_list: list[int] | None = None,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """V14 P1 评测管线 API：拉 MQT-Bench 标准电路（list 视图）。

    与 ``get_mqt_circuits`` 同源：先尝试 mqt.bench v2.x，失败 fallback 到
    本项目 ``src/benchmarks/circuits.py``。差别在于本函数返回 ``list[QuantumCircuit]``，
    每条电路设置 ``.name = f"{benchmark}_{n_qubits}"``，便于直接喂 SABRE / AI 评测。

    Args:
        circuit_names: 电路类型 list；默认 ["qft","qaoa","grover","ghz","vqe"]
        num_qubits_list: 比特数 list；默认 [5, 10, 20]
        basis_gates: 统一 transpile 目标；默认 IBM ``["cx","id","rz","sx","x"]``

    Returns:
        ``list[QuantumCircuit]`` — 至多 ``len(circuit_names) * len(num_qubits_list)`` 条；
        加载失败的电路会被 skip（已 log warning），不会出现在返回 list 中。
    """
    if circuit_names is None:
        circuit_names = [name for name, _ in DEFAULT_BENCHMARKS]
    if num_qubits_list is None:
        num_qubits_list = [5, 10, 20]

    using_mqt = is_mqt_available()
    if using_mqt:
        logger.info(
            "fetch_mqt_circuits: 使用 mqt.bench (%d 类 × %d 规模 = %d 条目标)",
            len(circuit_names),
            len(num_qubits_list),
            len(circuit_names) * len(num_qubits_list),
        )
    else:
        logger.warning(
            "fetch_mqt_circuits: mqt.bench 不可用 — fallback 到 src/benchmarks/circuits.py 的本地生成器"
        )

    circuits_dict = get_mqt_circuits(
        n_qubits_list=num_qubits_list,
        benchmark_names=circuit_names,
        basis_gates=basis_gates,
    )
    # 按 (benchmark, n_qubits) 顺序输出，name 已由 _try_mqt / _fallback_circuit 设置好
    return list(circuits_dict.values())
