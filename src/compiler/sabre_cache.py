"""SABRE baseline 缓存 (V14 §V14-1)

一个电路只跑一次 SABRE，用内容 hash 作为缓存键。
20 个 worker 共享同一个缓存 → 吞吐 1.0 → 15+ eps/s。

设计决策:
- 用进程内 dict 作缓存（训练期间够用）
- 键 = topology name + circuit qubits + (qubit_a, qubit_b, gate_name) 序列 hash
- 不用 pickle/磁盘缓存（worker 之间共享开销大于收益）
"""
from __future__ import annotations

import hashlib
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


_CACHE: dict[str, int] = {}
_HITS = 0
_MISSES = 0


def _circuit_fingerprint(circuit: QuantumCircuit, topology_name: str) -> str:
    """生成电路指纹。门序列完全相同 → 指纹相同。

    忽略门的经典控制、测量等无关路由的部分。
    """
    parts: list[str] = [topology_name, str(circuit.num_qubits)]
    for instr in circuit.data:
        gate_name = instr.operation.name
        # 只取 qubit 索引，不管 param（CX/SWAP 不带 param）
        qubits = tuple(circuit.find_bit(q).index for q in instr.qubits)
        parts.append(f"{gate_name}:{qubits}")
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode()).hexdigest()


def get_sabre_swaps(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    topology_name: str = "unknown",
) -> int:
    """查询 SABRE 在此电路上的 SWAP 数。命中缓存直接返回，否则跑 Qiskit transpile。"""
    global _HITS, _MISSES
    fp = _circuit_fingerprint(circuit, topology_name)
    if fp in _CACHE:
        _HITS += 1
        return _CACHE[fp]

    _MISSES += 1
    try:
        from qiskit import transpile
        transpiled = transpile(
            circuit,
            coupling_map=coupling_map,
            optimization_level=1,
            routing_method="sabre",
            seed_transpiler=42,
        )
        swaps = transpiled.count_ops().get("swap", 0)
    except Exception:
        swaps = 0

    _CACHE[fp] = swaps
    return swaps


def cache_stats() -> dict[str, int]:
    """返回缓存统计。训练 log 周期性打印用。"""
    total = _HITS + _MISSES
    hit_rate = _HITS / total if total > 0 else 0.0
    return {
        "size": len(_CACHE),
        "hits": _HITS,
        "misses": _MISSES,
        "hit_rate": hit_rate,
    }


def reset_cache() -> None:
    """清空缓存（测试用）。"""
    global _HITS, _MISSES
    _CACHE.clear()
    _HITS = 0
    _MISSES = 0
