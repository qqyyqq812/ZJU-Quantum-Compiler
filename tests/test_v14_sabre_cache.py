"""V14 §V14-1 SABRE 缓存 smoke test."""
from __future__ import annotations

import pytest

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.sabre_cache import (
    get_sabre_swaps,
    cache_stats,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_cache()
    yield
    reset_cache()


def _simple_circuit(n: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    # 串起来几个 CX 让 SABRE 有活干
    for i in range(n - 1):
        qc.cx(i, (i + 2) % n)
    return qc


def test_cache_hits_on_second_call():
    cm = CouplingMap.from_line(5)
    qc = _simple_circuit(5)

    s1 = get_sabre_swaps(qc, cm, "linear_5")
    stats1 = cache_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    s2 = get_sabre_swaps(qc, cm, "linear_5")
    stats2 = cache_stats()
    assert stats2["hits"] == 1
    assert stats2["misses"] == 1
    assert s1 == s2


def test_different_circuits_are_different_keys():
    cm = CouplingMap.from_line(5)
    qc_a = _simple_circuit(5)
    qc_b = QuantumCircuit(5)
    qc_b.cx(0, 4)  # 完全不同的电路

    get_sabre_swaps(qc_a, cm, "linear_5")
    get_sabre_swaps(qc_b, cm, "linear_5")

    stats = cache_stats()
    assert stats["size"] == 2
    assert stats["misses"] == 2


def test_different_topologies_are_different_keys():
    cm1 = CouplingMap.from_line(5)
    cm2 = CouplingMap.from_ring(5)
    qc = _simple_circuit(5)

    get_sabre_swaps(qc, cm1, "linear_5")
    get_sabre_swaps(qc, cm2, "ring_5")

    assert cache_stats()["size"] == 2
