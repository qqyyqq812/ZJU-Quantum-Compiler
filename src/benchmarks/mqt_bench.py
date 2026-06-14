"""MQT Bench adapter with local fallback circuits.

The project treats MQT Bench as optional. Stage boundary scripts only need a
small helper surface: try to load a benchmark from MQT when installed, otherwise
build a deterministic local circuit with the same qubit count.
"""
from __future__ import annotations

from math import pi
from typing import Iterable

from qiskit import QuantumCircuit, transpile

DEFAULT_BASIS_GATES = ("rx", "ry", "rz", "sx", "x", "h", "cx")


def is_mqt_available() -> bool:
    try:
        import mqt.bench  # noqa: F401
    except Exception:
        return False
    return True


def _try_mqt(
    benchmark: str,
    n_qubits: int,
    basis_gates: Iterable[str] = DEFAULT_BASIS_GATES,
) -> QuantumCircuit | None:
    try:
        from mqt.bench import BenchmarkLevel, get_benchmark

        circuit = get_benchmark(
            benchmark,
            BenchmarkLevel.INDEP,
            circuit_size=n_qubits,
            opt_level=1,
        )
        if circuit.num_clbits:
            circuit = circuit.remove_final_measurements(inplace=False)
        return transpile(circuit, basis_gates=list(basis_gates), optimization_level=1)
    except Exception:
        return None


def _fallback_circuit(benchmark: str, n_qubits: int) -> QuantumCircuit | None:
    key = benchmark.lower()
    if key == "ghz":
        return _ghz(n_qubits)
    if key == "qft":
        return _qft(n_qubits)
    if key == "qaoa":
        return _qaoa(n_qubits)
    if key == "vqe":
        return _vqe(n_qubits)
    return None


def _ghz(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}")
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def _qft(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name=f"qft_{n_qubits}")
    for target in range(n_qubits):
        circuit.h(target)
        for control in range(target + 1, n_qubits):
            circuit.cx(control, target)
            circuit.rz(pi / (2 ** (control - target)), target)
            circuit.cx(control, target)
    return circuit


def _qaoa(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name=f"qaoa_{n_qubits}")
    for qubit in range(n_qubits):
        circuit.h(qubit)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
        circuit.rz(0.4, qubit + 1)
        circuit.cx(qubit, qubit + 1)
    for qubit in range(n_qubits):
        circuit.rx(0.2, qubit)
    return circuit


def _vqe(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name=f"vqe_{n_qubits}")
    for qubit in range(n_qubits):
        circuit.ry(0.1 * (qubit + 1), qubit)
        circuit.rz(0.05 * (qubit + 1), qubit)
    for qubit in range(0, n_qubits - 1, 2):
        circuit.cx(qubit, qubit + 1)
    for qubit in range(1, n_qubits - 1, 2):
        circuit.cx(qubit, qubit + 1)
    return circuit
