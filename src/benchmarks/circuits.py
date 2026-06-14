"""Small deterministic benchmark circuit generators."""
from __future__ import annotations

import random
from math import pi

from qiskit import QuantumCircuit


def generate_random(n_qubits: int, *, depth: int, seed: int) -> QuantumCircuit:
    """Generate a deterministic random-ish circuit for local NPQR gates.

    The generator is intentionally simple and stable: each layer applies a few
    single-qubit rotations plus one entangling operation. It is sufficient for
    route-boundary canaries without depending on external benchmark packages.
    """
    rng = random.Random(seed)
    circuit = QuantumCircuit(n_qubits, name=f"random_{n_qubits}_d{depth}_s{seed}")
    for layer in range(depth):
        for qubit in range(n_qubits):
            if (layer + qubit + seed) % 3 == 0:
                circuit.rx(_angle(rng), qubit)
            elif (layer + qubit + seed) % 3 == 1:
                circuit.ry(_angle(rng), qubit)
            else:
                circuit.rz(_angle(rng), qubit)
        for first, second in _layer_pairs(n_qubits, layer=layer, rng=rng):
            if (layer + first + second) % 3 == 0:
                circuit.cx(first, second)
            elif (layer + first + second) % 3 == 1:
                circuit.cz(first, second)
            else:
                circuit.cx(second, first)
    return circuit


def _angle(rng: random.Random) -> float:
    return (rng.random() * 2.0 - 1.0) * pi


def _layer_pairs(
    n_qubits: int,
    *,
    layer: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    target_pairs = min(max(1, n_qubits // 3), n_qubits // 2)
    qubits = list(range(n_qubits))
    rng.shuffle(qubits)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for first in qubits:
        if first in used:
            continue
        candidates = [
            q
            for q in qubits
            if q != first and q not in used and tuple(sorted((first, q))) not in pairs
        ]
        if not candidates:
            candidates = [q for q in qubits if q != first and q not in used]
        if not candidates:
            continue
        second = candidates[(layer + first + rng.randrange(len(candidates))) % len(candidates)]
        pairs.append(tuple(sorted((first, second))))
        used.add(first)
        used.add(second)
        if len(pairs) >= target_pairs:
            break
    return pairs
