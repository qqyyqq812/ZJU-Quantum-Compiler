"""Smoke tests for local experiment scripts used by the project board."""
from __future__ import annotations

from qiskit import QuantumCircuit

from scripts.experiment_bounded_search import run_bounded_search
from scripts.experiment_sabre_trials import run_experiment
from src.benchmarks.topologies import get_topology


def test_sabre_trials_experiment_smoke():
    results = run_experiment(["ghz5"], [1, 4])

    assert len(results) == 6
    assert {row.heuristic for row in results} == {"basic", "lookahead", "decay"}
    assert {row.trials for row in results} == {1, 4}
    assert all(row.example == "ghz5" for row in results)
    assert all(row.swaps == 0 for row in results)
    assert all(row.depth == 7 for row in results)
    assert all(row.runtime_ms >= 0 for row in results)


def test_bounded_search_experiment_smoke():
    circuit = QuantumCircuit.from_qasm_file("examples/ghz5.qasm")
    result = run_bounded_search(
        "ghz5",
        circuit,
        get_topology("ibm_tokyo"),
        branch_factor=1,
        max_depth=1,
        max_steps=50,
    )

    assert result.example == "ghz5"
    assert result.method == "bounded"
    assert result.completed
    assert result.swaps >= 0
    assert result.depth is not None
    assert result.two_qubit_gates is not None
    assert result.runtime_ms >= 0
    assert result.trace_len >= 0
