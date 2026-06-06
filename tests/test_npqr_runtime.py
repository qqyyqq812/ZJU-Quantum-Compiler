from __future__ import annotations

from qiskit import QuantumCircuit, qasm2

from src.benchmarks.topologies import get_topology
from src.compiler.npqr_runtime import NPQRRuntime, NPQRRuntimeConfig
from src.compiler.npqr_trace import verify_routed_circuit_topology


def test_npqr_runtime_compiles_with_neural_selector_and_no_sabre_fallback():
    coupling_map = get_topology("ibm_tokyo")
    circuit = QuantumCircuit.from_qasm_file("examples/ghz5.qasm")
    circuit.name = "ghz5"
    runtime = NPQRRuntime(coupling_map, config=NPQRRuntimeConfig(max_steps=45))

    result = runtime.compile(circuit)

    assert result.status == "OK"
    assert result.completed is True
    assert result.algorithm == "npqr_neural_selector_suffix_v1"
    assert result.compiled_circuit is not None
    assert qasm2.dumps(result.compiled_circuit).startswith("OPENQASM 2.0;")
    assert verify_routed_circuit_topology(result.compiled_circuit, coupling_map).passed is True
    assert result.components["neural_beam"] is True
    assert result.components["mapping_selector"] is True
    assert result.components["sabre_fallback"] is False
