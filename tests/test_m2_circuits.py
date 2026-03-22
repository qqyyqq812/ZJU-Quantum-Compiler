"""M2 测试：基准电路集"""

import pytest
from qiskit import QuantumCircuit
from src.benchmarks.circuits import (
    generate_qft, generate_grover, generate_qaoa, generate_random,
    get_benchmark_suite,
)


class TestCircuitGeneration:
    @pytest.mark.parametrize("n", [5, 10, 15])
    def test_qft_generation(self, n):
        qc = generate_qft(n)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == n
        assert qc.size() > 0

    @pytest.mark.parametrize("n", [5, 10])
    def test_grover_generation(self, n):
        qc = generate_grover(n)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == n
        assert qc.size() > 0

    @pytest.mark.parametrize("n", [5, 10])
    def test_qaoa_generation(self, n):
        qc = generate_qaoa(n, p=2)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == n
        ops = dict(qc.count_ops())
        assert ops.get('cx', 0) > 0  # QAOA 必须有 CNOT

    @pytest.mark.parametrize("n", [5, 10])
    def test_random_generation(self, n):
        qc = generate_random(n, depth=n, seed=42)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == n

    def test_random_reproducible(self):
        qc1 = generate_random(5, depth=5, seed=42)
        qc2 = generate_random(5, depth=5, seed=42)
        assert qc1 == qc2

    def test_benchmark_suite(self):
        suite = get_benchmark_suite([5])
        assert len(suite) == 4  # qft, grover, qaoa, random
        for item in suite:
            assert 'name' in item
            assert 'circuit' in item
            assert 'n_qubits' in item
            assert item['n_qubits'] == 5
