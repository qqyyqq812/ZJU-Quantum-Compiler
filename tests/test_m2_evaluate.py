"""M2 测试：评估框架"""

import pytest
import tempfile
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from src.benchmarks.evaluate import (
    evaluate_compiler, compare_compilers, save_results, load_results, CompileResult
)


class TestEvaluate:
    def test_evaluate_sabre(self):
        qc = QuantumCircuit(5)
        qc.cx(0, 4)
        cm = CouplingMap.from_line(5)

        result = evaluate_compiler(
            qc, cm,
            circuit_name="test", topology_name="linear_5",
        )
        assert isinstance(result, CompileResult)
        assert result.compiled_cx >= result.original_cx
        assert result.compile_time_ms > 0

    def test_full_topology_no_overhead(self):
        qc = QuantumCircuit(5)
        qc.cx(0, 4)
        cm = CouplingMap.from_full(5)

        result = evaluate_compiler(
            qc, cm,
            circuit_name="test", topology_name="full_5",
        )
        assert result.compiled_cx == result.original_cx

    def test_compare_compilers(self):
        results = [
            CompileResult("qft_5", "linear_5", "sabre_O1", 10, 15, 4, 10, 2, 5.0, 5),
            CompileResult("qft_5", "full_5", "sabre_O1", 10, 10, 4, 4, 0, 3.0, 5),
        ]
        table = compare_compilers(results)
        assert "qft_5" in table
        assert "linear_5" in table

    def test_save_and_load(self):
        results = [
            CompileResult("test", "topo", "compiler", 5, 10, 2, 8, 2, 1.0, 5),
        ]
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        save_results(results, path)
        loaded = load_results(path)
        assert len(loaded) == 1
        assert loaded[0].circuit_name == "test"
        assert loaded[0].compiled_cx == 8

    def test_overhead_properties(self):
        r = CompileResult("t", "t", "c", 10, 15, 4, 10, 2, 1.0, 5)
        assert r.depth_overhead == pytest.approx(0.5)
        assert r.cx_overhead == pytest.approx(1.5)
