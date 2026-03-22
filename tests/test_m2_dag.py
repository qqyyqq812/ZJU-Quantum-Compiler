"""M2 测试：DAG 操作模块"""

import pytest
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from src.compiler.dag import CircuitDAG, GateNode


class TestCircuitDAG:
    def _make_simple_circuit(self) -> QuantumCircuit:
        """H(0) → CX(0,1) → CX(1,2)"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        return qc

    def test_dag_creation(self):
        qc = self._make_simple_circuit()
        dag = CircuitDAG(qc)
        assert dag.n_qubits == 3
        assert dag.n_gates == 3  # h, cx, cx

    def test_front_layer(self):
        qc = self._make_simple_circuit()
        dag = CircuitDAG(qc)
        front = dag.get_front_layer()
        # H(0) 没有前驱，应在前沿
        assert len(front) >= 1
        assert any(g.gate_type == 'h' for g in front)

    def test_execute_gate(self):
        qc = self._make_simple_circuit()
        dag = CircuitDAG(qc)
        front = dag.get_front_layer()
        initial_remaining = dag.remaining_gates()

        dag.execute_gate(front[0].gate_id)
        assert dag.remaining_gates() == initial_remaining - 1

    def test_execute_all_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dag = CircuitDAG(qc)

        while not dag.is_done():
            front = dag.get_front_layer()
            assert len(front) > 0
            dag.execute_gate(front[0].gate_id)

        assert dag.is_done()
        assert dag.remaining_gates() == 0

    def test_execute_non_front_raises(self):
        qc = self._make_simple_circuit()
        dag = CircuitDAG(qc)
        # 最后一个门 (CX(1,2)) 不在前沿
        with pytest.raises(ValueError):
            dag.execute_gate(2)

    def test_two_qubit_front(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        dag = CircuitDAG(qc)
        # 初始前沿: H(0), H(1) — 都是单比特
        two_q = dag.get_two_qubit_front()
        assert len(two_q) == 0
        # 执行 H 后前沿应包含 CX
        front = dag.get_front_layer()
        for g in front:
            dag.execute_gate(g.gate_id)
        two_q = dag.get_two_qubit_front()
        assert len(two_q) == 1

    def test_apply_swap(self):
        mapping = {0: 0, 1: 1, 2: 2}
        new_map = CircuitDAG.apply_swap(0, 1, mapping)
        assert new_map[0] == 1
        assert new_map[1] == 0
        assert new_map[2] == 2

    def test_execute_executable(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)  # 相邻
        qc.cx(0, 2)  # 在线性拓扑上不相邻
        dag = CircuitDAG(qc)
        cm = CouplingMap.from_line(3)
        mapping = {0: 0, 1: 1, 2: 2}
        executed = dag.execute_executable(mapping, cm)
        assert executed == 1  # 只有 CX(0,1) 相邻可执行

    def test_to_networkx(self):
        qc = self._make_simple_circuit()
        dag = CircuitDAG(qc)
        G = dag.to_networkx()
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes) == 3

    def test_node_features(self):
        qc = self._make_simple_circuit()
        dag = CircuitDAG(qc)
        features = dag.get_node_features()
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3  # 3 gates
        assert features.shape[1] > 0

    def test_interaction_graph(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        dag = CircuitDAG(qc)
        ig = dag.get_interaction_graph()
        assert ig.has_edge(0, 1)
        assert ig[0][1]['weight'] == 2
        assert ig.has_edge(1, 2)
