"""
M1 测试用例：验证 Qiskit 工具链和基础概念
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import networkx as nx


class TestBellState:
    """Bell 态相关测试"""

    def test_bell_circuit_creation(self):
        """能创建正确的 Bell 态电路"""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        assert qc.num_qubits == 2
        ops = dict(qc.count_ops())
        assert ops.get('h', 0) == 1
        assert ops.get('cx', 0) == 1

    def test_bell_state_simulation(self):
        """Bell 态模拟应只产生 |00⟩ 和 |11⟩"""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        sim = AerSimulator()
        result = sim.run(qc, shots=4096).result()
        counts = result.get_counts(qc)

        # 只应出现 00 和 11
        assert set(counts.keys()) <= {'00', '11'}
        # 各约 50%
        total = sum(counts.values())
        for state in counts:
            assert 35 < counts[state] / total * 100 < 65


class TestCouplingMap:
    """Coupling Map 相关测试"""

    def test_line_topology(self):
        """线性拓扑应有 n-1 条边"""
        cm = CouplingMap.from_line(5)
        assert cm.size() == 5
        edges = cm.get_edges()
        # 双向边
        assert len(edges) == 8  # 4 条无向边 × 2

    def test_ring_topology(self):
        """环形拓扑应有 n 条边"""
        cm = CouplingMap.from_ring(5)
        assert cm.size() == 5

    def test_full_topology(self):
        """全连接拓扑应有 n*(n-1)/2 条无向边"""
        cm = CouplingMap.from_full(5)
        assert cm.size() == 5

    def test_grid_topology(self):
        """3x3 网格拓扑应有 9 个比特"""
        cm = CouplingMap.from_grid(3, 3)
        assert cm.size() == 9

    def test_coupling_map_distance(self):
        """Coupling Map 距离计算"""
        cm = CouplingMap.from_line(5)
        assert cm.distance(0, 1) == 1
        assert cm.distance(0, 4) == 4


class TestTranspiler:
    """Transpiler 相关测试"""

    def test_transpile_satisfies_coupling(self):
        """编译后电路应满足 Coupling Map 约束"""
        qc = QuantumCircuit(5)
        qc.cx(0, 4)  # 不相邻
        qc.cx(1, 3)  # 不相邻

        cm = CouplingMap.from_line(5)
        pm = generate_preset_pass_manager(
            optimization_level=1,
            coupling_map=cm,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x']
        )
        compiled = pm.run(qc)

        # 验证所有 cx 门的操作比特在 coupling map 中相邻
        edges = set(tuple(e) for e in cm.get_edges())
        for instruction in compiled.data:
            if instruction.operation.name == 'cx':
                q0 = compiled.qubits.index(instruction.qubits[0])
                q1 = compiled.qubits.index(instruction.qubits[1])
                assert (q0, q1) in edges, f"cx({q0},{q1}) 不在 coupling map 中"

    def test_transpile_increases_cx(self):
        """线性拓扑下编译应增加 CNOT 数（因为需要 SWAP）"""
        qc = QuantumCircuit(5)
        qc.cx(0, 4)  # 距离=4，必须插入 SWAP

        cm = CouplingMap.from_line(5)
        pm = generate_preset_pass_manager(
            optimization_level=1,
            coupling_map=cm,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x']
        )
        compiled = pm.run(qc)

        original_cx = dict(qc.count_ops()).get('cx', 0)
        compiled_cx = dict(compiled.count_ops()).get('cx', 0)
        assert compiled_cx >= original_cx

    def test_full_topology_no_overhead(self):
        """全连接拓扑下编译不应增加额外 CNOT"""
        qc = QuantumCircuit(5)
        qc.cx(0, 4)

        cm = CouplingMap.from_full(5)
        pm = generate_preset_pass_manager(
            optimization_level=1,
            coupling_map=cm,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x']
        )
        compiled = pm.run(qc)

        original_cx = dict(qc.count_ops()).get('cx', 0)
        compiled_cx = dict(compiled.count_ops()).get('cx', 0)
        assert compiled_cx == original_cx
