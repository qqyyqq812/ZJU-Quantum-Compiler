"""M2 测试：拓扑集"""

import pytest
from qiskit.transpiler import CouplingMap
from src.benchmarks.topologies import get_topology, get_topology_info, get_standard_topologies


class TestTopologies:
    def test_linear(self):
        cm = get_topology('linear_5')
        assert cm.size() == 5

    def test_ring(self):
        cm = get_topology('ring_5')
        assert cm.size() == 5

    def test_grid(self):
        cm = get_topology('grid_3x3')
        assert cm.size() == 9

    def test_heavy_hex(self):
        cm = get_topology('heavy_hex_3')
        assert cm.size() > 10  # d=3 → 15+ qubits

    def test_full(self):
        cm = get_topology('full_5')
        assert cm.size() == 5

    def test_ibm_eagle(self):
        cm = get_topology('ibm_eagle')
        assert cm.size() >= 100  # Eagle ~127q

    def test_google_sycamore(self):
        cm = get_topology('google_sycamore')
        assert cm.size() >= 50

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError):
            get_topology('nonexistent_topology')

    def test_topology_info(self):
        cm = get_topology('grid_3x3')
        info = get_topology_info(cm)
        assert info['n_qubits'] == 9
        assert info['diameter'] > 0
        assert info['avg_degree'] > 0

    def test_standard_topologies(self):
        topos = get_standard_topologies()
        assert len(topos) >= 5
        for name, cm in topos.items():
            assert isinstance(cm, CouplingMap)
            assert cm.size() > 0

    def test_distance_consistency(self):
        cm = get_topology('linear_5')
        assert cm.distance(0, 1) == 1
        assert cm.distance(0, 4) == 4
