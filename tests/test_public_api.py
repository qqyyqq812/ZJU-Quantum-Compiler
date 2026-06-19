"""FastAPI smoke tests for the public local playground API."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient
from qiskit import qasm2

from src.benchmarks.topologies import get_topology
from src.server.app import app


client = TestClient(app)

SMALL_PUBLIC_EXAMPLES = ["qft5", "qaoa5", "ghz5", "qft10", "qaoa10", "ghz10", "vqe10"]
PUBLIC_EXAMPLES = [
    "qft5",
    "qaoa5",
    "ghz5",
    "qft10",
    "qaoa10",
    "ghz10",
    "vqe10",
    "line_ghz30",
    "random30_d4",
    "line_ghz50",
    "ring_sparse50",
]


def _assert_routed_qasm_matches_tokyo(compiled_qasm: str) -> None:
    circuit = qasm2.loads(
        compiled_qasm,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    coupling = get_topology("ibm_tokyo")
    edges = {tuple(edge) for edge in coupling.get_edges()}
    undirected = edges | {(b, a) for a, b in edges}
    invalid = []
    for index, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(qubits) != 2:
            continue
        pair = (int(qubits[0]), int(qubits[1]))
        if pair not in undirected:
            invalid.append((index, instruction.operation.name, pair))
    assert invalid == []


def test_api_status_reports_npqr_default_model():
    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "0.14.2"
    assert body["default_backend"] == "npqr"
    assert body["default_model"] == "models/default/npqr-default.pt"
    assert body["model_exists"] is True
    assert body["model_loadable"] is True
    assert "npqr_" + "model" not in body
    assert "npqr_" + "loadable" not in body
    assert "ai_loadable" not in body
    assert "ai_status" not in body
    assert "SABRE is returned as a comparison baseline" in body["status"]


def test_api_examples_lists_public_qasm_files():
    response = client.get("/api/examples")

    assert response.status_code == 200
    rows = response.json()
    ids = {row["id"] for row in rows}
    assert set(PUBLIC_EXAMPLES) <= ids
    by_id = {row["id"]: row for row in rows}
    assert by_id["line_ghz30"]["topology"] == "grid_5x6"
    assert by_id["line_ghz50"]["topology"] == "grid_5x10"


def test_api_topology_returns_json_safe_grid_edges():
    for name, qubits in [("tokyo", 20), ("grid_5x6", 30), ("grid_5x10", 50)]:
        response = client.get(f"/api/topology/{name}")
        assert response.status_code == 200
        body = response.json()
        assert body["info"]["n_qubits"] == qubits
        assert body["edges"]
        assert isinstance(body["edges"][0], list)
        assert all(isinstance(value, int) for value in body["edges"][0])


def test_api_validate_accepts_valid_openqasm2():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""

    response = client.post("/api/validate", json={"qasm": qasm, "topology": "tokyo"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["input_qubits"] == 2
    assert body["gate_count"] == 2
    assert body["cx_count"] == 1
    assert "cx" in body["supported_gates"]


def test_api_validate_accepts_selected_30_and_50_qubit_grid_topologies():
    qasm30 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
h q[0];
cx q[0],q[29];
"""
    qasm50 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
h q[0];
cx q[0],q[49];
"""

    ok30 = client.post("/api/validate", json={"qasm": qasm30, "topology": "grid_5x6"})
    wrong30 = client.post("/api/validate", json={"qasm": qasm30, "topology": "tokyo"})
    ok50 = client.post("/api/validate", json={"qasm": qasm50, "topology": "grid_5x10"})

    assert ok30.status_code == 200
    assert ok30.json()["status"] == "OK"
    assert ok30.json()["input_qubits"] == 30

    assert wrong30.status_code == 200
    assert wrong30.json()["status"] == "Invalid"
    assert wrong30.json()["input_qubits"] == 30

    assert ok50.status_code == 200
    assert ok50.json()["status"] == "OK"
    assert ok50.json()["input_qubits"] == 50


def test_api_compile_checked_in_large_examples_with_sabre_grid_topologies():
    cases = [
        ("line_ghz30", "grid_5x6", 30),
        ("random30_d4", "grid_5x6", 30),
        ("line_ghz50", "grid_5x10", 50),
        ("ring_sparse50", "grid_5x10", 50),
    ]
    for example, topology, qubits in cases:
        response = client.post(
            "/api/compile",
            json={"example": example, "backend": "sabre", "topology": topology},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "OK"
        assert body["backend"] == "sabre"
        assert body["topology"] == topology
        assert body["input_qubits"] == qubits
        assert body["route_trace"]


def test_api_validate_rejects_invalid_openqasm2():
    response = client.post("/api/validate", json={"qasm": "qreg q[2];\ncx q[0],q[1];"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "Invalid"
    assert "OpenQASM" in body["message"]


def test_api_benchmarks_returns_public_algorithm_boundary():
    response = client.get("/api/benchmarks")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["default_backend"] == "npqr"
    assert body["summary"]["comparison_baseline"] == "qiskit_sabre_swap_basic"
    assert body["summary"]["sabre_fallback"] is False
    assert body["summary"]["representative_10_20_basic"]["cases"] == 10
    assert body["summary"]["representative_10_20_basic"]["npqr_completed"] == 10
    assert body["summary"]["representative_10_20_basic"]["npqr_beats_sabre_basic"] == 10
    assert body["summary"]["representative_10_20_basic"]["sabre_fallback_used"] is False
    assert body["summary"]["scale_smoke_30_50_basic"]["cases"] == 4
    assert body["summary"]["scale_smoke_30_50_basic"]["npqr_completed"] == 4
    assert body["summary"]["scale_smoke_30_50_basic"]["npqr_beats_sabre_basic"] == 4
    assert "extension_scope" in body["summary"]
    assert "30/50-qubit examples" in body["summary"]["extension_scope"]
    assert body["algorithm_components"]
    assert "The reported comparison uses SABRE basic as the fixed baseline." in body["claims"]["scope"]


def test_api_npqr_evidence_returns_public_manifest():
    response = client.get("/api/npqr/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "npqr_public_algorithm_evidence_v1"
    assert body["default_route"]["backend"] == "npqr"
    assert body["default_route"]["model"] == "models/default/npqr-default.pt"
    assert body["default_route"]["sabre_fallback"] is False
    assert body["baseline"]["name"] == "SABRE"
    assert body["baseline"]["heuristic"] == "basic"
    assert body["baseline"]["not_our_algorithm"] is True
    assert body["representative_10_20_basic"]["summary"]["npqr_beats_sabre_basic"] == 10
    assert body["scale_smoke_30_50_basic"]["summary"]["npqr_beats_sabre_basic"] == 4
    assert body["scale_smoke_30_50_basic"]["extension_scope"]
    assert "graph modeling" in body["concept_mapping"]


def test_api_compile_defaults_to_npqr_with_sabre_baseline():
    response = client.post(
        "/api/compile",
        json={"example": "ghz5", "topology": "tokyo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "npqr"
    assert body["algorithm"] == "npqr_neural_selector_suffix_v1"
    assert body["circuit_name"] == "ghz5"
    assert body["compiled_qasm"].startswith("OPENQASM 2.0;")
    assert body["components"]["neural_beam"] is True
    assert body["components"]["mapping_selector"] is True
    assert body["components"]["sabre_fallback"] is False
    assert body["baseline"]["backend"] == "sabre"
    assert body["baseline"]["heuristic"] == "basic"
    assert body["baseline"]["status"] == "OK"
    assert body["model_path"] == "models/default/npqr-default.pt"
    first_event = body["route_trace"][0]
    assert "reason" in first_event
    assert "source_line" in first_event


def _wait_for_compile_job(job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/compile/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"compile job {job_id} did not finish")


def test_api_compile_job_returns_real_phase_timings_and_result():
    response = client.post(
        "/api/compile/jobs",
        json={"example": "ghz5", "backend": "sabre", "topology": "tokyo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"]
    assert body["status"] in {"queued", "running", "completed"}
    assert body["phase"] in {"parsing", "mapping", "routing", "output", "done"}

    final = _wait_for_compile_job(body["job_id"])
    assert final["status"] == "completed"
    assert final["phase"] == "done"
    assert final["result"]["backend"] == "sabre"
    assert final["result"]["status"] == "OK"
    phases = {phase["phase"]: phase for phase in final["phases"]}
    for name in ["parsing", "mapping", "routing", "output"]:
        assert name in phases
        assert phases[name]["status"] == "done"
        assert phases[name]["elapsed_ms"] >= 0


def test_api_compile_job_reports_failed_qasm_without_fake_result():
    response = client.post(
        "/api/compile/jobs",
        json={"qasm": "not qasm", "backend": "sabre", "topology": "tokyo"},
    )

    assert response.status_code == 200
    final = _wait_for_compile_job(response.json()["job_id"])
    assert final["status"] == "failed"
    assert final["phase"] == "error"
    assert final["result"] is None
    assert "Invalid OpenQASM 2 input" in final["error"]


def test_api_compile_npqr_accepts_frontier_pruning_staging_policy():
    response = client.post(
        "/api/compile",
        json={
            "example": "ghz5",
            "backend": "npqr",
            "topology": "tokyo",
            "npqr_frontier_pruning_policy": "frontier_touch_8",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "npqr"
    assert body["components"]["frontier_action_pruning"] is True
    assert body["components"]["frontier_action_pruning_policy"] == "frontier_touch_8"
    assert body["components"]["sabre_fallback"] is False
    assert body["baseline"]["backend"] == "sabre"


def test_api_compile_npqr_rejects_unknown_frontier_pruning_policy():
    response = client.post(
        "/api/compile",
        json={
            "example": "ghz5",
            "backend": "npqr",
            "topology": "tokyo",
            "npqr_frontier_pruning_policy": "extended_touch_8",
        },
    )

    assert response.status_code == 422


def test_api_compile_npqr_accepts_refined_frontier_trigger_profile():
    response = client.post(
        "/api/compile",
        json={
            "example": "ghz5",
            "backend": "npqr",
            "topology": "tokyo",
            "npqr_frontier_trigger_profile": "u485_d30_r060_c120",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "npqr"
    assert body["components"]["frontier_action_pruning"] is True
    assert body["components"]["frontier_action_pruning_policy"] == "frontier_touch_8"
    assert body["components"]["sabre_fallback"] is False


def test_api_compile_npqr_accepts_compact_response_flags():
    response = client.post(
        "/api/compile",
        json={
            "example": "ghz5",
            "backend": "npqr",
            "topology": "tokyo",
            "include_route_trace": False,
            "include_compiled_qasm": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "npqr"
    assert body["compiled_qasm"] is None
    assert body["route_trace"] is None
    assert body["trace_len"] is not None
    assert body["executed_gates"] >= body["input_cx"]
    assert body["components"]["sabre_fallback"] is False


def test_api_compile_npqr_rejects_unknown_frontier_trigger_profile():
    response = client.post(
        "/api/compile",
        json={
            "example": "ghz5",
            "backend": "npqr",
            "topology": "tokyo",
            "npqr_frontier_trigger_profile": "u470_d30_r060_c120",
        },
    )

    assert response.status_code == 422


def test_api_compile_sabre_public_example():
    response = client.post(
        "/api/compile",
        json={"example": "qft5", "backend": "sabre", "topology": "tokyo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "sabre"
    assert body["heuristic"] == "basic"
    assert body["input_qubits"] == 5
    assert body["swaps"] is not None
    assert "basic heuristic" in body["message"]
    assert body["route_trace"]
    assert body["trace_len"] == len(body["route_trace"])
    assert body["initial_mapping"] == {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}


def test_api_compile_sabre_accepts_heuristic_parameter():
    response = client.post(
        "/api/compile",
        json={
            "example": "qft5",
            "backend": "sabre",
            "topology": "tokyo",
            "heuristic": "decay",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "sabre"
    assert body["heuristic"] == "decay"
    assert "decay heuristic" in body["message"]


def test_api_compile_rejects_unknown_sabre_heuristic():
    response = client.post(
        "/api/compile",
        json={
            "example": "qft5",
            "backend": "sabre",
            "topology": "tokyo",
            "heuristic": "unknown",
        },
    )

    assert response.status_code == 422


def test_api_compile_rejects_removed_legacy_ai_backend():
    response = client.post(
        "/api/compile",
        json={"example": "qft5", "backend": "ai", "topology": "tokyo"},
    )

    assert response.status_code == 422


def test_api_compile_exposes_10q_heuristic_examples():
    for example in ["qft10", "qaoa10", "ghz10", "vqe10"]:
        response = client.post(
            "/api/compile",
            json={
                "example": example,
                "backend": "sabre",
                "topology": "tokyo",
                "heuristic": "lookahead",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "OK"
        assert body["heuristic"] == "lookahead"
        assert body["input_qubits"] == 10


def test_api_compile_checked_in_examples_with_npqr_route_trace():
    for example in SMALL_PUBLIC_EXAMPLES:
        response = client.post(
            "/api/compile",
            json={"example": example, "backend": "npqr", "topology": "tokyo"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "OK"
        assert body["backend"] == "npqr"
        assert body["compiled_qasm"].startswith("OPENQASM 2.0;")
        assert body["route_trace"]
        assert body["trace_len"] is not None
        assert body["executed_gates"] >= body["input_cx"]
        assert body["initial_mapping"]
        assert body["final_mapping"]
        first_event = body["route_trace"][0]
        assert first_event["op"] is not None
        assert first_event["logical_qubits"] is not None
        assert first_event["physical_qubits"]
        assert first_event["mapping_before"]
        assert first_event["mapping_after"]
        _assert_routed_qasm_matches_tokyo(body["compiled_qasm"])


def test_api_compile_accepts_small_inline_qasm():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
    response = client.post(
        "/api/compile",
        json={
            "qasm": qasm,
            "backend": "sabre",
            "topology": "tokyo",
            "heuristic": "basic",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["heuristic"] == "basic"
    assert body["input_qubits"] == 2
    assert body["compiled_qasm"].startswith("OPENQASM 2.0;")
    assert 'include "qelib1.inc";' in body["compiled_qasm"]


def test_api_compile_accepts_qelib1_sx_rz_inline_qasm():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
sx q[0];
rz(0.5) q[0];
"""
    response = client.post(
        "/api/compile",
        json={
            "qasm": qasm,
            "backend": "sabre",
            "topology": "tokyo",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["input_qubits"] == 1


def test_api_compile_rejects_invalid_inline_qasm():
    response = client.post(
        "/api/compile",
        json={"qasm": "not qasm", "backend": "sabre", "topology": "tokyo"},
    )

    assert response.status_code == 400
    assert "Invalid OpenQASM 2 input" in response.json()["detail"]


def test_api_compile_rejects_oversize_inline_qasm():
    response = client.post(
        "/api/compile",
        json={"qasm": "OPENQASM 2.0;\n" + ("// x\n" * 3000), "backend": "sabre"},
    )

    assert response.status_code == 400
    assert "limited to 8000 characters" in response.json()["detail"]
