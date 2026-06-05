"""FastAPI smoke tests for the public local playground API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.server.app import app


client = TestClient(app)


def test_api_status_reports_honest_ai_state():
    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "0.14.2"
    assert "not beaten SABRE" in body["status"]
    assert body["default_model"].endswith("checkpoint_ep25333.pt")


def test_api_examples_lists_public_qasm_files():
    response = client.get("/api/examples")

    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert {"qft5", "ghz5", "qaoa5", "qft10", "qaoa10", "ghz10", "vqe10"} <= ids


def test_api_benchmarks_returns_checked_in_summary():
    response = client.get("/api/benchmarks")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["sabre_completed"] == 12
    assert body["summary"]["ai_completed"] == 4
    assert body["summary"]["ai_beats_sabre"] == 0


def test_api_npqr_evidence_returns_unified_manifest():
    response = client.get("/api/npqr/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "npqr_evidence_manifest_v1"
    assert body["stage7"]["candidate"]["completed"] == 7
    assert body["stage7"]["trace_replay_rows"] == 7
    assert body["stage8"]["decision"] == "hold_stage7_offline_enhancement"
    assert body["stage8"]["promote_checkpoint"] is False
    assert body["stage10_mapping_selector"]["top1_completed"] == 2
    assert body["stage10_mapping_selector"]["selector"]["uses_probe_outcomes_for_ranking"] is False
    assert body["stage11_selector_runtime"]["passed"] is True
    assert body["stage11_selector_runtime"]["route"]["candidate"]["completed"] == 7
    assert body["stage12_selector_boundary"]["decision"] == "hold_full_boundary_use_adaptive_topk"
    assert body["stage12_selector_boundary"]["vqe10_top4_rescue"]["passed"] is True
    assert "NPQR does not generally outperform SABRE" in body["npqr_boundary"]["not_claimed"][0]


def test_api_compile_sabre_public_example():
    response = client.post(
        "/api/compile",
        json={"example": "qft5", "backend": "sabre", "topology": "tokyo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "sabre"
    assert body["input_qubits"] == 5
    assert body["swaps"] is not None


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


def test_api_compile_checked_in_examples_with_all_sabre_heuristics():
    for example in ["qft5", "qaoa5", "ghz5", "qft10", "qaoa10", "ghz10", "vqe10"]:
        for heuristic in ["basic", "lookahead", "decay"]:
            response = client.post(
                "/api/compile",
                json={
                    "example": example,
                    "backend": "sabre",
                    "topology": "tokyo",
                    "heuristic": heuristic,
                },
            )

            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "OK"
            assert body["swaps"] is not None
            assert body["depth"] is not None
            assert body["compiled_qasm"].startswith("OPENQASM 2.0;")


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


def test_api_compile_returns_routed_qasm_for_uploaded_openqasm():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0],q[2];
cx q[2],q[1];
"""
    response = client.post(
        "/api/compile",
        json={
            "qasm": qasm,
            "backend": "sabre",
            "topology": "tokyo",
            "heuristic": "lookahead",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["circuit_name"] == "inline_qasm"
    assert body["input_qubits"] == 3
    assert body["compiled_qasm"].startswith("OPENQASM 2.0;")
    assert "qreg q[" in body["compiled_qasm"]
    assert body["depth"] is not None


def test_api_compile_rejects_large_inline_qasm():
    response = client.post(
        "/api/compile",
        json={
            "qasm": " " * 8001,
            "backend": "sabre",
            "topology": "tokyo",
        },
    )

    assert response.status_code == 400
    assert "limited to 8000 characters" in response.json()["detail"]
