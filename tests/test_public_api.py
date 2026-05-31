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
