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
    assert {"qft5", "ghz5", "qaoa5"} <= ids


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
