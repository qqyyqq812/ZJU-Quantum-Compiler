"""FastAPI smoke tests for the public local playground API."""
from __future__ import annotations

from fastapi.testclient import TestClient
from qiskit import qasm2

from src.benchmarks.topologies import get_topology
from src.server.app import app


client = TestClient(app)

PUBLIC_EXAMPLES = ["qft5", "qaoa5", "ghz5", "qft10", "qaoa10", "ghz10", "vqe10"]


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


def test_api_status_reports_honest_ai_state():
    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "0.14.2"
    assert "not beaten SABRE" in body["status"]
    assert body["default_backend"] == "npqr"
    assert body["npqr_model"].endswith("wave2_stage2_e120_lr5e5_s51_h02.pt")
    assert body["npqr_loadable"] is True
    assert body["default_model"].endswith("checkpoint_ep25333.pt")


def test_api_examples_lists_public_qasm_files():
    response = client.get("/api/examples")

    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert set(PUBLIC_EXAMPLES) <= ids


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
    assert body["stage13_adaptive_selector_boundary"]["passed"] is True
    assert body["stage13_adaptive_selector_boundary"]["route"]["candidate"]["completed"] == 6
    assert body["stage13_adaptive_selector_boundary"]["selection"]["top4_rescue_completed_circuits"] == [
        "vqe_10"
    ]
    assert body["stage14_adaptive_dataset"]["accepted_traces"] == 16
    assert body["stage14_adaptive_dataset"]["samples"] == 95
    assert body["stage16_hardcase_scout"]["completed_circuits"] == ["qaoa_10"]
    assert body["stage16_hardcase_scout"]["missing_completed_circuits"] == ["ghz_10"]
    assert body["stage17_hardcase_dataset"]["accepted_traces"] == 18
    assert body["stage17_hardcase_dataset"]["samples"] == 154
    assert body["stage17_hardcase_dataset"]["stage16_scout"]["missing_completed_circuits"] == ["ghz_10"]
    assert body["stage18_finetune_gate"]["promote_checkpoint"] is False
    assert body["stage18_finetune_gate"]["decision"] == "hold_stage18_no_raw_checked7_completion_gain"
    assert body["stage18_finetune_gate"]["checked7"]["candidate"]["completed"] == 5
    assert body["stage18_finetune_gate"]["training"]["raw_samples"] == 154
    assert body["stage19_training_diagnostics"]["raw_samples"] == 154
    assert body["stage19_training_diagnostics"]["gpu_decision"]["gpu_recommended"] is False
    assert (
        body["stage19_training_diagnostics"]["gpu_decision"]["decision"]
        == "hold_gpu_until_objective_or_hyperparameter_change"
    )
    assert body["stage20_ghz10_stall_diagnostics"]["selected_executed_gates"] == 10
    assert body["stage20_ghz10_stall_diagnostics"]["short_suffix_exists"] is True
    assert body["stage20_ghz10_stall_diagnostics"]["suffix"] == [0, 36, 40]
    assert body["stage20_ghz10_stall_diagnostics"]["gpu_recommended"] is False
    assert body["stage21_suffix_repair_gate"]["passed"] is True
    assert body["stage21_suffix_repair_gate"]["baseline_completed"] == 6
    assert body["stage21_suffix_repair_gate"]["candidate_completed"] == 8
    assert body["stage21_suffix_repair_gate"]["repaired_circuits"] == ["qaoa_10", "ghz_10"]
    assert body["stage21_suffix_repair_gate"]["trace_replay_rows"] == 8
    assert body["stage21_suffix_repair_gate"]["gpu_recommended"] is False
    assert body["stage22_suffix_training_readiness"]["raw_samples"] == 226
    assert body["stage22_suffix_training_readiness"]["repaired_suffix_samples"] == 40
    assert body["stage22_suffix_training_readiness"]["gpu_decision"]["gpu_recommended"] is True
    assert body["stage22_suffix_training_readiness"]["gpu_decision"]["long_training_recommended"] is False
    assert body["stage23_gpu_sweep_plan"]["candidate_count"] == 3
    assert body["stage23_gpu_sweep_plan"]["overnight_allowed"] is False
    assert body["stage23_gpu_sweep_plan"]["promotion_policy"]["required_checked7_completion_gain"] == 1
    assert body["stage23_gpu_sweep_summary"]["decision"] == "hold_stage23_no_candidate_promoted"
    assert body["stage23_gpu_sweep_summary"]["pending_count"] == 0
    assert body["stage23_gpu_sweep_summary"]["completed_count"] == 3
    assert body["stage23_gpu_sweep_summary"]["promoted_count"] == 0
    assert body["stage24_training_go_no_go"]["decision"] == "hold_long_training_after_stage23_no_promotion"
    assert body["stage24_training_go_no_go"]["bounded_gpu_sweep_recommended"] is True
    assert body["stage24_training_go_no_go"]["long_training_allowed"] is False
    assert body["stage24_training_go_no_go"]["overnight_allowed"] is False
    assert body["stage24_training_go_no_go"]["training_readiness"]["accepted_traces"] == 26
    assert body["stage24_training_go_no_go"]["training_readiness"]["samples"] == 226
    assert body["stage24_training_go_no_go"]["training_readiness"]["stage23_seed_valid"] is False
    assert body["stage24_training_go_no_go"]["cloud_preflight"]["passed"] is True
    assert body["stage24_training_go_no_go"]["cloud_preflight"]["command_boundary_ok"] is True
    assert body["stage25_post_sweep_decision"]["decision"] == "hold_stage25_no_promoted_stage23_checkpoint"
    assert body["stage25_post_sweep_decision"]["long_training_allowed"] is False
    assert body["stage25_post_sweep_decision"]["stage23_pending_count"] == 0
    assert body["stable_public_default"]["backend"] == "npqr"
    assert body["stable_public_default"]["baseline"]["backend"] == "sabre"
    assert "NPQR does not generally outperform SABRE" in body["npqr_boundary"]["not_claimed"][0]


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
    assert body["baseline"]["status"] == "OK"


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
    for example in PUBLIC_EXAMPLES:
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


def test_api_compile_checked_in_examples_with_npqr_route_trace():
    for example in PUBLIC_EXAMPLES:
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


def test_api_compile_npqr_accepts_small_inline_qasm():
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
            "backend": "npqr",
            "topology": "tokyo",
            "heuristic": "lookahead",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["backend"] == "npqr"
    assert body["circuit_name"] == "inline_qasm"
    assert body["compiled_qasm"].startswith("OPENQASM 2.0;")
    assert body["route_trace"]
    _assert_routed_qasm_matches_tokyo(body["compiled_qasm"])


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
