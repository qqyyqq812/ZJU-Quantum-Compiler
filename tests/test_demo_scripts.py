"""Regression tests for final public helper scripts."""
from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit

from scripts import measure_phase_timing
from src import cli
from src.compiler.profile_timing import PhaseProfiler, maybe_measure
from src.server.mcp_app import compile_sabre, list_examples


def test_public_demo_script_uses_default_npqr_and_api_smoke():
    script = Path("run_public_demo.sh").read_text(encoding="utf-8")

    assert "models/default/npqr-default.pt" in script
    assert "qcompiler info" in script
    assert "TestClient" in script
    assert "from src.server.app import app" in script
    assert "/api/status" in script
    assert "/api/compile" in script
    assert '"backend": "npqr"' in script
    assert '"backend": "sabre"' in script
    assert "checkpoint_" + "ep" not in script
    assert "V" + "14" not in script


def test_teacher_eval_script_wraps_public_demo():
    script = Path("run_teacher_eval.sh").read_text(encoding="utf-8")

    assert "run_public_demo.sh" in script
    assert "results/teacher_demo" in script
    assert "NPQR is the default route" in script
    assert "SABRE is the comparison baseline" in script
    assert "checkpoint_" + "ep" not in script


def test_submission_package_script_defines_public_review_manifest():
    script = Path("scripts/package_submission.py").read_text(encoding="utf-8")

    required_entries = [
        "README.md",
        "docs/index.html",
        "docs/项目说明.md",
        "docs/playground-user-guide.md",
        "docs/ai-collaboration.md",
        "docs/final-closure-report.md",
        "docs/report_latex/main.pdf",
        "examples/qft5.qasm",
        "examples/line_ghz30.qasm",
        "examples/random30_d4.qasm",
        "examples/line_ghz50.qasm",
        "examples/ring_sparse50.qasm",
        "readiness.md",
        "algorithm_matrix.json",
        "public_algorithm_evidence.json",
        "algorithm_summary.md",
        "npqr-technical-report.pdf",
        "results/submission_package",
        "项目说明.md",
        "ai-collaboration.md",
    ]

    for entry in required_entries:
        assert entry in script

    assert "npqr_" + "stage" not in script
    assert "St" + "age" not in script


def test_cli_and_mcp_expose_checked_in_large_examples():
    assert "line_ghz50" in cli._EXAMPLES
    assert cli._EXAMPLE_TOPOLOGIES["line_ghz50"] == "grid_5x10"

    examples = {row["id"]: row for row in list_examples()["examples"]}
    assert examples["line_ghz30"]["topology"] == "grid_5x6"
    assert examples["ring_sparse50"]["topology"] == "grid_5x10"

    result = compile_sabre(
        example="line_ghz30",
        heuristic="basic",
        topology="grid_5x6",
    )
    assert result["status"] == "OK"
    assert result["backend"] == "sabre"
    assert result["topology"] == "grid_5x6"
    assert result["input_qubits"] == 30


def test_checked_in_large_qasm_files_are_parseable():
    for path in [
        Path("examples/line_ghz30.qasm"),
        Path("examples/random30_d4.qasm"),
        Path("examples/line_ghz50.qasm"),
        Path("examples/ring_sparse50.qasm"),
    ]:
        circuit = QuantumCircuit.from_qasm_file(str(path))
        assert circuit.num_qubits in {30, 50}


def test_npqr_phase_timing_script_exposes_required_stage_contract():
    labels = measure_phase_timing.PHASE_LABELS

    for phase in [
        "data_preprocessing",
        "dependency_graph_build",
        "topology_distance_matrix",
        "logical_interaction_graph",
        "initial_mapping_candidates",
        "main_search",
        "action_generation_mask",
        "neural_network_inference",
        "beam_expand_prune",
        "postprocessing",
        "suffix_repair",
        "trace_replay_validation",
        "total",
    ]:
        assert phase in labels

    assert measure_phase_timing.CASE_SPECS["line_ghz30"].topology == "grid_5x6"
    assert measure_phase_timing.CASE_SPECS["line_ghz50"].topology == "grid_5x10"
    assert "qaoa10" in measure_phase_timing.DEFAULT_CASES
    assert "brickwork20_profile" in measure_phase_timing.DEFAULT_CASES


def test_npqr_phase_summary_preserves_timeout_wall_time():
    runs = [
        {
            "case_id": "line_ghz30",
            "status": "timeout",
            "wall_ms": 20000.0,
            "phase_timings": {},
        },
        {
            "case_id": "line_ghz30",
            "status": "timeout",
            "wall_ms": 21000.0,
            "phase_timings": {},
        },
    ]

    rows = measure_phase_timing._summarize_runs(runs)
    total = next(row for row in rows if row["phase"] == "total")

    assert total["ok_runs"] == 0
    assert total["timeouts"] == 2
    assert total["mean_ms"] == 0.0
    assert total["timeout_wall_mean_ms"] == 20500.0
    assert total["timeout_wall_min_ms"] == 20000.0
    assert total["timeout_wall_max_ms"] == 21000.0


def test_phase_profiler_records_nested_measurements():
    profiler = PhaseProfiler()

    with maybe_measure(profiler, "unit_test_phase"):
        sum(range(100))
    profiler.add("unit_test_phase", 1.5)

    data = profiler.to_dict()["unit_test_phase"]
    assert data["count"] == 2
    assert data["total_ms"] >= 1.5
    assert data["max_ms"] >= data["min_ms"]
