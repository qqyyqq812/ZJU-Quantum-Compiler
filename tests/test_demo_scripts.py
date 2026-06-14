"""Regression tests for final public helper scripts."""
from __future__ import annotations

from pathlib import Path


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
        "docs/final-closure-report.md",
        "docs/report_latex/main.pdf",
        "docs/plans/组员分工.md",
        "docs/slides/quantum-routing-algorithm-showcase-final.pptx",
        "examples/qft5.qasm",
        "readiness.md",
        "algorithm_matrix.json",
        "public_algorithm_evidence.json",
        "algorithm_summary.md",
        "npqr-course-report.pdf",
        "results/submission_package",
        "项目说明.md",
        "组员分工.md",
    ]

    for entry in required_entries:
        assert entry in script

    assert "npqr_" + "stage" not in script
    assert "St" + "age" not in script
