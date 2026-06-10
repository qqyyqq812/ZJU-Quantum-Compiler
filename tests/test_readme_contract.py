"""Contract tests for the public README."""
from __future__ import annotations

import re
from pathlib import Path


FORBIDDEN_PUBLIC_PATTERNS = [
    r"\b" + "Stage" + r"\d+\b",
    "npqr_" + "stage",
    "npqr_" + "model_",
    "NEXT_" + "HAND" + "OFF",
    "HAND" + "OFF",
    "checkpoint_" + "ep",
    "wave2_" + "stage",
    "results/" + "npqr_",
]


def test_readme_documents_public_run_paths():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "# ZJU Quantum Compiler" in readme
    assert "pip install -r requirements.txt" in readme
    assert "qcompiler info" in readme
    assert "uvicorn src.server.app:app --host 0.0.0.0 --port 8765" in readme
    assert "qcompiler-mcp-http" in readme
    assert "http://127.0.0.1:8000/mcp" in readme
    assert "curl -s http://127.0.0.1:8000/health" in readme
    assert "models/default/npqr-default.pt" in readme


def test_readme_documents_algorithm_without_internal_experiment_names():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "NPQR is a combination pipeline" in readme
    assert "neural model is the core action scorer" in readme
    assert "initial mapping selection" in readme
    assert "bounded beam search" in readme
    assert "suffix repair" in readme
    assert "SABRE is the baseline" in readme
    assert "not as a hidden fallback" in readme
    assert "Course algorithm mapping" in readme

    for pattern in FORBIDDEN_PUBLIC_PATTERNS:
        assert re.search(pattern, readme) is None


def test_readme_documents_api_and_mcp_tools():
    readme = Path("README.md").read_text(encoding="utf-8")

    for term in [
        "GET /api/status",
        "POST /api/compile",
        "GET /api/npqr/evidence",
        "compile_qasm",
        "compile_npqr",
        "compile_sabre",
        "get_algorithm_evidence",
    ]:
        assert term in readme


def test_readme_documents_readiness_and_package_commands():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python scripts/experiment_algorithm_matrix.py --quick" in readme
    assert "python scripts/check_submission_readiness.py" in readme
    assert "python scripts/package_submission.py" in readme
    assert "git diff --check" in readme
    assert "results/submission_package/" in readme
