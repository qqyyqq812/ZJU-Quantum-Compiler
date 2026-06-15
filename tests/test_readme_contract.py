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
    "组员分工",
    "大" + "作业",
    "算法" + "设计与" + "智能" + "计算",
    "算法" + "大" + "作业",
    "课程" + "项目",
    "课程" + "报告",
    "课程" + "作业",
    "课程" + "代表",
    "课程" + "算法",
    "智能" + "计算",
    "人工PR",
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
    assert "REST API" in readme
    assert "MCP service" in readme
    assert "docs/项目说明.md" in readme
    assert "docs/ai-collaboration.md" in readme


def test_readme_documents_algorithm_without_internal_experiment_names():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Score legal SWAP actions with a neural model" in readme
    assert "candidate logical-to-physical initial mappings" in readme
    assert "bounded beam search" in readme
    assert "Repair difficult suffixes" in readme
    assert "SABRE is kept as a fixed Qiskit baseline" in readme
    assert "not used as a hidden fallback" in readme
    assert "Algorithm overview" in readme
    assert "algorithmic ideas" in readme

    for pattern in FORBIDDEN_PUBLIC_PATTERNS:
        assert re.search(pattern, readme) is None


def test_readme_documents_api_and_mcp_tools():
    readme = Path("README.md").read_text(encoding="utf-8")

    for term in [
        "GET /api/status",
        "GET /api/examples",
        "POST /api/validate",
        "POST /api/compile",
        "POST /api/compile/jobs",
        "GET /api/topology/{name}",
        "GET /api/npqr/evidence",
        "compile_qasm",
        "compile_npqr",
        "compile_sabre",
        "get_algorithm_evidence",
    ]:
        assert term in readme


def test_readme_documents_readiness_and_package_commands():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "qcompiler matrix --quick" in readme
    assert "python scripts/check_submission_readiness.py" in readme
    assert "python scripts/package_submission.py" in readme
    assert "git diff --check" in readme
    assert "results/submission_package/" in readme
    assert "examples/line_ghz50.qasm" in readme


def test_readme_uses_objective_project_language_without_process_notes():
    readme = Path("README.md").read_text(encoding="utf-8")

    forbidden_terms = [
        "清" + "理版",
        "第几" + "版",
        "最终" + "版本",
        "公开仓库包含" + "最终",
        "public " + "tree",
        "public release " + "branch",
        "按" + "要求",
        "我" + "要求",
        "提示" + "词",
        "交" + "接",
        "组员" + "分工",
        "大" + "作业",
    ]

    for term in forbidden_terms:
        assert term not in readme

    assert "neural-assisted quantum circuit routing compiler" in readme
    assert "The project does not claim that NPQR beats SABRE on every circuit." in readme
