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
    assert "启动 REST API" in readme
    assert "启动 MCP 服务" in readme
    assert "docs/项目说明.md" in readme
    assert "docs/plans/组员分工.md" in readme


def test_readme_documents_algorithm_without_internal_experiment_names():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "神经网络对合法 SWAP 动作进行评分" in readme
    assert "初始映射候选" in readme
    assert "有界束搜索" in readme
    assert "后缀修复" in readme
    assert "SABRE 是对比基线" in readme
    assert "不是 NPQR 结果的隐藏完成路径" in readme
    assert "课程算法概念对应" in readme

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
    ]

    for term in forbidden_terms:
        assert term not in readme

    assert "这是一个面向受限量子硬件的神经辅助量子路由编译器" in readme
    assert "项目不声明 NPQR 在所有电路上都优于 SABRE" in readme
