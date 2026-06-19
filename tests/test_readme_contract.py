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
    r"(?<!量子信息基础)大作业",
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
    assert "语言 / Language" in readme
    assert "## 快速入口" in readme
    assert "普通体验" in readme
    assert "本地部署" in readme
    assert "工具调用" in readme
    assert "English version" in readme
    assert "pip install -r requirements.txt" in readme
    assert "qcompiler info" in readme
    assert "uvicorn src.server.app:app --host 0.0.0.0 --port 8765" in readme
    assert "qcompiler-mcp-http" in readme
    assert "http://127.0.0.1:8000/mcp" in readme
    assert "curl -s http://127.0.0.1:8000/health" in readme
    assert "models/default/npqr-default.pt" in readme
    assert "REST API" in readme
    assert "MCP 服务" in readme
    assert "docs/项目说明.md" in readme
    assert "docs/ai-collaboration.md" in readme


def test_readme_documents_algorithm_without_internal_experiment_names():
    readme = Path("README.md").read_text(encoding="utf-8")
    compact = readme.replace("\n", "")

    assert "使用神经模型对合法 SWAP 动作评分" in readme
    assert "初始映射候选" in readme
    assert "有界束搜索" in readme
    assert "局部修复" in readme
    assert "SABRE basic 作为固定对照基线" in compact
    assert "方法概述" in readme
    assert "图约束下的近似优化问题" in readme

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


def test_readme_documents_readiness_commands():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "qcompiler matrix --quick" in readme
    assert "python scripts/check_submission_readiness.py" in readme
    assert "git diff --check" in readme
    assert "examples/line_ghz50.qasm" in readme


def test_readme_uses_objective_project_language_without_process_notes():
    readme = Path("README.md").read_text(encoding="utf-8")
    compact = readme.replace("\n", "")

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
        "算法" + "大" + "作业",
        "课程" + "项目",
        "课程" + "报告",
        "hidden fallback",
        "80/100Q",
        "80Q",
        "100Q",
    ]

    for term in forbidden_terms:
        assert term not in readme

    assert "受限硬件拓扑的量子线路路由编译器" in compact
    assert "快速入口" in readme
    assert "普通体验" in readme
    assert "本地部署" in readme
    assert "MCP 面向工具客户端" in compact
    assert "30Q 和 50Q 示例属于扩展规模，需要部署" in compact
