"""Contract tests for the repository README entry points."""
from __future__ import annotations

from pathlib import Path


def test_readme_documents_live_website_demo_checklist():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## 现场演示 checklist" in readme
    assert "uvicorn src.server.app:app --reload --port 8765" in readme
    assert "python -m http.server 8766 --directory docs" in readme
    assert "http://127.0.0.1:8766/index.html" in readme
    assert "qcompiler compile examples/qft10.qasm --topology tokyo --backend sabre --heuristic lookahead" in readme
    assert "`--heuristic basic|lookahead|decay`" in readme
    assert "The default is `lookahead`, with `seed=42`" in readme
    assert "选择 `QFT 10` 和 `lookahead`" in readme
    assert "`SWAP` 应为 `29`" in readme
    assert "`match` 应为 `yes`" in readme
    assert "mini GHZ OpenQASM" in readme
    assert "如果现场 API 无法启动" in readme
    assert "不能声称完成了 live API 复现" in readme


def test_readme_keeps_honest_project_positioning():
    readme = Path("README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "SABRE / LightSABRE 是稳定实用基线" in normalized
    assert "V14/V15 AI checkpoint" in normalized
    assert "尚未超过 SABRE" in normalized
    assert "bounded search 负样本" in normalized
    assert "AI-router 历史只作为工作量和失败分析证据" in normalized
