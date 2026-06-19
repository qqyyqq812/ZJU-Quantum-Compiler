"""Contract tests for the public Playground user guide."""
from __future__ import annotations

from pathlib import Path


GUIDE = Path("docs/playground-user-guide.md")


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_playground_user_guide_exists_and_covers_browser_workflow():
    guide = _guide()

    assert "# 网页体验指南" in guide
    assert "量子信息基础大作业" in guide
    assert "三分钟体验" in guide
    assert "**Run**" in guide
    assert "**Step**" in guide
    assert "Examples" in guide
    assert "Custom QASM" in guide
    assert "Generate" in guide
    assert "NPQR QASM" in guide
    assert "SABRE QASM" in guide
    assert "compiled_qasm" in guide


def test_playground_user_guide_explains_metrics_mapping_and_trace():
    guide = _guide()

    assert "IBM Tokyo 20Q" in guide
    assert "grid_5x6" in guide
    assert "grid_5x10" in guide
    assert "扩展规模" in guide
    assert "SWAP" in guide
    assert "Depth" in guide
    assert "elapsed_ms" in guide
    assert "route_trace" in guide
    assert "mapping_before" in guide
    assert "mapping_after" in guide
    assert "physical_qubits" in guide
    assert "logical_qubits" in guide


def test_playground_user_guide_documents_surface_roles():
    guide = _guide()

    assert "GitHub Pages" in guide
    assert "REST API" in guide
    assert "MCP" in guide
    assert "运行位置" in guide
    assert "公共服务器" in guide
    assert "本地后端" in guide
    assert "不是普通网页体验的必需入口" in guide
    assert "SABRE basic 是固定 Qiskit baseline" in guide
    assert "同一输入线路、同一拓扑" in guide


def test_readme_and_public_site_link_to_playground_user_guide():
    readme = Path("README.md").read_text(encoding="utf-8")
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert "docs/playground-user-guide.md" in readme
    assert "快速入口" in readme
    assert "fixed Qiskit baseline" in readme
    assert "IBM Tokyo 20Q" in readme
    assert 'href="playground-user-guide.md"' in html
    assert "指南" in html
