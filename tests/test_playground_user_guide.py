"""Contract tests for the public Playground user guide."""
from __future__ import annotations

from pathlib import Path


GUIDE = Path("docs/playground-user-guide.md")


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_playground_user_guide_exists_and_covers_browser_workflow():
    guide = _guide()

    assert "# Quantum Compiler Playground user guide" in guide
    assert "backend indicator" in guide
    assert "**Run**" in guide
    assert "**Step**" in guide
    assert "**Reset**" in guide
    assert "Examples" in guide
    assert "Custom QASM" in guide
    assert "Generate" in guide
    assert "NPQR QASM" in guide
    assert "SABRE QASM" in guide
    assert "compiled_qasm" in guide


def test_playground_user_guide_explains_metrics_mapping_and_trace():
    guide = _guide()

    assert "IBM Tokyo 20Q" in guide
    assert "20 points are physical qubits" in guide
    assert "logical-to-physical mapping" in guide
    assert "SWAP" in guide
    assert "Depth" in guide
    assert "elapsed_ms" in guide
    assert "route_trace" in guide
    assert "mapping_before" in guide
    assert "mapping_after" in guide
    assert "physical_qubits" in guide
    assert "logical_qubits" in guide


def test_playground_user_guide_documents_backend_boundary():
    guide = _guide()

    assert "REST API" in guide
    assert "src.server.app:app" in guide
    assert "POST /api/compile" in guide
    assert "backend=\"npqr\"" in guide
    assert "backend=\"sabre\"" in guide
    assert "neural-assisted selector, search, and repair runtime" in guide
    assert "Qiskit `SabreSwap`" in guide
    assert "standard `basic` heuristic" in guide
    assert "`seed=42`" in guide
    assert "`trials=1`" in guide
    assert "does not claim that NPQR always beats SABRE" in guide
    assert "MCP is an advanced helper" in guide


def test_readme_and_public_site_link_to_playground_user_guide():
    readme = Path("README.md").read_text(encoding="utf-8")
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert "docs/playground-user-guide.md" in readme
    assert "固定 SABRE 基线" in readme
    assert "Tokyo 映射" in readme
    assert 'href="playground-user-guide.md"' in html
    assert "Playground user guide" in html
