"""Deployment contract for the public REST playground backend."""
from __future__ import annotations

from pathlib import Path


def test_render_blueprint_deploys_fastapi_rest_backend_not_mcp():
    render_yaml = Path("render.yaml").read_text(encoding="utf-8")

    assert "name: zju-quantum-compiler-api" in render_yaml
    assert "runtime: python" in render_yaml
    assert "buildCommand: pip install -r requirements-api.txt" in render_yaml
    assert "startCommand: uvicorn src.server.app:app --host 0.0.0.0 --port $PORT" in render_yaml
    assert "healthCheckPath: /api/status" in render_yaml
    assert "src.server.mcp_app:app" not in render_yaml
    assert "POST /mcp" not in render_yaml


def test_optional_render_blueprint_deploys_mcp_helper_not_browser_api():
    render_yaml = Path("render-mcp.yaml").read_text(encoding="utf-8")

    assert "name: zju-quantum-compiler-mcp" in render_yaml
    assert "runtime: python" in render_yaml
    assert "buildCommand: pip install -r requirements-mcp.txt" in render_yaml
    assert "startCommand: uvicorn src.server.mcp_app:app --host 0.0.0.0 --port $PORT" in render_yaml
    assert "healthCheckPath: /health" in render_yaml
    assert "src.server.app:app" not in render_yaml
    assert "/api/compile" not in render_yaml


def test_api_requirements_include_npqr_runtime_but_keep_mcp_out():
    requirements = Path("requirements-api.txt").read_text(encoding="utf-8")

    for package in [
        "fastapi",
        "uvicorn",
        "qiskit",
        "networkx",
        "numpy",
        "scipy",
        "rustworkx",
        "torch",
        "gymnasium",
        "PyYAML",
    ]:
        assert package in requirements

    assert "mcp" not in requirements


def test_mcp_requirements_include_npqr_runtime_and_mcp_server():
    requirements = Path("requirements-mcp.txt").read_text(encoding="utf-8")

    for package in ["mcp[cli]", "fastapi", "uvicorn", "qiskit", "torch", "gymnasium", "PyYAML"]:
        assert package in requirements


def test_deployment_guide_separates_browser_api_from_mcp_helper():
    guide = Path("docs/API_MCP_DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "Browser route" in guide
    assert "src.server.app:app" in guide
    assert "POST /api/compile" in guide
    assert "MCP helper route" in guide
    assert "src.server.mcp_app:app" in guide
    assert "POST /mcp" in guide
    assert "The browser does not call MCP" in guide
    assert '"backend":"npqr"' in guide
    assert '"backend":"sabre"' in guide
    assert "What still needs a human" in guide


def test_rest_app_lazily_loads_ai_and_npqr_router_for_deploys():
    app_source = Path("src/server/app.py").read_text(encoding="utf-8")

    assert "\nfrom src.compiler.pass_manager import AIRouter\n" not in app_source
    assert "\nfrom src.compiler.npqr_runtime import NPQRRuntime\n" not in app_source
    assert "def _router_for(" in app_source
    assert "def _npqr_runtime_for(" in app_source
    assert "from src.compiler.pass_manager import AIRouter" in app_source
    assert "from src.compiler.npqr_runtime import NPQRRuntime" in app_source
    assert "except ImportError as exc:" in app_source
