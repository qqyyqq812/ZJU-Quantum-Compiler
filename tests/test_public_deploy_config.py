"""Deployment contract for the public REST and MCP backends."""
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


def test_docker_images_copy_only_default_model():
    api_dockerfile = Path("Dockerfile.api").read_text(encoding="utf-8")
    mcp_dockerfile = Path("Dockerfile.mcp").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "COPY models/default/npqr-default.pt ./models/default/npqr-default.pt" in api_dockerfile
    assert "COPY models/default/npqr-default.pt ./models/default/npqr-default.pt" in mcp_dockerfile
    assert "!models/default/npqr-default.pt" in dockerignore
    assert "checkpoint_" + "ep" not in api_dockerfile
    assert "checkpoint_" + "ep" not in mcp_dockerfile


def test_rest_app_lazily_loads_npqr_router_for_deploys():
    app_source = Path("src/server/app.py").read_text(encoding="utf-8")

    assert "\nfrom src.compiler.pass_" + "manager import AI" + "Router\n" not in app_source
    assert "\nfrom src.compiler.npqr_runtime import NPQRRuntime\n" not in app_source
    assert "def _router_for(" not in app_source
    assert "def _npqr_runtime_for(" in app_source
    assert "AI" + "Router" not in app_source
    assert "from src.compiler.npqr_runtime import NPQRRuntime" in app_source
    assert "except ImportError as exc:" in app_source
