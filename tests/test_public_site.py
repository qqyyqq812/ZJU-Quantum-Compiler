"""Smoke tests for the single public quantum compiler page."""
from __future__ import annotations

from pathlib import Path


DOCS = Path("docs")


def _html() -> str:
    return (DOCS / "index.html").read_text(encoding="utf-8")


def _first_screen(html: str) -> str:
    body = html.split('<body class="compiler-console">', 1)[1]
    return body.split('<details class="page" id="advanced"', 1)[0]


def test_public_site_has_one_formal_entry_page():
    html = _html()
    css = (DOCS / "console.css").read_text(encoding="utf-8")
    extra_pages = sorted(path.name for path in DOCS.glob("index-*.html"))

    assert extra_pages == []
    assert "<title>ZJU Quantum Compiler Console</title>" in html
    assert '<body class="compiler-console">' in html
    assert '<link rel="stylesheet" href="console.css">' in html
    assert "ZJU Quantum Compiler Console" in html
    assert "量子电路编译控制台" in html
    assert "grid-template-columns: 300px minmax(620px, 1fr) 330px" in css
    assert '"data topology output"\n    "data inspector output";' in css
    assert "instrument-preview" not in html
    assert "instrument-v2" not in html
    assert "quantum-studio-preview" not in html
    for phrase in ["tour", "Guide", "Skip", "TOUR_STORAGE_KEY", "data-tour"]:
        assert phrase not in html


def test_public_site_keeps_first_screen_action_focused():
    first_screen = _first_screen(_html())

    for element_id in ["run-button", "step-button", "reset-button", "example-select", "backend-status"]:
        assert f'id="{element_id}"' in first_screen
    assert 'id="run-label">Run</span>' in first_screen
    assert "Embedded Review" in first_screen
    assert "REST: Optional" in first_screen
    assert 'id="tour-button"' not in first_screen
    assert 'id="validation-bar"' in first_screen
    assert "Pending" in first_screen
    assert "data-heuristic" not in first_screen
    assert "seed=42" not in first_screen
    assert "trials" not in first_screen
    assert "本地部署" not in first_screen
    assert "MCP" not in first_screen
    assert "登录" not in first_screen
    assert "注册" not in first_screen


def test_public_site_is_safe_for_https_github_pages():
    html = _html()

    assert 'const PUBLIC_API_BASE = "";' in html
    assert 'let apiMode = apiBase ? "remote" : "embedded";' in html
    assert 'window.location.protocol === "https:"' in html
    assert 'params.get("allowHttpApi") !== "1"' in html
    assert "Embedded review data active" in html
    assert "HTTPS is required on GitHub Pages" in html
    assert "allowHttpApi=1" in html
    assert "?api=https://your-api.example" in html
    assert "http://localhost:8765/api/compile" not in html
    assert "http://127.0.0.1:8765/api/compile" not in html


def test_public_site_runs_embedded_examples_without_second_display_mode():
    html = _html()

    assert "const embeddedResults = {" in html
    assert "function embeddedCompileResult" in html
    assert "function compileWithBestAvailableBackend" in html
    assert "makeEmbeddedTrace" in html
    assert "Embedded review data active; HTTPS REST API enables live compilation." in html
    assert "Custom or generated circuits require HTTPS REST API." in html
    assert "return callApi(\"/api/compile\"" in html
    assert "compileWithBestAvailableBackend(\"npqr\")" in html
    assert "compileWithBestAvailableBackend(\"sabre\")" in html
    assert "Promise.allSettled" in html

    for example in ["qft5", "ghz5", "qaoa5", "qft10", "qaoa10", "ghz10", "vqe10", "custom"]:
        assert f'data-example="{example}"' in html
        assert f'<option value="{example}"' in html


def test_public_site_exposes_real_inputs_outputs_and_fixed_baseline():
    html = _html()

    for element_id in [
        "qasm-input",
        "generator-family",
        "generator-qubits",
        "generator-layers",
        "generate-qasm",
        "npqr-status",
        "npqr-swaps",
        "npqr-depth",
        "npqr-elapsed",
        "sabre-status",
        "sabre-swaps",
        "sabre-depth",
        "sabre-elapsed",
        "delta-swaps",
        "delta-depth",
        "qasm-view-npqr",
        "qasm-view-sabre",
        "compiled-qasm-output",
        "compiled-qasm-code",
        "copy-qasm",
        "download-qasm",
    ]:
        assert f'id="{element_id}"' in html

    assert "SABRE" in html
    assert "data-heuristic" not in html
    assert 'id="heuristic-select"' not in html
    assert "MAX_INLINE_QASM_CHARS = 8000" in html
    assert "First line must be OPENQASM 2.0;." in html


def test_public_site_renders_tokyo_topology_and_trace_review():
    html = _html()
    css = (DOCS / "console.css").read_text(encoding="utf-8")

    assert '<svg class="topology-stage" id="tokyo-topology"' in html
    assert 'aria-label="IBM Tokyo 20Q topology animation"' in html
    assert "const topologyNodes = [" in html
    assert "const topologyEdges = [" in html
    assert "traceEdge(event)" in html
    assert "traceNode(event)" in html
    assert "traceStats" in html
    assert 'id="trace-op"' in html
    assert 'id="route-view-npqr"' in html
    assert 'id="route-view-sabre"' in html
    assert 'id="mapping-grid"' in html
    assert 'id="route-timeline"' in html
    assert "mapping_after" in html
    assert "previewPath" in html
    assert "function circuitPreviewTrace" in html
    assert "function displayTrace" in html
    assert "Preview | ${stats.edges} circuit edges" in html
    assert ".topology-edge.preview-edge" in css
    assert ".topology-stage-wrap" in css


def test_public_site_keeps_mcp_as_folded_advanced_entry():
    html = _html()
    first_screen = _first_screen(html)

    assert '<details class="page" id="advanced"' in html
    assert "Service Interfaces" in html
    assert "MCP Server" in html
    assert "qcompiler-mcp-http" in html
    assert "uvicorn src.server.mcp_app:app --host 0.0.0.0 --port $PORT" in html
    assert "POST /mcp" in html
    assert "MCP" not in first_screen

    for tool in [
        "compile_qasm",
        "compile_npqr",
        "compile_sabre",
        "list_examples",
        "qcompiler_status",
        "get_benchmarks",
        "get_npqr_boundary",
        "get_algorithm_evidence",
    ]:
        assert tool in html


def test_public_site_preserves_accessibility_and_responsive_basics():
    html = _html()
    css = (DOCS / "console.css").read_text(encoding="utf-8")

    assert 'lang="zh-CN"' in html
    assert 'aria-label="Quantum compiler console"' in html
    assert 'aria-label="Run controls"' in html
    assert 'aria-label="NPQR and SABRE comparison"' in html
    assert 'aria-label="Service interfaces"' in html
    assert "@media (max-width: 1120px)" in css
    assert "@media (max-width: 760px)" in css


def test_public_site_does_not_reintroduce_old_overclaims():
    html = _html()
    forbidden = [
        "AI OUTPERFORMS HEURISTICS",
        "Quantum Router V" + "9",
        "V" + "9 Universal Model",
        "MTx100",
        "quickstart_v" + "9.py",
        "models/v" + "9_tokyo20",
        "打败经典启发式",
    ]

    for phrase in forbidden:
        assert phrase not in html


def test_public_site_removes_explanatory_first_screen_copy():
    html = _html()
    first_screen = _first_screen(html)
    forbidden = [
        "高级入口已折叠",
        "高级入口：",
        "高级入口",
        "实验台",
        "GitHub Pages 内置示例",
        "这不是",
        "普通用户",
        "维护者",
    ]

    for phrase in forbidden:
        assert phrase not in html
        assert phrase not in first_screen


def test_public_site_has_no_chinese_code_comments():
    files = [
        DOCS / "index.html",
        DOCS / "console.css",
    ]

    for path in files:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.lstrip()
            is_comment = (
                stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
                or stripped.startswith("<!--")
            )
            assert not (is_comment and any("\u4e00" <= char <= "\u9fff" for char in stripped)), (
                path,
                line,
            )
