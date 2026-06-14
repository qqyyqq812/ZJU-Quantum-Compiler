"""Smoke tests for the single public quantum compiler page."""
from __future__ import annotations

from pathlib import Path


DOCS = Path("docs")


def _html() -> str:
    return (DOCS / "index.html").read_text(encoding="utf-8")


def _first_screen(html: str) -> str:
    body = html.split('<body class="instrument-v2 quantum-studio-preview">', 1)[1]
    return body.split('<details class="page" id="advanced"', 1)[0]


def test_public_site_has_one_formal_entry_page():
    html = _html()
    extra_pages = sorted(path.name for path in DOCS.glob("index-*.html"))

    assert extra_pages == []
    assert "<title>ZJU 量子电路编译器 · Quantum Studio</title>" in html
    assert '<body class="instrument-v2 quantum-studio-preview">' in html
    assert "量子编译实验台" in html
    assert "--studio-bg: #eef2f5" in html
    assert "grid-template-columns: 282px minmax(620px, 1fr) 304px" in html
    assert '"data topology output"\n        "data inspector output";' in html


def test_public_site_keeps_first_screen_action_focused():
    first_screen = _first_screen(_html())

    for element_id in ["run-button", "step-button", "reset-button", "example-select", "backend-status"]:
        assert f'id="{element_id}"' in first_screen
    assert ">运行</span>" in first_screen
    assert 'id="tour-button"' in first_screen
    assert 'id="validation-bar"' in first_screen
    assert "等待校验" in first_screen
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
    assert "GitHub Pages 默认使用内置示例结果" in html
    assert "GitHub Pages 需要 HTTPS REST API" in html
    assert "避免浏览器混合内容拦截" in html
    assert "?api=https://your-api.example" in html
    assert "http://localhost:8765/api/compile" not in html
    assert "http://127.0.0.1:8765/api/compile" not in html


def test_public_site_runs_embedded_examples_without_second_display_mode():
    html = _html()

    assert "const embeddedResults = {" in html
    assert "function embeddedCompileResult" in html
    assert "function compileWithBestAvailableBackend" in html
    assert "makeEmbeddedTrace" in html
    assert "当前使用 GitHub Pages 内置示例结果" in html
    assert "自定义或生成电路需要连接 HTTPS REST API" in html
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
    assert "第一行需要 OPENQASM 2.0;。" in html


def test_public_site_renders_tokyo_topology_and_trace_review():
    html = _html()

    assert '<svg class="topology-stage" id="tokyo-topology"' in html
    assert 'aria-label="SVG IBM Tokyo 20Q topology animation"' in html
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
    assert "sample.path" not in html
    assert "Preview Tokyo edge" not in html


def test_public_site_keeps_mcp_as_folded_advanced_entry():
    html = _html()
    first_screen = _first_screen(html)

    assert '<details class="page" id="advanced"' in html
    assert "高级入口" in html
    assert "MCP `/mcp` 保留给高级客户端和审阅流程" in html
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

    assert 'lang="zh-CN"' in html
    assert 'aria-label="一键量子编译 Playground"' in html
    assert 'aria-label="运行控制"' in html
    assert 'aria-label="NPQR 与 SABRE 对比"' in html
    assert 'aria-label="高级 MCP 和部署入口"' in html
    assert "@media (max-width: 1120px)" in html
    assert "@media (max-width: 760px)" in html


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
