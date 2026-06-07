"""Smoke tests for the TensorFlow Playground-style public site."""
from __future__ import annotations

from pathlib import Path


def _html() -> str:
    return Path("docs/index.html").read_text(encoding="utf-8")


def test_public_site_uses_tensorflow_playground_style_shell():
    html = _html()

    assert "<title>ZJU 量子电路编译器</title>" in html
    assert "Quantum <b>Compiler</b> Playground" in html
    assert "--tf-header: #183D4E" in html
    assert "--tf-orange: #f59322" in html
    assert "--tf-blue: #0877bd" in html
    assert 'id="top-controls"' in html
    assert 'id="main-part"' in html
    assert 'class="column data"' in html
    assert 'class="column topology"' in html
    assert 'class="column output"' in html
    assert "0 24px 58px" not in html


def test_public_site_adds_restrained_qcanvas_atmosphere():
    html = _html()

    assert "--qc-void: #060812" in html
    assert "--qc-cyan: #22d3ee" in html
    assert "--qc-violet: #7c5cff" in html
    assert "radial-gradient(circle at 18px 18px" in html
    assert "linear-gradient(135deg, var(--qc-void)" in html
    assert "body::before" in html
    assert "body::after" in html
    assert "border-image: linear-gradient(90deg, var(--qc-cyan), var(--qc-violet), var(--tf-orange)) 1" in html
    assert "body.is-running .topology-stage-wrap::after" not in html
    assert "@keyframes topology-scan" not in html


def test_public_site_keeps_first_screen_action_focused():
    html = _html()

    assert 'id="run-button"' in html
    assert 'id="step-button"' in html
    assert 'id="reset-button"' in html
    assert 'class="play-button"' in html
    assert "▶" in html
    assert "↺" in html
    assert "▸" in html
    assert "command-label" in html
    assert ">Run</span>" in html
    assert ">Reset</span>" in html
    assert ">Step</span>" in html
    assert "中文实验台" in html
    assert "选择 OpenQASM 示例" in html
    assert "点击 Run 编译当前电路" in html
    assert "setRunVisual" in html
    assert "route-step" not in html
    assert "route-steps" not in html
    assert "seed=42" not in html
    assert "trials=1" not in html
    assert "43 undirected" not in html
    assert "编译上传电路" not in html
    assert "量子电路编译器 MCP 入口" not in html


def test_public_site_defaults_to_public_fastapi_rest_backend():
    html = _html()

    assert 'const PUBLIC_API_BASE = "http://1.95.70.10";' in html
    assert "LOCAL_API_BASE" not in html
    assert "src.server.app:app" in html
    assert "fetch(`${base}${path}`" in html
    assert "const API_TIMEOUT_MS = 120000;" in html
    assert "const API_STATUS_TIMEOUT_MS = 20000;" in html
    assert '"/api/status"' in html
    assert 'callApi("/api/compile"' in html
    assert 'callApiAt(candidate, "/api/status"' in html
    assert "normalizeTrace(data.route_trace)" in html
    assert 'method: "POST"' in html
    assert 'backend: "npqr"' in html
    assert 'topology: "tokyo"' in html
    assert "?api=" in html
    assert 'id="api-base-input"' in html
    assert 'id="apply-api-base"' in html
    assert 'window.location.protocol === "file:"' not in html
    assert "public users need the deployed rest api" in html.lower()
    assert "http://localhost:8765/api/compile" not in html
    assert "http://127.0.0.1:8765/api/compile" not in html
    assert "GitHub Pages 不能单独编译电路" not in html


def test_public_site_exposes_examples_custom_qasm_and_heuristic_controls():
    html = _html()

    for example in ["qft5", "ghz5", "qaoa5", "qft10", "qaoa10", "ghz10", "vqe10", "custom"]:
        assert f'data-example="{example}"' in html
        assert f'<option value="{example}"' in html

    for heuristic in ["basic", "lookahead", "decay"]:
        assert f'data-heuristic="{heuristic}"' in html
        assert f'<option value="{heuristic}"' in html

    assert 'id="qasm-input"' in html
    assert "MAX_INLINE_QASM_CHARS = 8000" in html
    assert "Custom input must start with OPENQASM 2.0;" in html


def test_public_site_renders_lightweight_svg_tokyo_topology():
    html = _html()

    assert '<svg class="topology-stage" id="tokyo-topology"' in html
    assert 'aria-label="SVG IBM Tokyo 20Q topology animation"' in html
    assert "const topologyNodes = [" in html
    assert "const topologyEdges = [" in html
    assert ".topology-edge.active" in html
    assert ".topology-edge.gate-edge" in html
    assert ".topology-edge.swap-edge" in html
    assert ".topology-node.active" in html
    assert "showTooltip" in html
    assert "stepOnce" in html
    assert "startTraceReplay" in html
    assert "traceEdge(event)" in html
    assert 'id="trace-op"' in html
    assert 'id="trace-progress"' in html
    assert 'id="trace-bar"' in html
    assert 'class="compiler-flow"' in html
    assert 'class="flow-stage is-active" data-stage="0"' in html
    assert "@keyframes route-pulse" in html
    assert "animation: route-pulse 0.85s linear infinite" in html
    assert "stroke: var(--qc-cyan)" in html
    assert "filter: drop-shadow" not in html


def test_public_site_exposes_required_outputs_and_brief_error_state():
    html = _html()

    for element_id in [
        "metric-status",
        "metric-swaps",
        "metric-depth",
        "metric-elapsed",
        "compiled-qasm-output",
        "compiled-qasm-code",
        "compiled-state",
        "copy-qasm",
        "download-qasm",
    ]:
        assert f'id="{element_id}"' in html

    assert "elapsed_ms" in html
    assert "等待结果" in html
    assert "setCompiledState" in html
    assert 'id="copy-qasm" type="button" disabled' in html
    assert 'id="download-qasm" type="button" disabled' in html
    assert "Backend unavailable." in html
    assert "Backend unavailable. REST API is offline." in html
    assert "REST API is offline, not MCP." in html
    assert "Compile timed out before the browser received the result." in html
    assert "The REST API may still be computing; this is not necessarily offline." in html
    assert "这个电路计算时间较长，服务器可能仍在运行，请稍后重试。" in html
    assert "// no compiled_qasm" in html
    assert "公开页面不会自动探测访问者电脑上的本地服务" in html
    assert "OpenQASM routing on IBM Tokyo" in html


def test_public_site_keeps_mcp_as_folded_advanced_entry():
    html = _html()

    assert '<details class="page" id="advanced"' in html
    assert "Advanced" in html
    assert "MCP `/mcp` 保留给高级客户端和审阅流程" in html
    assert "qcompiler-mcp-http" in html
    assert "uvicorn src.server.mcp_app:app --host 0.0.0.0 --port $PORT" in html
    assert "POST /mcp" in html

    for tool in [
        "compile_qasm",
        "compile_npqr",
        "compile_sabre",
        "list_examples",
        "qcompiler_status",
        "get_benchmarks",
        "get_npqr_boundary",
        "get_npqr_stage7_evidence",
    ]:
        assert tool in html


def test_public_site_keeps_required_domain_terms_and_honest_boundary():
    html = _html()

    for term in ["MCP", "OpenQASM 2", "SABRE", "SWAP", "Qiskit", "IBM Tokyo", "FastAPI REST"]:
        assert term in html

    assert "NPQR evidence boundary; no general SABRE win claim." in html
    assert "后端默认使用 NPQR，Qiskit SABRE 作为对比基线" in html
    assert "不宣传全面超过 SABRE" not in html
    assert "登录" not in html
    assert "注册" not in html


def test_public_site_preserves_accessibility_and_responsive_basics():
    html = _html()

    assert 'lang="zh-CN"' in html
    assert 'aria-label="一键量子编译 Playground"' in html
    assert 'aria-label="运行控制"' in html
    assert 'aria-label="编译结果摘要"' in html
    assert 'aria-label="高级 MCP 和部署入口"' in html
    assert "overflow-x: clip;" in html
    assert "@media (max-width: 980px)" in html
    assert "@media (max-width: 520px)" in html


def test_public_site_does_not_reintroduce_old_overclaims():
    html = _html()
    forbidden = [
        "AI OUTPERFORMS HEURISTICS",
        "Quantum Router V9",
        "V9 Universal Model",
        "MTx100",
        "quickstart_v9.py",
        "models/v9_tokyo20",
        "打败经典启发式",
    ]

    for phrase in forbidden:
        assert phrase not in html
