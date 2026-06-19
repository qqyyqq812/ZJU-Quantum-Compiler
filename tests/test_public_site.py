"""Smoke tests for the single public quantum compiler page."""
from __future__ import annotations

from pathlib import Path


DOCS = Path("docs")


def _html() -> str:
    return (DOCS / "index.html").read_text(encoding="utf-8")


def _first_screen(html: str) -> str:
    body = html.split('<body class="instrument-v2">', 1)[1]
    return body.split('<aside class="service-panel" id="advanced"', 1)[0]


def _embedded_example_qasm(html: str, example: str) -> str:
    marker = f"      {example}: {{"
    start = html.index(marker)
    qasm_start = html.index("        qasm: `", start) + len("        qasm: `")
    qasm_end = html.index("`\n      }", qasm_start)
    return html[qasm_start:qasm_end]


def test_public_site_has_one_formal_entry_page():
    html = _html()
    extra_pages = sorted(path.name for path in DOCS.glob("index-*.html"))

    assert extra_pages == []
    assert "<title>ZJU 量子电路编译器 · Lab Instrument V2</title>" in html
    assert '<body class="instrument-v2">' in html
    assert "量子编译实验台" in html
    assert "Lab Instrument Playground V2" in html
    assert "Quantum Studio" not in html
    assert "grid-template-columns: 276px minmax(600px, 1fr) 314px" in html
    assert "height: min(860px, calc(100vh - 112px))" in html
    assert 'grid-template-areas: "data topology output";' in html
    assert '"inspector inspector' not in html


def test_public_site_keeps_first_screen_action_focused():
    first_screen = _first_screen(_html())

    for element_id in ["run-button", "step-button", "reset-button", "example-select", "backend-status"]:
        assert f'id="{element_id}"' in first_screen
    assert ">运行</span>" in first_screen
    assert 'data-input-group="small"' in first_screen
    assert 'data-input-group="large"' in first_screen
    assert 'data-input-group="custom"' in first_screen
    assert ">10/20</button>" in first_screen
    assert ">30/50</button>" in first_screen
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


def test_public_site_uses_public_remote_rest_api():
    html = _html()
    first_screen = _first_screen(html)

    assert 'const PUBLIC_API_BASE = "http://1.95.70.10";' in html
    assert 'const PUBLIC_APP_BASE = "http://1.95.70.10/";' in html
    assert 'window.location.hostname.endsWith("github.io")' in html
    assert 'normalizeApiBase(params.get("api"))?.startsWith("http://")' in html
    assert 'window.location.replace(target.toString());' in html
    assert "function apiCandidates()" in html
    assert "return [apiBase].map(normalizeApiBase);" in html
    assert "运行会同时调用 NPQR 和固定 SABRE 基准，结果来自公网 REST API。" in html
    assert "浏览器调用项目部署的 FastAPI 服务" in html
    assert 'id="service-button"' in first_screen
    assert "Service Interfaces" in html
    assert "POST /api/compile/jobs" in html
    assert "http://localhost:8765/api/compile" not in html
    assert "http://127.0.0.1:8765/api/compile" not in html
    assert "GitHub Pages 默认使用内置示例结果" not in html


def test_public_site_runs_remote_compile_without_embedded_fallback():
    html = _html()

    assert "const embeddedResults = {" not in html
    assert "function embeddedCompileResult" not in html
    assert "function compileWithBestAvailableBackend" not in html
    assert "makeEmbeddedTrace" not in html
    assert html.count('callApi("/api/compile"') == 1
    assert html.count('callApi("/api/compile/jobs"') == 1
    assert "function createCompileJob(backend)" in html
    assert "async function pollCompileJob(job, backend, runId)" in html
    assert "function compileBackendJob(backend, runId)" in html
    assert "function compileBackend(backend)" in html
    assert "compileBackendJob(\"npqr\", runId)" in html
    assert "compileBackendJob(\"sabre\", runId)" in html
    assert "function isLargeTopologyRun()" in html
    assert "function largeTopologyNpqrBudgetError()" in html
    assert "大拓扑 NPQR 实时编译超过公共 API 时间预算。" in html
    assert "实时预算外" in html
    assert "大拓扑实时编译中：SABRE 优先。" in html
    assert "selected && selected.capacity > 20" in html
    assert "return { qasm: el.qasm.value, ...base };" in html
    assert "{ example: state.example, ...base }" not in html
    assert "Promise.allSettled" in html
    assert "startTraceReplay();" in html
    assert "REST API 可能仍在计算，请稍后再试。" in html

    for example in [
        "qft5",
        "ghz5",
        "qaoa5",
        "qft10",
        "qaoa10",
        "ghz10",
        "vqe10",
        "line_ghz30",
        "random30_d4",
        "line_ghz50",
        "ring_sparse50",
        "custom",
    ]:
        assert f'data-example="{example}"' in html
        assert f'<option value="{example}"' in html

    assert 'data-example-group="small"' in html
    assert 'data-example-group="large"' in html
    assert "const exampleGroups = {" in html
    assert "function syncExampleGroups()" in html
    assert 'topology: "tokyo"' not in html
    assert "topology: state.activeTopology.id" in html
    assert "topology: selected?.id || state.activeTopology.id" in html


def test_public_site_embeds_full_example_qasm_for_source_replay():
    html = _html()

    assert "完整示例由后端按名称加载" not in html
    for example in ["qft5", "ghz5", "qaoa5", "qft10", "qaoa10", "ghz10", "vqe10"]:
        expected = (Path("examples") / f"{example}.qasm").read_text(encoding="utf-8").rstrip()
        assert _embedded_example_qasm(html, example) == expected


def test_public_site_exposes_real_inputs_outputs_and_fixed_baseline():
    html = _html()

    for element_id in [
        "qasm-input",
        "qasm-replay",
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


def test_public_site_uses_truthful_compile_replay_state_machine():
    html = _html()

    assert "el.qasmDetails.open = Boolean(enabled);" in html
    assert "编译完成后自动显示 NPQR/SABRE 的编译后 QASM" in html
    assert 'compilePhase: "idle"' in html
    assert 'swapPhase: "none"' in html
    assert "state.step = (state.step + 1) % 4" not in html
    assert "function startAnimation()" not in html
    assert 'state.compilePhase = "requesting";' not in html
    assert 'phaseTimings: []' in html
    assert 'phaseBackend: "npqr"' in html
    assert 'state.phaseBackend = largeTopologyRun ? "sabre" : "npqr";' in html
    assert "state.phaseTimings = job.phases || [];" in html
    assert "backendPhaseReadout" in html
    assert 'state.compilePhase = "mapping";' in html
    assert 'state.compilePhase = "routing";' in html
    assert 'state.compilePhase = "output";' in html
    assert "function nextReplayIndex()" in html
    assert "function fastReplayWindow()" in html
    assert "function replaySpeedProfile()" in html
    assert "function bestRouteView(" in html
    assert "directEventSourceLine" in html
    assert "fallbackEventSourceLine" in html
    assert "state.trace[index]?.kind === \"swap\"" in html
    assert "state.traceIndex = nextReplayIndex();" in html
    assert "state.replaySignature !== signature" in html
    assert "state.timelineSignature !== signature" in html
    assert 'state.compilePhase = "parsing";' in html
    assert "swapDelay: 100" in html
    assert "swapDelay: 500" in html
    assert "gateDelay: 90" in html
    assert "快进 ${skipped} 个门" in html
    assert "mapping-focus" in html
    assert "is-dimmed" in html
    assert "is-review" in html
    assert "is-error" in html
    assert "is-done" in html
    assert "后台编译 ·" in html
    assert "完成 · 可视化结果" in html
    assert "animation: swap-label-shift 1s ease-in-out;" in html
    assert 'setRunVisual("replaying");' in html
    assert 'state.swapPhase = "before";' not in html
    assert 'state.swapPhase = nextEvent?.kind === "swap" ? "before" : "none";' in html
    assert "event?.mapping_before" in html
    assert "node-map-label is-swapping" not in html
    assert "mapText.classList.add(\"is-swapping\")" in html


def test_public_site_renders_dynamic_topology_and_trace_review():
    html = _html()

    assert '<svg class="topology-stage" id="tokyo-topology"' in html
    assert 'aria-label="IBM Tokyo 20Q topology replay"' in html
    assert 'preserveAspectRatio="xMidYMid meet"' in html
    assert 'id="topology-title"' in html
    assert "const topologyNodes = [" in html
    assert "const topologyEdges = [" in html
    assert "function selectTopologyForQubits" in html
    assert 'grid_5x6: generateGridTopology("grid_5x6", "Grid 5×6 30Q", 5, 6)' in html
    assert 'grid_5x10: generateGridTopology("grid_5x10", "Grid 5×10 50Q", 5, 10)' in html
    assert "当前前端演示上界为选定 50 比特结构。" in html
    assert 'max="50"' in html
    assert 'link.download = `compiled_${state.activeTopology.id}_${backend}.qasm`;' in html
    assert "traceEdge(event)" in html
    assert "traceNode(event)" in html
    assert "traceStats" in html
    assert "renderSourceReplay" in html
    assert "buildSourceReplayRows" in html
    assert "sourceGateLookup" in html
    assert 'event?.gate_index ?? event?.insertion_index' in html
    assert "event.next_gate_index ?? event.blocked_gate_index ?? event.insertion_index" in html
    assert 'id="trace-op"' in html
    assert 'id="route-view-npqr"' in html
    assert 'id="route-view-sabre"' in html
    assert 'id="mapping-grid"' in html
    assert 'id="route-timeline"' in html
    assert '<section class="timeline-panel" hidden>' in html
    assert 'aria-label="路由事件检查器"' not in html
    assert "mapping_after" in html
    assert "sample.path" not in html
    assert "Preview Tokyo edge" not in html


def test_public_site_uses_static_truthful_topology_replay():
    html = _html()

    for token in [
        "animateMotion",
        "topology-runner",
        "topology-packet",
        "chip-scan",
        "route-pulse",
        "marker-end",
        "marker-start",
        "gate-arrow",
        "swap-arrow",
    ]:
        assert token not in html

    assert "topology-focus-band" in html
    assert "topology-edge-label" in html


def test_public_site_keeps_mcp_as_folded_advanced_entry():
    html = _html()
    first_screen = _first_screen(html)

    assert '<aside class="service-panel" id="advanced" aria-label="REST API 和 MCP 服务接口" hidden>' in html
    assert "Service Interfaces" in html
    assert "MCP 面向工具客户端和自动化流程" in html
    assert "qcompiler-mcp-http" in html
    assert "src.server.mcp_app:app" in html
    assert "POST /mcp" in html
    assert "MCP" not in first_screen
    assert "高级入口" not in html

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
    assert 'aria-label="REST API 和 MCP 服务接口"' in html
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
