"""Smoke tests for the public GitHub Pages entry point."""
from __future__ import annotations

from pathlib import Path


def _html() -> str:
    return Path("docs/index.html").read_text(encoding="utf-8")


def test_public_site_is_chinese_upload_first_compiler_tool():
    html = _html()

    assert "<title>ZJU 量子电路编译器</title>" in html
    assert "量子电路编译工作台" in html
    assert "上传编译" in html
    assert "GitHub Pages 负责界面，本机 API 负责真实编译。" in html
    assert "编译上传电路" in html
    assert "尚未选择文件。浏览器只读取文本内容，不上传到 GitHub。" in html
    assert "Compare SabreSwap heuristics" not in html
    assert "Quantum routing heuristic lab." not in html
    assert "NPQR evidence dashboard" not in html


def test_public_site_wires_real_file_upload_to_inline_qasm_compile():
    html = _html()

    assert 'id="circuit-file"' in html
    assert 'type="file"' in html
    assert 'accept=".qasm,.txt,text/plain"' in html
    assert "async function readSelectedFile(file)" in html
    assert "const text = await file.text();" in html
    assert "els.qasmInput.value = text;" in html
    assert "async function compileUploadedCircuit()" in html
    assert "const qasm = currentQasm();" in html
    assert "payload = {" in html
    assert "qasm," in html
    assert 'backend: "sabre"' in html
    assert 'topology: "tokyo"' in html
    assert "heuristic" in html
    assert "fetch(`${apiBase}/api/compile`" in html


def test_public_site_exposes_compiled_qasm_and_downloads():
    html = _html()

    assert 'id="compiled-qasm"' in html
    assert "路由后的 OpenQASM" in html
    assert "API 返回的 compiled_qasm" in html
    assert "els.compiledQasm.value = data.compiled_qasm || \"\";" in html
    assert 'id="download-qasm"' in html
    assert 'id="download-json"' in html
    assert "compiled-tokyo-sabre.qasm" in html
    assert "quantum-compile-result.json" in html
    assert "lastResult = {" in html


def test_public_site_guides_local_api_recovery_in_chinese():
    html = _html()

    assert "本地 API 未连接" in html
    assert "GitHub Pages 不能单独编译电路" in html
    assert "uvicorn src.server.app:app --reload --port 8765" in html
    assert "复制启动命令" in html
    assert "页面请求 <code>http://localhost:8765/api/compile</code>。" in html
    assert "GitHub Pages 不能运行 Python，所以必须先启动本地服务。" in html


def test_public_site_keeps_required_domain_terms_and_honest_boundary():
    html = _html()

    for term in ["OpenQASM 2", "SABRE", "SWAP", "Qiskit", "API", "IBM Tokyo 20Q"]:
        assert term in html

    assert "AI router</strong> 只作为研究证据保留，公开默认路径仍是 SABRE。" in html
    assert "seed=42" in html
    assert "trials=1" in html


def test_public_site_preserves_accessibility_and_responsive_basics():
    html = _html()

    assert 'lang="zh-CN"' in html
    assert 'aria-label="量子电路编译器"' in html
    assert 'aria-label="编译结果摘要"' in html
    assert 'alt="IBM Tokyo 20Q coupling topology"' in html
    assert "overflow-x: clip;" in html
    assert "@media (max-width: 1180px)" in html
    assert "@media (max-width: 680px)" in html


def test_public_site_uses_compact_workbench_not_hero_page():
    html = _html()

    assert 'class="workbench"' in html
    assert 'class="bench-toolbar"' in html
    assert 'class="bench-body"' in html
    assert 'class="input-pane"' in html
    assert 'class="result-pane"' in html
    assert "grid-template-columns: minmax(320px, 0.92fr) minmax(360px, 1.08fr);" in html
    assert 'class="hero"' not in html
    assert "font-size: clamp(36px, 6vw, 70px)" not in html


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
