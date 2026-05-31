"""Smoke tests for the public GitHub Pages entry point."""
from __future__ import annotations

from pathlib import Path


def test_public_site_uses_current_honest_positioning():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert "ZJU Quantum Routing Heuristic Lab" in html
    assert "Compare SabreSwap heuristics" in html
    assert "Run checked-in or custom OpenQASM through the local API." in html
    assert "AI-router history is kept as workload and failure-analysis evidence only." in html
    assert "LightSABRE modifications" in html
    assert "seed=42</code> and <code>trials=1</code>" in html
    assert "qcompiler compile examples/qft5.qasm" in html
    assert "assets/ibm-tokyo-topology.png" in html


def test_public_site_wires_live_sabre_heuristic_api():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert 'data-heuristic="basic"' in html
    assert 'data-heuristic="lookahead"' in html
    assert 'data-heuristic="decay"' in html
    assert "heuristic: selectedHeuristic" in html
    assert "now accepts a <code>heuristic</code> parameter" in html
    assert "intentionally fixed to <code>trials=1</code>" in html
    assert "Live API note: this response runs the selected SABRE heuristic." in html


def test_public_site_documents_multitrial_boundary():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert "Add a documented multi-trial mode only after adding matching expected rows." in html
    assert "Keep bounded search as failure-analysis evidence, not a website algorithm." in html
    assert "seed=42, trials=1" in html
    assert "Multi-trial mode can be stronger on some cases" in html


def test_public_site_reports_static_live_consistency():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert 'id="live-status"' in html
    assert 'id="live-swaps"' in html
    assert 'id="live-depth"' in html
    assert 'id="live-match"' in html
    assert "setLiveSummary" in html
    assert "SWAP</strong> counts inserted routing swaps" in html
    assert "Depth</strong> is the routed" in html
    assert "Match</strong> says whether the live API" in html
    assert "formatLiveCheck" in html
    assert "expected_swaps:" in html
    assert "live_swaps:" in html
    assert "static_live_match:" in html
    assert "reproducibility warning" in html


def test_public_site_preserves_tool_first_structure_order():
    html = Path("docs/index.html").read_text(encoding="utf-8")
    ordered_tokens = [
        'id="playground"',
        "Quantum routing heuristic lab.",
        "Live checks",
        "Compare SabreSwap heuristics",
        "Example circuits",
        'data-heuristic="basic"',
        'id="custom-qasm"',
        'id="live-status"',
        'id="run-local"',
        'id="benchmarks"',
        'id="local-api"',
        'id="figures"',
    ]

    positions = [html.index(token) for token in ordered_tokens]
    assert positions == sorted(positions)


def test_public_site_preserves_visual_and_accessibility_basics():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert 'aria-label="Current project status"' in html
    assert 'aria-label="Interactive compiler playground"' in html
    assert 'aria-label="Live API result summary"' in html
    assert 'alt="IBM Tokyo 20Q coupling topology"' in html
    assert 'alt="Animated quantum circuit route demo on IBM Tokyo topology"' in html
    assert "Compare basic, lookahead, and decay with seed=42 and trials=1." in html
    assert "Qiskit 2.3.1 implements SabreSwap with LightSABRE modifications" in html
    assert "qcompiler compile examples/qft5.qasm --topology tokyo --backend sabre --heuristic lookahead" in html


def test_public_site_supports_custom_qasm_input():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert 'id="custom-qasm"' in html
    assert 'id="fill-ghz-qasm"' in html
    assert 'id="use-custom-qasm"' in html
    assert 'id="clear-custom-qasm"' in html
    assert "maxInlineQasmChars = 8000" in html
    assert "miniGhzQasm" in html
    assert "fillMiniGhzInput" in html
    assert "payload.qasm = qasm" in html
    assert "custom input has no static expected row" in html


def test_public_site_exposes_all_checked_in_api_examples():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    for sample in ["qft5", "qaoa5", "ghz5", "qft10", "qaoa10", "ghz10", "vqe10"]:
        assert f'data-sample="{sample}"' in html
        assert f"{sample}: {{" in html


def test_public_site_does_not_reintroduce_old_overclaims():
    html = Path("docs/index.html").read_text(encoding="utf-8")
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
