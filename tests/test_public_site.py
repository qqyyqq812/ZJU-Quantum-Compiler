"""Smoke tests for the public GitHub Pages entry point."""
from __future__ import annotations

from pathlib import Path


def test_public_site_uses_current_honest_positioning():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    assert "ZJU Quantum Compiler Playground" in html
    assert "AI experimental" in html
    assert "V14 P1 benchmark" in html
    assert "qcompiler compile examples/qft5.qasm" in html
    assert "assets/ibm-tokyo-topology.png" in html


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
