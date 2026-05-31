"""Regression tests for teacher-facing local demo scripts."""
from __future__ import annotations

from pathlib import Path


def test_public_demo_script_documents_heuristic_lab_evidence():
    script = Path("run_public_demo.sh").read_text(encoding="utf-8")

    assert "[1/6] qcompiler info" in script
    assert "[6/6] SabreSwap heuristic lab 证据" in script
    assert "--heuristic lookahead" in script
    assert "scripts/experiment_sabre_trials.py" in script
    assert "scripts/experiment_bounded_search.py" in script
    assert "06_sabre_heuristic_lab.md" in script
    assert "07_bounded_search_negative_sample.md" in script
    assert "uvicorn src.server.app:app --reload --port 8765" in script
    assert "python -m http.server 8766 --directory docs" in script
    assert "bounded search v1 是负样本" in script


def test_teacher_eval_index_uses_current_project_positioning():
    script = Path("run_teacher_eval.sh").read_text(encoding="utf-8")

    assert "GitHub Pages heuristic lab" in script
    assert "06_sabre_heuristic_lab.md" in script
    assert "07_bounded_search_negative_sample.md" in script
    assert "SABRE / LightSABRE is the stable practical baseline." in script
    assert "The public website is now a heuristic lab" in script
    assert "lookahead + seed=42 + trials=1" in script
    assert "Bounded search v1 is a negative sample" in script
    assert "have not beaten SABRE" in script


def test_teacher_eval_generated_index_lists_all_demo_files():
    script = Path("run_teacher_eval.sh").read_text(encoding="utf-8")

    expected_files = [
        "01_qcompiler_info.txt",
        "02_sabre_compile.txt",
        "03_qcompiler_eval.txt",
        "04_mqt_5q_demo.md",
        "04_mqt_5q_demo.json",
        "05_ibm_tokyo_topology.png",
        "06_sabre_heuristic_lab.md",
        "07_bounded_search_negative_sample.md",
    ]

    for file_name in expected_files:
        assert file_name in script
