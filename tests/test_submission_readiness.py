"""Contract tests for final public release readiness checks."""

from __future__ import annotations

from pathlib import Path

from scripts.check_submission_readiness import check_readiness, render_markdown


def test_submission_readiness_marks_public_release_ready():
    items = check_readiness()
    by_item = {item.item: item for item in items}

    expected_ready = [
        "Default NPQR model is present",
        "REST API defaults to NPQR",
        "HTTP MCP exposes final compiler tools",
        "Public evidence avoids internal experiment logs",
        "README explains install, API, MCP, and algorithm",
        "Public docs hide internal experiment names",
        "Team report guide uses course algorithm language",
        "Detailed Chinese project guide exists",
        "Website calls REST API without changing the visual shell",
        "Final package script is user-facing",
        "Algorithm matrix remains reproducible",
        "Presentation material exists",
    ]
    for item in expected_ready:
        assert by_item[item].status == "READY"


def test_submission_readiness_markdown_is_reviewable():
    markdown = render_markdown(check_readiness())

    assert "| category | item | status | evidence |" in markdown
    assert "Default NPQR model is present" in markdown
    assert "REST API defaults to NPQR" in markdown
    assert "HTTP MCP exposes final compiler tools" in markdown
    assert "All public release checks are ready." in markdown


def test_final_submission_smoke_script_runs_required_gates():
    script = Path("scripts/check_final_submission.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "git diff --check" in script
    assert "scripts/check_submission_readiness.py" in script
    assert "pytest -s -q" in script
