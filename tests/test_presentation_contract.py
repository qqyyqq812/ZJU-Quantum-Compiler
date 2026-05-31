"""Contract tests for presentation evidence and report consistency."""
from __future__ import annotations

import re
import zipfile
from html import unescape
from pathlib import Path


def _pptx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as deck:
        slide_names = sorted(
            [
                name
                for name in deck.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ],
            key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)),
        )
        parts: list[str] = []
        for slide_name in slide_names:
            xml = deck.read(slide_name).decode("utf-8", errors="ignore")
            parts.extend(unescape(text) for text in re.findall(r"<a:t>(.*?)</a:t>", xml))
        return "\n".join(parts)


def test_pptx_uses_current_bounded_search_negative_sample():
    text = _pptx_text(Path("docs/slides/quantum-routing-algorithm-showcase-draft.pptx"))

    assert "qft10: 52 SWAP" in text
    assert "qft10: 59 SWAP" not in text
    assert "lookahead 29" in text
    assert "bounded search 负样本" in text


def test_report_and_board_use_current_bounded_search_result():
    report = Path("docs/technical/项目报告.md").read_text(encoding="utf-8")
    board = Path("docs/technical/02_宏观任务看板.md").read_text(encoding="utf-8")

    assert "`qft10` 上需要 52 个 SWAP" in report
    assert "`qft10` 上需要 59 个 SWAP" not in report
    assert "qcompiler compile examples/qft5.qasm --topology tokyo --backend sabre --heuristic lookahead" in report
    assert "`qft10: lookahead 29 vs bounded 52`" in board
