"""Check the public release tree for final course-project readiness."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
SITE_PATH = PROJECT_ROOT / "docs" / "index.html"
WORK_SPLIT_PATH = PROJECT_ROOT / "docs" / "plans" / "组员分工.md"
PROJECT_GUIDE_PATH = PROJECT_ROOT / "docs" / "项目说明.md"
PPTX_PATH = PROJECT_ROOT / "docs" / "slides" / "quantum-routing-algorithm-showcase-final.pptx"
MODEL_PATH = PROJECT_ROOT / "models" / "default" / "npqr-default.pt"
REST_APP = PROJECT_ROOT / "src" / "server" / "app.py"
MCP_APP = PROJECT_ROOT / "src" / "server" / "mcp_app.py"
EVIDENCE_MODULE = PROJECT_ROOT / "src" / "evidence.py"
PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_submission.py"
MATRIX_SCRIPT = PROJECT_ROOT / "scripts" / "experiment_algorithm_matrix.py"

FORBIDDEN_PUBLIC_PATTERNS = [
    r"\b" + "Stage" + r"\d+\b",
    "npqr_" + "stage",
    "npqr_" + "model_",
    "NEXT_" + "HAND" + "OFF",
    "HAND" + "OFF",
    "checkpoint_" + "ep",
    "wave2_" + "stage",
    "results/" + "npqr_",
]


@dataclass(frozen=True)
class ReadinessItem:
    category: str
    item: str
    status: str
    evidence: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _status(ok: bool) -> str:
    return "READY" if ok else "BLOCKED"


def _pptx_text(path: Path = PPTX_PATH) -> str:
    if not path.exists():
        return ""
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


def _has_forbidden_public_terms(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in FORBIDDEN_PUBLIC_PATTERNS)


def check_readiness() -> list[ReadinessItem]:
    readme = _read_text(README_PATH)
    site = _read_text(SITE_PATH)
    work_split = _read_text(WORK_SPLIT_PATH)
    project_guide = _read_text(PROJECT_GUIDE_PATH)
    rest_app = _read_text(REST_APP)
    mcp_app = _read_text(MCP_APP)
    evidence = _read_text(EVIDENCE_MODULE)
    package_script = _read_text(PACKAGE_SCRIPT)
    matrix_script = _read_text(MATRIX_SCRIPT)
    pptx_text = _pptx_text()
    public_docs = "\n".join([readme, work_split])

    return [
        ReadinessItem(
            "model",
            "Default NPQR model is present",
            _status(MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000),
            "models/default/npqr-default.pt",
        ),
        ReadinessItem(
            "api",
            "REST API defaults to NPQR",
            _status(
                'backend: Literal["npqr", "sabre"] = "npqr"' in rest_app
                and "models\" / \"default\" / \"npqr-default.pt" in rest_app
                and '"sabre_fallback": False' in rest_app
                and '@app.post("/api/compile"' in rest_app
                and "AI" + "Router" not in rest_app
            ),
            "src/server/app.py",
        ),
        ReadinessItem(
            "mcp",
            "HTTP MCP exposes final compiler tools",
            _status(
                'streamable_http_path="/mcp"' in mcp_app
                and "def compile_qasm(" in mcp_app
                and "def compile_npqr(" in mcp_app
                and "def get_algorithm_evidence(" in mcp_app
                and "get_npqr_" + "stage" not in mcp_app
            ),
            "src/server/mcp_app.py",
        ),
        ReadinessItem(
            "evidence",
            "Public evidence avoids internal experiment logs",
            _status(
                "npqr_public_algorithm_evidence_v1" in evidence
                and "algorithm_components" in evidence
                and "course_algorithm_mapping" in evidence
                and "representative_10_20_basic" in evidence
                and "scale_smoke_30_50_basic" in evidence
                and "npqr_beats_sabre_basic" in evidence
                and "npqr_" + "stage" not in evidence
            ),
            "src/evidence.py",
        ),
        ReadinessItem(
            "docs",
            "README explains install, API, MCP, and algorithm",
            _status(
                "# ZJU Quantum Compiler" in readme
                and "启动 REST API" in readme
                and "启动 MCP 服务" in readme
                and "算法说明" in readme
                and "课程算法概念对应" in readme
                and "models/default/npqr-default.pt" in readme
                and "docs/项目说明.md" in readme
                and "docs/plans/组员分工.md" in readme
                and "清" + "理版" not in readme
                and "按" + "要求" not in readme
                and "提示" + "词" not in readme
            ),
            "README.md",
        ),
        ReadinessItem(
            "docs",
            "Public docs hide internal experiment names",
            _status(not _has_forbidden_public_terms(public_docs + "\n" + project_guide)),
            "README.md + docs/plans/组员分工.md + docs/项目说明.md",
        ),
        ReadinessItem(
            "docs",
            "Team report guide uses course algorithm language",
            _status(
                "图问题" in work_split
                and "变治法" in work_split
                and "贪婪" in work_split
                and "时空权衡" in work_split
                and "神经网络" in work_split
                and "SABRE 是主要基线算法" in work_split
            ),
            "docs/plans/组员分工.md",
        ),
        ReadinessItem(
            "docs",
            "Detailed Chinese project guide exists",
            _status(
                "项目概述" in project_guide
                and "算法流程" in project_guide
                and "接口说明" in project_guide
                and "目录结构" in project_guide
                and "结果边界" in project_guide
                and "SABRE 是对比基线" in project_guide
            ),
            "docs/项目说明.md",
        ),
        ReadinessItem(
            "website",
            "Website opens from GitHub Pages with optional HTTPS REST",
            _status(
                "ZJU Quantum Compiler Console" in site
                and "量子电路编译控制台" in site
                and "Embedded review data active" in site
                and 'const PUBLIC_API_BASE = "";' in site
                and "?api=https://your-api.example" in site
                and "compileWithBestAvailableBackend(\"npqr\")" in site
                and "compileWithBestAvailableBackend(\"sabre\")" in site
                and "Service Interfaces" in site
                and "MCP Server" in site
                and "POST /mcp" in site
            ),
            "docs/index.html",
        ),
        ReadinessItem(
            "scripts",
            "Final package script is user-facing",
            _status(
                "public_algorithm_evidence.json" in package_script
                and "algorithm_summary.md" in package_script
                and "npqr_" + "stage" not in package_script
                and "St" + "age" not in package_script
            ),
            "scripts/package_submission.py",
        ),
        ReadinessItem(
            "scripts",
            "Algorithm matrix remains reproducible",
            _status(
                "def run_matrix" in matrix_script
                and "MatrixRow" in matrix_script
                and "--json" in matrix_script
                and "--csv" in matrix_script
            ),
            "scripts/experiment_algorithm_matrix.py",
        ),
        ReadinessItem(
            "slides",
            "Presentation material exists",
            _status(PPTX_PATH.exists() and ("NPQR" in pptx_text or PPTX_PATH.stat().st_size > 0)),
            "docs/slides/quantum-routing-algorithm-showcase-final.pptx",
        ),
    ]


def render_markdown(items: list[ReadinessItem]) -> str:
    lines = [
        "# Public release readiness",
        "",
        "| category | item | status | evidence |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(f"| {item.category} | {item.item} | {item.status} | `{item.evidence}` |")
    lines.append("")
    blocked = [item for item in items if item.status != "READY"]
    if blocked:
        lines.append(f"Blocked items: {len(blocked)}")
    else:
        lines.append("All public release checks are ready.")
    return "\n".join(lines) + "\n"


def main() -> None:
    print(render_markdown(check_readiness()))


if __name__ == "__main__":
    main()
