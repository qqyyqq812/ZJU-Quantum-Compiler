"""Check the public release tree for reviewer-friendly submission readiness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
SITE_PATH = PROJECT_ROOT / "docs" / "index.html"
PROJECT_GUIDE_PATH = PROJECT_ROOT / "docs" / "项目说明.md"
USER_GUIDE_PATH = PROJECT_ROOT / "docs" / "playground-user-guide.md"
AI_DISCLOSURE_PATH = PROJECT_ROOT / "docs" / "ai-collaboration.md"
FINAL_CLOSURE_PATH = PROJECT_ROOT / "docs" / "final-closure-report.md"
REPORT_SOURCE_PATHS = [
    PROJECT_ROOT / "docs" / "report_latex" / "main.tex",
    *sorted((PROJECT_ROOT / "docs" / "report_latex" / "sections").glob("*.tex")),
    *sorted((PROJECT_ROOT / "docs" / "report_latex" / "tables").glob("*.tex")),
]
MODEL_PATH = PROJECT_ROOT / "models" / "default" / "npqr-default.pt"
REST_APP = PROJECT_ROOT / "src" / "server" / "app.py"
MCP_APP = PROJECT_ROOT / "src" / "server" / "mcp_app.py"
EVIDENCE_MODULE = PROJECT_ROOT / "src" / "evidence.py"
PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_submission.py"
MATRIX_SCRIPT = PROJECT_ROOT / "scripts" / "experiment_algorithm_matrix.py"

ALLOWED_ASSIGNMENT_PHRASE = "量子信息基础大作业"

FORBIDDEN_PUBLIC_PATTERNS = [
    r"\bStage\d+\b",
    "npqr_stage",
    "npqr_model_",
    "NEXT_HANDOFF",
    "HANDOFF",
    "checkpoint_ep",
    "wave2_stage",
    "results/npqr_",
    "组员分工",
    r"(?<!量子信息基础)大作业",
    "算法设计与智能计算",
    "算法大作业",
    "课程项目",
    "课程报告",
    "课程作业",
    "课程代表",
    "课程算法",
    "智能计算",
    "人工PR",
    "course-project",
    "course project",
    "course assignment",
    "course submission",
    "80/100Q",
    "80Q",
    "100Q",
    "hidden fallback",
    "non-claim",
    "boundary rows",
    "审计",
    "审阅",
    "自审",
    "不声明",
    "不能保证",
    "老师可以",
    "便于检查",
]

LARGE_EXAMPLE_FILES = [
    PROJECT_ROOT / "examples" / "line_ghz30.qasm",
    PROJECT_ROOT / "examples" / "random30_d4.qasm",
    PROJECT_ROOT / "examples" / "line_ghz50.qasm",
    PROJECT_ROOT / "examples" / "ring_sparse50.qasm",
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


def _has_forbidden_public_terms(text: str) -> bool:
    normalized = text.replace(ALLOWED_ASSIGNMENT_PHRASE, "")
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in FORBIDDEN_PUBLIC_PATTERNS)


def _report_source_text() -> str:
    return "\n".join(_read_text(path) for path in REPORT_SOURCE_PATHS if path.exists())


def check_readiness() -> list[ReadinessItem]:
    readme = _read_text(README_PATH)
    site = _read_text(SITE_PATH)
    project_guide = _read_text(PROJECT_GUIDE_PATH)
    user_guide = _read_text(USER_GUIDE_PATH)
    ai_disclosure = _read_text(AI_DISCLOSURE_PATH)
    final_closure = _read_text(FINAL_CLOSURE_PATH)
    rest_app = _read_text(REST_APP)
    mcp_app = _read_text(MCP_APP)
    evidence = _read_text(EVIDENCE_MODULE)
    package_script = _read_text(PACKAGE_SCRIPT)
    matrix_script = _read_text(MATRIX_SCRIPT)
    report_source = _report_source_text()
    public_docs = "\n".join(
        [readme, project_guide, user_guide, ai_disclosure, final_closure, report_source]
    )

    return [
        ReadinessItem(
            "model",
            "Default NPQR model is present",
            _status(MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000),
            "models/default/npqr-default.pt",
        ),
        ReadinessItem(
            "api",
            "REST API defaults to NPQR and exposes large examples",
            _status(
                'backend: Literal["npqr", "sabre"] = "npqr"' in rest_app
                and "models\" / \"default\" / \"npqr-default.pt" in rest_app
                and '"sabre_fallback": False' in rest_app
                and '@app.post("/api/compile"' in rest_app
                and '"line_ghz50"' in rest_app
                and '"grid_5x10"' in rest_app
                and '[[int(a), int(b)] for a, b in coupling_map.get_edges()]' in rest_app
                and "AIRouter" not in rest_app
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
                and "get_npqr_stage" not in mcp_app
            ),
            "src/server/mcp_app.py",
        ),
        ReadinessItem(
            "evidence",
            "Public evidence avoids internal experiment logs",
            _status(
                "npqr_public_algorithm_evidence_v1" in evidence
                and "algorithm_components" in evidence
                and "concept_mapping" in evidence
                and "representative_10_20_basic" in evidence
                and "scale_smoke_30_50_basic" in evidence
                and "npqr_beats_sabre_basic" in evidence
                and "npqr_stage" not in evidence
            ),
            "src/evidence.py",
        ),
        ReadinessItem(
            "docs",
            "README provides reviewer quick entry",
            _status(
                "# ZJU Quantum Compiler" in readme
                and "语言 / Language" in readme
                and "## 快速入口" in readme
                and "普通体验" in readme
                and "本地部署" in readme
                and "工具调用" in readme
                and "English version" in readme
                and "GitHub Pages" in readme
                and "REST API" in readme
                and "MCP" in readme
                and "30Q 和 50Q 示例属于扩展规模，需要部署" in readme
                and "docs/项目说明.md" in readme
                and "docs/playground-user-guide.md" in readme
                and "docs/ai-collaboration.md" in readme
                and "examples/line_ghz50.qasm" in readme
            ),
            "README.md",
        ),
        ReadinessItem(
            "docs",
            "Playground guide explains the three-minute review path",
            _status(
                "Three-minute review" in user_guide
                and "ghz5" in user_guide
                and "qft5" in user_guide
                and "qaoa5" in user_guide
                and "extension-scale examples" in user_guide
                and "GitHub Pages" in user_guide
                and "Real compiler backend" in user_guide
                and "not required for normal browser use" in user_guide
            ),
            "docs/playground-user-guide.md",
        ),
        ReadinessItem(
            "docs",
            "Public docs hide internal process terms",
            _status(not _has_forbidden_public_terms(public_docs)),
            "README.md + docs/*.md + report sources",
        ),
        ReadinessItem(
            "docs",
            "AI collaboration disclosure is concise and reviewable",
            _status(
                "AI collaboration disclosure" in ai_disclosure
                and "Human-controlled decisions" in ai_disclosure
                and "Verification" in ai_disclosure
                and "prompt" not in ai_disclosure.lower()
            ),
            "docs/ai-collaboration.md",
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
                and "方法要点" in project_guide
                and "SABRE 是固定" in project_guide
            ),
            "docs/项目说明.md",
        ),
        ReadinessItem(
            "examples",
            "Checked-in 30/50Q examples are public",
            _status(
                all(path.exists() for path in LARGE_EXAMPLE_FILES)
                and "qreg q[30];" in _read_text(PROJECT_ROOT / "examples" / "line_ghz30.qasm")
                and "qreg q[50];" in _read_text(PROJECT_ROOT / "examples" / "line_ghz50.qasm")
            ),
            "examples/",
        ),
        ReadinessItem(
            "website",
            "Website opens from GitHub Pages with REST and MCP descriptions",
            _status(
                "量子编译实验台" in site
                and 'const PUBLIC_API_BASE = "http://1.95.70.10";' in site
                and "createCompileJob" in site
                and "pollCompileJob" in site
                and "random30_d4" in site
                and 'href="playground-user-guide.md"' in site
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
                and "docs/ai-collaboration.md" in package_script
                and "examples/line_ghz50.qasm" in package_script
                and "npqr_stage" not in package_script
                and "Stage" not in package_script
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
