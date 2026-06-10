"""Build a small local review package from the public release files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "results" / "submission_package"

PACKAGE_MANIFEST = [
    ("README.md", "README.md"),
    ("docs/index.html", "website/index.html"),
    ("docs/plans/2026-06-05-mcp-work-split.md", "report-work-split.md"),
    (
        "docs/slides/quantum-routing-algorithm-showcase-final.pptx",
        "slides/quantum-routing-algorithm-showcase-final.pptx",
    ),
    ("examples/qft5.qasm", "examples/qft5.qasm"),
    ("examples/ghz5.qasm", "examples/ghz5.qasm"),
    ("examples/qaoa5.qasm", "examples/qaoa5.qasm"),
    ("results/submission_package/readiness.md", "readiness.md"),
    ("results/submission_package/algorithm_matrix.json", "algorithm_matrix.json"),
    ("results/submission_package/public_algorithm_evidence.json", "public_algorithm_evidence.json"),
    ("results/submission_package/algorithm_summary.md", "algorithm_summary.md"),
]


def _run(command: list[str], output_path: Path | None = None) -> None:
    if output_path is None:
        subprocess.check_call(command, cwd=PROJECT_ROOT)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        subprocess.check_call(command, cwd=PROJECT_ROOT, stdout=handle)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"required package source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_algorithm_summary(output_dir: Path) -> None:
    summary = """# NPQR algorithm summary

NPQR is a neural-assisted quantum routing workflow. It uses a checked-in neural
policy to score SWAP actions, then combines that model with initial mapping
selection, bounded beam search, trigger-based pruning, suffix repair, and trace
replay verification.

SABRE is the baseline. It is used for comparison and explanation, not as the
self-developed algorithm and not as an NPQR fallback.

Course concepts used in the report:

- Graph modeling: the hardware topology is a coupling graph.
- Transform-and-conquer: routing becomes graph-constrained mapping and search.
- Greedy heuristics: front-gate distance helps score local moves.
- Decrease-and-conquer: each executed gate reduces the remaining task.
- Time-space tradeoff: distance matrices and candidate beams reduce recomputation.
- Iterative improvement: swaps and suffix repair improve the route step by step.
- Search pruning: beam width, trigger rules, and action limits bound the search.
- Approximation: the algorithm seeks high-quality feasible routes in bounded time.
- Neural network inference: the policy model supplies learned action preferences.
"""
    (output_dir / "algorithm_summary.md").write_text(summary, encoding="utf-8")


def build_submission_package(output_dir: Path = PACKAGE_DIR) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "scripts/check_submission_readiness.py"], output_dir / "readiness.md")
    _run(
        [sys.executable, "scripts/experiment_algorithm_matrix.py", "--quick", "--json"],
        output_dir / "algorithm_matrix.json",
    )
    _run(
        [
            sys.executable,
            "scripts/generate_npqr_evidence_manifest.py",
            "--output",
            str(output_dir / "public_algorithm_evidence.json"),
            "--no-docs-copy",
        ]
    )

    for source, destination in PACKAGE_MANIFEST:
        if source.startswith("results/submission_package/"):
            continue
        _copy_file(PROJECT_ROOT / source, output_dir / destination)
    _write_algorithm_summary(output_dir)

    manifest_lines = [
        "# Submission package manifest",
        "",
        f"Generated at: `{output_dir}`",
        "",
        "| Source | Package path |",
        "| --- | --- |",
    ]
    for source, destination in PACKAGE_MANIFEST:
        manifest_lines.append(f"| `{source}` | `{destination}` |")
    manifest_lines.extend(
        [
            "",
            "This package is generated locally by `python scripts/package_submission.py`.",
            "Generated files under `results/submission_package/` are not committed.",
        ]
    )
    (output_dir / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_DIR,
        help="Package output directory. Defaults to results/submission_package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = build_submission_package(args.output)
    print(f"submission package generated: {output_dir}")


if __name__ == "__main__":
    main()
