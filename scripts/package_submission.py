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
    ("docs/项目说明.md", "项目说明.md"),
    ("docs/playground-user-guide.md", "playground-user-guide.md"),
    ("docs/ai-collaboration.md", "ai-collaboration.md"),
    ("docs/final-closure-report.md", "final-closure-report.md"),
    ("docs/report_latex/main.pdf", "report/npqr-technical-report.pdf"),
    ("examples/qft5.qasm", "examples/qft5.qasm"),
    ("examples/ghz5.qasm", "examples/ghz5.qasm"),
    ("examples/qaoa5.qasm", "examples/qaoa5.qasm"),
    ("examples/line_ghz30.qasm", "examples/line_ghz30.qasm"),
    ("examples/random30_d4.qasm", "examples/random30_d4.qasm"),
    ("examples/line_ghz50.qasm", "examples/line_ghz50.qasm"),
    ("examples/ring_sparse50.qasm", "examples/ring_sparse50.qasm"),
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
    summary = """# NPQR 算法摘要

NPQR 是一个神经辅助量子路由流程。它使用默认神经网络模型对 SWAP 动作进行
评分，并结合初始映射选择、有界束搜索、触发式剪枝、后缀修复和轨迹复放验证。

SABRE basic 是固定对照基线。项目使用它进行指标对比和算法说明，NPQR 与
SABRE basic 的结果在相同线路、拓扑和指标口径下比较。

公开证据使用 SABRE basic 作为主要质量基线。NPQR 在代表性 10/20 比特电路上
10/10 完成，并且 10/10 的 SWAP 数低于 SABRE basic。30/50 比特扩展测试中，
NPQR 4/4 完成，并且 4/4 优于 SABRE basic。5/10 比特示例适合网页快速体验，
30/50 比特示例需要部署后端支持真实编译。

方法要点包括：

- 图建模：硬件拓扑是耦合图。
- 问题转化：路由被转化为图约束下的映射和搜索。
- 局部评分：前沿门距离可辅助评价局部动作。
- 递进执行：每执行一个门都会减少剩余任务。
- 时空权衡：距离矩阵和候选路线减少重复计算。
- 迭代改进：SWAP 和后缀修复逐步改进路线。
- 搜索剪枝：束宽、触发规则和动作限制控制搜索规模。
- 近似求解：算法在有限时间内寻找高质量可行解。
- 神经网络推理：模型提供学习得到的动作偏好。
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
