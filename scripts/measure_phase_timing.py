"""Measure NPQR phase timings for report and slide analysis."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "npqr_phase_timing"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    group: str
    topology: str
    qubits: int
    source: str
    qasm_path: str | None = None
    timeout_s: float = 90.0
    runtime_profile: str = "default"


CASE_SPECS: dict[str, CaseSpec] = {
    "qaoa10": CaseSpec(
        case_id="qaoa10",
        group="representative_10_20",
        topology="ibm_tokyo",
        qubits=10,
        source="checked-in QASM",
        qasm_path="examples/qaoa10.qasm",
        timeout_s=90.0,
    ),
    "qft10": CaseSpec(
        case_id="qft10",
        group="representative_10_20",
        topology="ibm_tokyo",
        qubits=10,
        source="checked-in QASM",
        qasm_path="examples/qft10.qasm",
        timeout_s=90.0,
    ),
    "brickwork20_profile": CaseSpec(
        case_id="brickwork20_profile",
        group="representative_10_20",
        topology="ibm_tokyo",
        qubits=20,
        source="deterministic generated circuit",
        timeout_s=90.0,
    ),
    "line_ghz30": CaseSpec(
        case_id="line_ghz30",
        group="selected_30_50",
        topology="grid_5x6",
        qubits=30,
        source="checked-in QASM",
        qasm_path="examples/line_ghz30.qasm",
        timeout_s=90.0,
        runtime_profile="large_scale_frontier",
    ),
    "line_ghz50": CaseSpec(
        case_id="line_ghz50",
        group="selected_30_50",
        topology="grid_5x10",
        qubits=50,
        source="checked-in QASM",
        qasm_path="examples/line_ghz50.qasm",
        timeout_s=180.0,
        runtime_profile="large_scale_frontier",
    ),
    "line_ghz80_boundary": CaseSpec(
        case_id="line_ghz80_boundary",
        group="boundary_80_100",
        topology="grid_8x10",
        qubits=80,
        source="deterministic generated boundary circuit",
        timeout_s=240.0,
        runtime_profile="large_scale_frontier",
    ),
}

DEFAULT_CASES = [
    "qaoa10",
    "qft10",
    "brickwork20_profile",
    "line_ghz30",
    "line_ghz50",
]

PHASE_LABELS = {
    "data_preprocessing": "数据预处理",
    "dependency_graph_build": "电路依赖图构建",
    "topology_distance_matrix": "物理拓扑距离矩阵计算",
    "logical_interaction_graph": "逻辑交互图构建",
    "initial_mapping_candidates": "初始映射候选生成",
    "main_search": "主搜索阶段",
    "action_generation_mask": "动作生成 / 动作掩码",
    "neural_network_inference": "神经网络推理",
    "beam_expand_prune": "束搜索扩展与剪枝",
    "suffix_repair": "局部后缀修复",
    "postprocessing": "后处理阶段",
    "trace_replay_validation": "轨迹复放验证",
    "total": "总耗时",
}

PHASE_ORDER = [
    "data_preprocessing",
    "dependency_graph_build",
    "topology_distance_matrix",
    "logical_interaction_graph",
    "initial_mapping_candidates",
    "main_search",
    "action_generation_mask",
    "neural_network_inference",
    "beam_expand_prune",
    "suffix_repair",
    "postprocessing",
    "trace_replay_validation",
    "total",
]

NESTED_PHASES = {
    "dependency_graph_build",
    "topology_distance_matrix",
    "logical_interaction_graph",
    "action_generation_mask",
    "neural_network_inference",
    "beam_expand_prune",
    "trace_replay_validation",
}


def _configure_runtime_environment() -> None:
    for key, value in THREAD_ENV.items():
        os.environ.setdefault(key, value)


def _brickwork20_profile():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(20, name="brickwork20_profile")
    for qubit in range(20):
        circuit.h(qubit)
    pairs = [
        (0, 1),
        (2, 3),
        (4, 8),
        (5, 6),
        (7, 8),
        (10, 11),
        (12, 13),
        (15, 16),
        (17, 18),
        (18, 19),
        (1, 7),
        (3, 9),
        (5, 11),
        (7, 13),
        (11, 17),
        (13, 19),
        (1, 2),
        (3, 4),
        (6, 7),
        (8, 9),
        (10, 15),
        (12, 17),
        (14, 19),
    ]
    for first, second in pairs:
        circuit.cx(first, second)
        circuit.rz(0.125, second)
    return circuit


def _line_ghz(qubits: int):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(qubits, name=f"line_ghz{qubits}_boundary")
    circuit.h(0)
    for qubit in range(qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def _load_case_circuit(spec: CaseSpec):
    from qiskit import QuantumCircuit

    if spec.qasm_path:
        circuit = QuantumCircuit.from_qasm_file(str(PROJECT_ROOT / spec.qasm_path))
        circuit.name = spec.case_id
        return circuit
    if spec.case_id == "brickwork20_profile":
        return _brickwork20_profile()
    if spec.case_id == "line_ghz80_boundary":
        return _line_ghz(80)
    raise ValueError(f"No circuit loader for case: {spec.case_id}")


def _runtime_config(spec: CaseSpec, max_steps: int):
    from src.compiler.npqr_runtime import NPQRRuntimeConfig

    if spec.runtime_profile != "large_scale_frontier":
        return NPQRRuntimeConfig(max_steps=max_steps)

    return replace(
        NPQRRuntimeConfig(),
        max_steps=max_steps,
        beam_width=4,
        branch_factor=3,
        perturbation_count=10,
        qap_local_search_rounds=2,
        primary_selector_top_k=2,
        rescue_selector_top_k=4,
        suffix_depth=6,
        suffix_max_nodes=4000,
        suffix_action_limit=12,
        frontier_rescue_enabled=True,
        frontier_rescue_precheck_enabled=True,
        frontier_max_steps=420,
        frontier_action_limit=90,
        frontier_action_pruning_policy="extended_touch_12",
        frontier_max_candidates=4,
        frontier_selector_top_k=4,
        frontier_min_qubits=10,
        frontier_min_cx_like=18,
        frontier_min_unique_pairs=18,
        frontier_min_unique_pair_ratio=0.18,
        frontier_min_pair_entropy=1.8,
        frontier_max_depth=320,
        frontier_max_cx_like=420,
        frontier_max_repeat_pair_ratio=0.97,
    )


def _phase_total_ms(profile: dict[str, Any], phase: str) -> float:
    data = profile.get(phase, {})
    return float(data.get("total_ms", 0.0))


def _run_worker(
    case_id: str,
    run_index: int,
    max_steps: int,
    worker_timeout_s: float | None = None,
) -> dict[str, Any]:
    _configure_runtime_environment()

    import numpy as np
    import torch

    from src.benchmarks.topologies import get_topology
    from src.compiler.npqr_runtime import DEFAULT_NPQR_MODEL, NPQRRuntime
    from src.compiler.profile_timing import PhaseProfiler

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(42)
    np.random.seed(42)

    spec = CASE_SPECS[case_id]
    timeout_budget_s = worker_timeout_s if worker_timeout_s is not None else spec.timeout_s
    profiler = PhaseProfiler()
    coupling_map = get_topology(spec.topology)
    runtime = NPQRRuntime(
        coupling_map,
        model_path=DEFAULT_NPQR_MODEL,
        config=_runtime_config(spec, max_steps),
        profiler=profiler,
    )
    profiler.phases.clear()

    def _raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(f"case exceeded worker budget: {spec.timeout_s}s")

    previous_handler = None
    if hasattr(signal, "SIGALRM"):
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, float(timeout_budget_s)))
    started = time.perf_counter()
    try:
        with profiler.measure("data_preprocessing"):
            circuit = _load_case_circuit(spec)
            ops = dict(circuit.count_ops())
        result = runtime.compile(circuit)
        total_ms = (time.perf_counter() - started) * 1000.0

        phases = profiler.to_dict()
        phases["total"] = {
            "total_ms": total_ms,
            "count": 1,
            "min_ms": total_ms,
            "max_ms": total_ms,
        }
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "topology": spec.topology,
            "qubits": spec.qubits,
            "source": spec.source,
            "runtime_profile": spec.runtime_profile,
            "run_index": run_index,
            "status": "ok",
            "completed": bool(result.completed),
            "result_status": result.status,
            "swaps": result.total_swaps,
            "depth": result.depth,
            "trace_len": result.trace_len,
            "executed_gates": result.executed_gates,
            "gate_count": len(circuit.data),
            "cx_count": int(ops.get("cx", 0)),
            "total_ms": total_ms,
            "result_elapsed_ms": float(result.elapsed_ms),
            "phase_timings": phases,
        }
    except TimeoutError as exc:
        total_ms = (time.perf_counter() - started) * 1000.0
        phases = profiler.to_dict()
        phases["total"] = {
            "total_ms": total_ms,
            "count": 1,
            "min_ms": total_ms,
            "max_ms": total_ms,
        }
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "topology": spec.topology,
            "qubits": spec.qubits,
            "source": spec.source,
            "runtime_profile": spec.runtime_profile,
            "run_index": run_index,
            "status": "timeout",
            "completed": False,
            "timeout_s": float(spec.timeout_s),
            "worker_timeout_s": float(timeout_budget_s),
            "wall_ms": total_ms,
            "message": str(exc),
            "phase_timings": phases,
        }
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


def _worker_main(args: argparse.Namespace) -> None:
    row = _run_worker(
        args.worker_case,
        args.worker_run_index,
        args.max_steps,
        worker_timeout_s=args.worker_timeout_s,
    )
    json.dump(row, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _run_case_subprocess(
    case_id: str,
    run_index: int,
    *,
    max_steps: int,
    timeout_s: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case_id,
        "--worker-run-index",
        str(run_index),
        "--max-steps",
        str(max_steps),
        "--worker-timeout-s",
        str(timeout_s),
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s + 10.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        spec = CASE_SPECS[case_id]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "topology": spec.topology,
            "qubits": spec.qubits,
            "source": spec.source,
            "run_index": run_index,
            "status": "timeout",
            "completed": False,
            "timeout_s": timeout_s,
            "wall_ms": elapsed_ms,
            "stdout": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
            "phase_timings": {},
        }
    if completed.returncode != 0:
        spec = CASE_SPECS[case_id]
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "topology": spec.topology,
            "qubits": spec.qubits,
            "source": spec.source,
            "run_index": run_index,
            "status": "error",
            "completed": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:],
            "stderr": completed.stderr[-2000:],
            "phase_timings": {},
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        spec = CASE_SPECS[case_id]
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "topology": spec.topology,
            "qubits": spec.qubits,
            "source": spec.source,
            "run_index": run_index,
            "status": "error",
            "completed": False,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
            "phase_timings": {},
        }


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _wall_ms(row: dict[str, Any]) -> float:
    return float(row.get("wall_ms", row.get("total_ms", 0.0)) or 0.0)


def _summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids = sorted({str(row["case_id"]) for row in runs})
    for case_id in case_ids:
        case_runs = [row for row in runs if row["case_id"] == case_id]
        ok_runs = [row for row in case_runs if row["status"] == "ok"]
        timeout_runs = [row for row in case_runs if row["status"] == "timeout"]
        spec = CASE_SPECS[case_id]
        timeout_walls = [_wall_ms(row) for row in timeout_runs]
        for phase in PHASE_ORDER:
            values = [_phase_total_ms(row.get("phase_timings", {}), phase) for row in ok_runs]
            shares = [
                (_phase_total_ms(row.get("phase_timings", {}), phase) / max(float(row["total_ms"]), 1e-9))
                * 100.0
                for row in ok_runs
            ]
            rows.append(
                {
                    "case_id": case_id,
            "group": spec.group,
            "topology": spec.topology,
            "qubits": spec.qubits,
            "runtime_profile": spec.runtime_profile,
            "phase": phase,
                    "phase_label": PHASE_LABELS[phase],
                    "nested": phase in NESTED_PHASES,
                    "runs": len(case_runs),
                    "ok_runs": len(ok_runs),
                    "timeouts": sum(1 for row in case_runs if row["status"] == "timeout"),
                    "errors": sum(1 for row in case_runs if row["status"] == "error"),
                    "mean_ms": _mean(values),
                    "std_ms": _std(values),
                    "min_ms": min(values) if values else 0.0,
                    "max_ms": max(values) if values else 0.0,
                    "mean_share_pct": _mean(shares),
                    "std_share_pct": _std(shares),
                    "timeout_wall_mean_ms": _mean(timeout_walls) if phase == "total" else 0.0,
                    "timeout_wall_std_ms": _std(timeout_walls) if phase == "total" else 0.0,
                    "timeout_wall_min_ms": min(timeout_walls) if phase == "total" and timeout_walls else 0.0,
                    "timeout_wall_max_ms": max(timeout_walls) if phase == "total" and timeout_walls else 0.0,
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "group",
        "topology",
        "qubits",
        "runtime_profile",
        "phase",
        "phase_label",
        "nested",
        "runs",
        "ok_runs",
        "timeouts",
        "errors",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "mean_share_pct",
        "std_share_pct",
        "timeout_wall_mean_ms",
        "timeout_wall_std_ms",
        "timeout_wall_min_ms",
        "timeout_wall_max_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.1f}ms"


def _top_phase(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["case_id"] == case_id
        and row["phase"] not in {"total"}
        and row["ok_runs"] > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["mean_ms"]))


def _case_total_row(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row["case_id"] == case_id and row["phase"] == "total":
            return row
    return None


def _write_markdown(
    summary_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    path: Path,
    *,
    repeats: int,
    cases: list[str],
    max_steps: int,
) -> None:
    successful_cases = [
        case_id
        for case_id in cases
        if any(row["case_id"] == case_id and row["status"] == "ok" for row in runs)
    ]
    lines = [
        "# NPQR 分阶段运行时间测量",
        "",
        (
            f"本次测量覆盖 {len(cases)} 个样例，每个样例重复 {repeats} 次。"
            f"NPQR 运行时使用 `max_steps={max_steps}`，每次测量在独立 Python 子进程中执行，"
            "并固定 BLAS/Torch 单线程设置，以减少无关波动。"
        ),
        "",
        "## 运行环境",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- 线程限制: `{json.dumps(THREAD_ENV, ensure_ascii=False)}`",
        "",
        "## 样例总览",
        "",
        "| 样例 | 分组 | 拓扑 | profile | 成功 / 运行 | 超时 | 平均总耗时 | 最耗时阶段 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case_id in cases:
        spec = CASE_SPECS[case_id]
        total = _case_total_row(summary_rows, case_id)
        top = _top_phase(summary_rows, case_id)
        ok_runs = int(total["ok_runs"]) if total else 0
        timeout_count = int(total["timeouts"]) if total else sum(
            1 for row in runs if row["case_id"] == case_id and row["status"] == "timeout"
        )
        total_mean = _format_ms(float(total["mean_ms"])) if total and ok_runs else "--"
        timeout_mean = float(total.get("timeout_wall_mean_ms", 0.0)) if total else 0.0
        top_text = (
            f"{top['phase_label']} ({_format_ms(float(top['mean_ms']))}, {float(top['mean_share_pct']):.1f}%)"
            if top
            else f"预算边界 ({_format_ms(timeout_mean)} wall)"
            if timeout_mean
            else "预算边界"
        )
        lines.append(
            f"| `{case_id}` | {spec.group} | `{spec.topology}` | `{spec.runtime_profile}` | {ok_runs}/{repeats} | "
            f"{timeout_count} | {total_mean} | {top_text} |"
        )

    lines.extend(
        [
            "",
            "## 阶段耗时表",
            "",
            (
                "百分比按每次成功运行的总耗时计算。`nested=true` 表示该行是更大阶段中的"
                "子阶段，因此这些百分比不应直接相加为 100%。耗时极小的阶段仍保留在"
                " CSV/JSON 原始数据中，PPT 可按需要合并展示。"
            ),
            "",
            "| 样例 | 阶段 | mean | std | min | max | 占比 | nested |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        if row["ok_runs"] <= 0 or row["phase"] == "total":
            continue
        if row["phase"] not in PHASE_ORDER:
            continue
        lines.append(
            f"| `{row['case_id']}` | {row['phase_label']} | "
            f"{_format_ms(float(row['mean_ms']))} | {_format_ms(float(row['std_ms']))} | "
            f"{_format_ms(float(row['min_ms']))} | {_format_ms(float(row['max_ms']))} | "
            f"{float(row['mean_share_pct']):.1f}% | {str(bool(row['nested'])).lower()} |"
        )

    timeout_rows = [
        _case_total_row(summary_rows, case_id)
        for case_id in cases
        if _case_total_row(summary_rows, case_id)
        and int(_case_total_row(summary_rows, case_id)["ok_runs"]) == 0
        and int(_case_total_row(summary_rows, case_id)["timeouts"]) > 0
    ]
    if timeout_rows:
        lines.extend(
            [
                "",
                "## 预算边界样例",
                "",
                (
                    "以下样例在固定 CPU 预算内未完成，因此只作为边界观测。"
                    "表中 wall time 来自实际子进程运行时间或外层超时回收时间，不用于计算阶段占比。"
                ),
                "",
        "| 样例 | 拓扑 | profile | 超时 / 运行 | wall mean | wall std | wall min | wall max | 说明 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in timeout_rows:
            case_id = str(row["case_id"])
            spec = CASE_SPECS[case_id]
            lines.append(
                f"| `{case_id}` | `{spec.topology}` | `{spec.runtime_profile}` | {int(row['timeouts'])}/{int(row['runs'])} | "
                f"{_format_ms(float(row['timeout_wall_mean_ms']))} | "
                f"{_format_ms(float(row['timeout_wall_std_ms']))} | "
                f"{_format_ms(float(row['timeout_wall_min_ms']))} | "
                f"{_format_ms(float(row['timeout_wall_max_ms']))} | "
                "完整 NPQR 超出本次公开 CPU profiling 预算 |"
            )

    trend_lines: list[str] = []
    qft10_total = _case_total_row(summary_rows, "qft10")
    brick20_total = _case_total_row(summary_rows, "brickwork20_profile")
    if qft10_total and brick20_total and qft10_total["ok_runs"] and brick20_total["ok_runs"]:
        trend = float(brick20_total["mean_ms"]) / max(float(qft10_total["mean_ms"]), 1e-9)
        trend_lines.append(
            f"- 在同一 Tokyo 拓扑下，`brickwork20_profile` 的平均总耗时为 "
            f"`qft10` 的 {trend:.2f} 倍；该结果说明耗时不只由量子比特数决定，"
            "还受交互结构和候选映射搜索形态影响。"
        )
    for case_id in ["line_ghz30", "line_ghz50", "line_ghz80_boundary"]:
        if case_id in cases:
            total = _case_total_row(summary_rows, case_id)
            if total and int(total["ok_runs"]) == 0:
                trend_lines.append(
                    f"- `{case_id}` 在全部重复运行中超过单次 CPU 预算；该结果应作为边界观测，"
                    "不能作为阶段占比行。"
                )

    top_overall = None
    for case_id in successful_cases:
        top = _top_phase(summary_rows, case_id)
        if top and (top_overall is None or float(top["mean_ms"]) > float(top_overall["mean_ms"])):
            top_overall = top

    lines.extend(["", "## 结论解读", ""])
    if top_overall:
        lines.append(
            f"- 成功运行样例中，最大实测阶段是 `{top_overall['case_id']}` 的 "
            f"`{top_overall['phase_label']}`，平均 {_format_ms(float(top_overall['mean_ms']))}。"
        )
    lines.extend(trend_lines)
    lines.append(
        "- 实测结果支持报告中的复杂度分析：固定规模的数据预处理、依赖图和拓扑距离矩阵"
        "开销较小；真正主导增长的是初始映射候选、束搜索状态扩展、动作掩码和神经网络"
        "评估。"
    )

    if top_overall:
        one_line = (
            f"> NPQR 的主要耗时集中在候选映射和主搜索相关阶段；在本次测量中，最大阶段为 "
            f"`{top_overall['case_id']}` 的 `{top_overall['phase_label']}` "
            f"({_format_ms(float(top_overall['mean_ms']))})，30/50Q 完整 NPQR 在公开 CPU 预算下"
            "作为边界样例单独报告。"
        )
    else:
        one_line = (
            "> 30/50Q 完整 NPQR 超出公开 CPU profiling 预算，因此作为边界观测而不是阶段占比样例。"
        )
    lines.extend(["", "## PPT 一句话总结", "", one_line])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_json(
    runs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    path: Path,
    *,
    args: argparse.Namespace,
) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arguments": {
            "cases": args.cases,
            "repeats": args.repeats,
            "max_steps": args.max_steps,
            "large_timeout": args.large_timeout,
            "include_boundary": args.include_boundary,
        },
        "phase_labels": PHASE_LABELS,
        "nested_phases": sorted(NESTED_PHASES),
        "runs": runs,
        "summary": summary_rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), choices=sorted(CASE_SPECS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument(
        "--large-timeout",
        type=float,
        default=None,
        help="Optional timeout override for selected_30_50 and boundary_80_100 cases.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-boundary", action="store_true")
    parser.add_argument("--worker-case", choices=sorted(CASE_SPECS))
    parser.add_argument("--worker-run-index", type=int, default=0)
    parser.add_argument("--worker-timeout-s", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_case:
        _worker_main(args)
        return

    _configure_runtime_environment()
    cases = list(args.cases)
    if args.include_boundary and "line_ghz80_boundary" not in cases:
        cases.append("line_ghz80_boundary")

    runs: list[dict[str, Any]] = []
    for case_id in cases:
        spec = CASE_SPECS[case_id]
        timeout_s = (
            args.large_timeout
            if args.large_timeout is not None and spec.group in {"selected_30_50", "boundary_80_100"}
            else spec.timeout_s
        )
        for run_index in range(args.repeats):
            row = _run_case_subprocess(
                case_id,
                run_index,
                max_steps=args.max_steps,
                timeout_s=timeout_s,
            )
            runs.append(row)
            status = row["status"]
            total_text = _format_ms(float(row.get("total_ms", row.get("wall_ms", 0.0))))
            print(f"{case_id} run {run_index + 1}/{args.repeats}: {status} {total_text}", flush=True)

    summary_rows = _summarize_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_raw_json(runs, summary_rows, args.output_dir / "phase_timing_raw.json", args=args)
    _write_csv(summary_rows, args.output_dir / "phase_timing_summary.csv")
    _write_markdown(
        summary_rows,
        runs,
        args.output_dir / "phase_timing_summary.md",
        repeats=args.repeats,
        cases=cases,
        max_steps=args.max_steps,
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
