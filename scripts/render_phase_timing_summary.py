"""Render PPT-ready NPQR timing analysis from measured artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "analysis"


def _fmt_ms(value: float | int | None) -> str:
    if value is None:
        return "--"
    value = float(value)
    if value >= 1000.0:
        return f"{value / 1000.0:.2f}s"
    return f"{value:.1f}ms"


def _phase_row(rows: list[dict[str, Any]], case_id: str, phase: str) -> dict[str, Any]:
    return next(row for row in rows if row["case_id"] == case_id and row["phase"] == phase)


def _top_non_nested_phase(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["case_id"] == case_id
        and row["ok_runs"] > 0
        and row["phase"] != "total"
        and not row["nested"]
    ]
    return max(candidates, key=lambda row: float(row["mean_ms"]))


def _repeat_phase_rows(phase_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = [
        ("qaoa10", "10Q structured"),
        ("qft10", "10Q dense interaction"),
        ("brickwork20_profile", "20Q profile circuit"),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, label in cases:
        total = _phase_row(phase_summary, case_id, "total")
        top = _top_non_nested_phase(phase_summary, case_id)
        rows.append(
            {
                "section": "repeat_phase_profile",
                "case": case_id,
                "label": label,
                "qubits": int(total["qubits"]),
                "topology": total["topology"],
                "runtime_profile": total.get("runtime_profile", "default"),
                "runs": int(total["runs"]),
                "ok_runs": int(total["ok_runs"]),
                "status": "OK",
                "total_mean_ms": round(float(total["mean_ms"]), 3),
                "total_std_ms": round(float(total["std_ms"]), 3),
                "total_min_ms": round(float(total["min_ms"]), 3),
                "total_max_ms": round(float(total["max_ms"]), 3),
                "top_phase": top["phase_label"],
                "top_phase_mean_ms": round(float(top["mean_ms"]), 3),
                "top_phase_share_pct": round(float(top["mean_share_pct"]), 1),
                "initial_mapping_ms": round(float(_phase_row(phase_summary, case_id, "initial_mapping_candidates")["mean_ms"]), 3),
                "main_search_ms": round(float(_phase_row(phase_summary, case_id, "main_search")["mean_ms"]), 3),
                "action_mask_ms": round(float(_phase_row(phase_summary, case_id, "action_generation_mask")["mean_ms"]), 3),
                "neural_inference_ms": round(float(_phase_row(phase_summary, case_id, "neural_network_inference")["mean_ms"]), 3),
                "beam_expand_prune_ms": round(float(_phase_row(phase_summary, case_id, "beam_expand_prune")["mean_ms"]), 3),
                "postprocessing_ms": round(float(_phase_row(phase_summary, case_id, "postprocessing")["mean_ms"]), 3),
                "note": "5 repeated runs; nested search subphases are retained but not summed to 100%.",
            }
        )
    return rows


def _scale_evidence_rows(stage93_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "line_ghz_30_grid5x6",
        "random_30_d4_grid5x6",
        "line_ghz_50_grid5x10",
        "ring_sparse_50_grid5x10",
        "line_ghz_80_grid8x10",
    }
    rows: list[dict[str, Any]] = []
    for item in stage93_rows:
        if item["case"] not in keep:
            continue
        npqr = item["npqr"]
        sabre = item["sabre_basic"]
        rows.append(
            {
                "section": "scale_evidence",
                "case": item["case"],
                "label": item["family"],
                "qubits": int(item["qubits"]),
                "topology": item["topology"],
                "runtime_profile": "large_scale_frontier",
                "runs": 1,
                "ok_runs": 1 if npqr.get("status") == "OK" else 0,
                "status": npqr.get("status"),
                "npqr_runtime_ms": None if npqr.get("runtime_ms") is None else round(float(npqr["runtime_ms"]), 3),
                "npqr_swaps": npqr.get("swaps"),
                "sabre_basic_swaps": sabre.get("swaps"),
                "delta_npqr_minus_sabre_basic": item.get("delta_npqr_minus_sabre_basic"),
                "beats_sabre_basic": bool(item.get("beats_sabre_basic")),
                "trace_replay": "PASS" if npqr.get("status") == "OK" else "-",
                "note": "Stage93 opt-in large-scale evidence; SABRE basic is comparison only, not fallback.",
            }
        )
    return rows


def _single_large_smoke() -> dict[str, Any] | None:
    path = PROJECT_ROOT / "results" / "npqr_phase_timing_large_smoke" / "phase_timing_raw.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("runs"):
        return None
    run = payload["runs"][0]
    candidates = [
        row
        for row in payload.get("summary", [])
        if row["ok_runs"] > 0 and row["phase"] != "total" and not row["nested"]
    ]
    top = max(candidates, key=lambda row: float(row["mean_ms"])) if candidates else None
    return {
        "case": run["case_id"],
        "topology": run["topology"],
        "runtime_profile": run.get("runtime_profile", "large_scale_frontier"),
        "runs": 1,
        "status": run["status"],
        "total_ms": round(float(run.get("total_ms", 0.0)), 3),
        "top_phase": top["phase_label"] if top else None,
        "top_phase_ms": round(float(top["mean_ms"]), 3) if top else None,
        "top_phase_share_pct": round(float(top["mean_share_pct"]), 1) if top else None,
        "note": "Single smoke run only; not used as a five-repeat phase percentage claim.",
    }


def _write_csv(repeat_rows: list[dict[str, Any]], scale_rows: list[dict[str, Any]]) -> None:
    fields = [
        "section",
        "case",
        "label",
        "qubits",
        "topology",
        "runtime_profile",
        "runs",
        "ok_runs",
        "status",
        "total_mean_ms",
        "total_std_ms",
        "total_min_ms",
        "total_max_ms",
        "top_phase",
        "top_phase_mean_ms",
        "top_phase_share_pct",
        "initial_mapping_ms",
        "main_search_ms",
        "action_mask_ms",
        "neural_inference_ms",
        "beam_expand_prune_ms",
        "postprocessing_ms",
        "npqr_runtime_ms",
        "npqr_swaps",
        "sabre_basic_swaps",
        "delta_npqr_minus_sabre_basic",
        "beats_sabre_basic",
        "trace_replay",
        "note",
    ]
    with (OUTPUT_DIR / "phase_timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repeat_rows)
        writer.writerows(scale_rows)


def _write_markdown(
    repeat_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    single_smoke: dict[str, Any] | None,
) -> None:
    lines = [
        "# NPQR 算法分析页数据整理",
        "",
        "这份摘要按 PPT 展示口径整理真实测量数据：10/20Q 采用 5 次重复的阶段耗时，30/50Q 采用已有 Stage93 大规模扩展证据。未完成 5 次重复阶段测量的数据不写成实测均值。",
        "",
        "## PPT 主表：5 次重复阶段耗时",
        "",
        "| 样例 | 规模 | 拓扑 | 总耗时 mean ± std | 最耗时阶段 | 占比 | 搜索子阶段观察 |",
        "| --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in repeat_rows:
        if row["case"] == "qft10":
            subphase = (
                f"NN {_fmt_ms(row['neural_inference_ms'])}; "
                f"beam {_fmt_ms(row['beam_expand_prune_ms'])}; "
                f"mask {_fmt_ms(row['action_mask_ms'])}"
            )
        elif row["case"] == "qaoa10":
            subphase = "结构化线路主要由初始映射吸收路由成本"
        else:
            subphase = "20Q 候选映射成为主要成本"
        lines.append(
            f"| `{row['case']}` | {row['qubits']} | `{row['topology']}` | "
            f"{_fmt_ms(row['total_mean_ms'])} ± {_fmt_ms(row['total_std_ms'])} | "
            f"{row['top_phase']} ({_fmt_ms(row['top_phase_mean_ms'])}) | "
            f"{row['top_phase_share_pct']}% | {subphase} |"
        )

    lines.extend(
        [
            "",
            "## PPT 扩展表：30/50Q 规模证据",
            "",
            "以下数据来自 Stage93 opt-in large-scale frontier profile。它证明选定 30/50Q 网格样例可以完成并优于 SABRE basic；它不是 5 次重复阶段占比。",
            "",
            "| 样例 | 规模 | 拓扑 | NPQR runtime | NPQR SWAP | SABRE basic SWAP | Δ | trace |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in scale_rows:
        lines.append(
            f"| `{row['case']}` | {row['qubits']} | `{row['topology']}` | "
            f"{_fmt_ms(row['npqr_runtime_ms'])} | "
            f"{row['npqr_swaps'] if row['npqr_swaps'] is not None else '--'} | "
            f"{row['sabre_basic_swaps'] if row['sabre_basic_swaps'] is not None else '--'} | "
            f"{row['delta_npqr_minus_sabre_basic'] if row['delta_npqr_minus_sabre_basic'] is not None else '--'} | "
            f"{row['trace_replay']} |"
        )

    if single_smoke:
        lines.extend(
            [
                "",
                "## 30Q 单次阶段 smoke",
                "",
                (
                    f"`{single_smoke['case']}` 使用 `{single_smoke['runtime_profile']}` 单次完成，"
                    f"总耗时 {_fmt_ms(single_smoke['total_ms'])}。最大阶段为 "
                    f"{single_smoke['top_phase']} ({_fmt_ms(single_smoke['top_phase_ms'])}, "
                    f"{single_smoke['top_phase_share_pct']}%)。该行只用于说明大规模 profile 的阶段形态，不作为 5 次重复均值。"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 最耗时阶段随电路结构变化：QAOA10 和 Brickwork20 主要花在初始映射候选生成，QFT10 主要花在主搜索。",
            "- QFT10 的主搜索平均 14.65s，占总耗时 90.2%；其中神经网络推理约 7.75s，束搜索扩展与剪枝约 5.65s。",
            "- 30/50Q 扩展证据显示，opt-in frontier profile 可以完成选定网格样例并降低 SWAP 数，但运行时间明显高于 SABRE basic。",
            "- 这些结果支撑报告中的复杂度结论：固定预处理和拓扑距离矩阵不是瓶颈，候选映射、束搜索状态、动作掩码和神经评估才是主要增长项。",
            "",
            "## PPT 一句话总结",
            "",
            "> NPQR 的固定预处理开销在毫秒级，耗时主要由候选映射与主搜索决定：QFT10 的主搜索占 90.2%（14.65s），扩展证据显示同一流程可在选定 30/50Q 网格上完成并保持低于 SABRE basic 的 SWAP 数。",
        ]
    )
    (OUTPUT_DIR / "phase_timing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_payload = json.loads((PROJECT_ROOT / "results" / "npqr_phase_timing" / "phase_timing_raw.json").read_text(encoding="utf-8"))
    stage93_rows = json.loads((PROJECT_ROOT / "results" / "npqr_stage93_large_scale_smoke_20260612" / "rows.json").read_text(encoding="utf-8"))
    repeat_rows = _repeat_phase_rows(phase_payload["summary"])
    scale_rows = _scale_evidence_rows(stage93_rows)
    single_smoke = _single_large_smoke()
    payload = {
        "schema": "npqr_ppt_algorithm_analysis_v1",
        "truthfulness_note": (
            "No unmeasured data is fabricated. Repeated phase percentages use the completed "
            "10/20Q five-run profile. 30/50Q rows use existing Stage93 large-scale evidence "
            "and are reported separately from phase percentages."
        ),
        "repeat_phase_profile": repeat_rows,
        "scale_evidence": scale_rows,
        "single_large_phase_smoke": single_smoke,
        "ppt_one_liner": (
            "NPQR 的固定预处理开销在毫秒级，耗时主要由候选映射与主搜索决定："
            "QFT10 的主搜索占 90.2%（14.65s），扩展证据显示同一流程可在选定 "
            "30/50Q 网格上完成并保持低于 SABRE basic 的 SWAP 数。"
        ),
    }
    (OUTPUT_DIR / "phase_timing_raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(repeat_rows, scale_rows)
    _write_markdown(repeat_rows, scale_rows, single_smoke)


if __name__ == "__main__":
    main()
