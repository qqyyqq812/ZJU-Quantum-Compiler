"""Unified machine-readable evidence for NPQR and public routing claims."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE7_DIR = PROJECT_ROOT / "results" / "npqr_stage7_metric_sprint_recovered_trace"
STAGE8_DIR = PROJECT_ROOT / "results" / "npqr_stage8_metric_sprint"
STAGE8_HARDCASE_DIR = PROJECT_ROOT / "results" / "npqr_stage8_hardcase_sprint"
STAGE9_TEACHER_SCAN_DIR = PROJECT_ROOT / "results" / "npqr_stage9_teacher_scan"
STAGE9_TEACHER_SCAN_EXT_DIR = PROJECT_ROOT / "results" / "npqr_stage9_teacher_scan_ghz_vqe_ext"
STAGE9_MIXED_DATASET_DIR = PROJECT_ROOT / "results" / "npqr_stage9_mixed_dataset"
STAGE9_METRIC_SPRINT_DIR = PROJECT_ROOT / "results" / "npqr_stage9_metric_sprint"
STAGE9_MAPPING_PROBE_DIR = PROJECT_ROOT / "results" / "npqr_stage9_mapping_probe"
STAGE10_MAPPING_SELECTOR_DIR = PROJECT_ROOT / "results" / "npqr_stage10_mapping_selector"
STAGE11_SELECTOR_RUNTIME_DIR = PROJECT_ROOT / "results" / "npqr_stage11_selector_runtime"
STAGE12_SELECTOR_BOUNDARY_DIR = PROJECT_ROOT / "results" / "npqr_stage12_selector_boundary"
STAGE12_SELECTOR_VQE10_TOP4_DIR = PROJECT_ROOT / "results" / "npqr_stage12_selector_boundary_vqe10_top4"
STAGE13_ADAPTIVE_SELECTOR_DIR = PROJECT_ROOT / "results" / "npqr_stage13_adaptive_selector_boundary"
STAGE14_ADAPTIVE_DATASET_DIR = PROJECT_ROOT / "results" / "npqr_stage14_adaptive_dataset"
STAGE15_METRIC_SPRINT_DIR = PROJECT_ROOT / "results" / "npqr_stage15_metric_sprint"
STAGE16_HARDCASE_SCOUT_DIR = PROJECT_ROOT / "results" / "npqr_stage16_hardcase_scout"
STAGE17_HARDCASE_DATASET_DIR = PROJECT_ROOT / "results" / "npqr_stage17_hardcase_dataset"
STAGE18_METRIC_SPRINT_DIR = PROJECT_ROOT / "results" / "npqr_stage18_metric_sprint"
STAGE19_TRAINING_DIAGNOSTICS_DIR = PROJECT_ROOT / "results" / "npqr_stage19_training_diagnostics"
STAGE20_GHZ10_STALL_DIAGNOSTICS_DIR = PROJECT_ROOT / "results" / "npqr_stage20_ghz10_stall_diagnostics"
STAGE21_SUFFIX_REPAIR_GATE_DIR = PROJECT_ROOT / "results" / "npqr_stage21_suffix_repair_gate"
STAGE22_SUFFIX_TRAINING_READINESS_DIR = PROJECT_ROOT / "results" / "npqr_stage22_suffix_training_readiness"
STAGE23_GPU_SWEEP_PLAN_DIR = PROJECT_ROOT / "results" / "npqr_stage23_gpu_sweep_plan"
STAGE23_GPU_SWEEP_SUMMARY_DIR = PROJECT_ROOT / "results" / "npqr_stage23_gpu_sweep_summary"
STAGE24_TRAINING_GO_NO_GO_DIR = PROJECT_ROOT / "results" / "npqr_stage24_training_go_no_go"
STAGE25_POST_SWEEP_DECISION_DIR = PROJECT_ROOT / "results" / "npqr_stage25_post_sweep_decision"
STAGE26_DISTILLATION_AUDIT_DIR = PROJECT_ROOT / "results" / "npqr_stage26_distillation_audit"
DEFAULT_EVIDENCE_PATH = PROJECT_ROOT / "results" / "npqr_evidence_manifest.json"


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_from_gate(gate: dict[str, Any] | None, side: str) -> dict[str, Any]:
    if not gate:
        return {"completed": None, "total": None, "avg_swaps_completed": None, "by_circuit": {}}
    summary = gate.get(side) or {}
    return {
        "completed": summary.get("completed"),
        "total": summary.get("total"),
        "avg_swaps_completed": summary.get("avg_swaps_completed"),
        "by_circuit": summary.get("by_circuit", {}),
    }


def _dataset_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE8_DIR / "combined_stage8_samples.manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "samples": manifest.get("samples", 0),
        "accepted_traces": manifest.get("accepted_traces", 0),
        "rejected_traces": manifest.get("rejected_traces", []),
        "weight_policy": manifest.get("weight_policy", {}),
        "path": "results/npqr_stage8_metric_sprint/combined_stage8_samples.jsonl",
    }


def _training_summary() -> dict[str, Any]:
    metrics = _read_json(STAGE8_DIR / "stage8_finetune_metrics.json") or {}
    history = metrics.get("history") or []
    return {
        "schema": metrics.get("schema"),
        "output_model": metrics.get("output_model"),
        "raw_samples": metrics.get("raw_samples", 0),
        "materialized_samples": metrics.get("materialized_samples", 0),
        "epochs": metrics.get("epochs", 0),
        "lr": metrics.get("lr"),
        "value_loss_weight": metrics.get("value_loss_weight"),
        "last_epoch": history[-1] if history else None,
    }


def _stage8_attempt_summary(
    *,
    attempt_id: str,
    label: str,
    source_dir: Path,
    dataset_manifest_name: str,
    dataset_path: str,
    metrics_name: str,
    trace_path: str,
) -> dict[str, Any]:
    manifest = _read_json(source_dir / "manifest.json") or {}
    raw_gate = _read_json(source_dir / "raw_route_gate.json") or {}
    quick_gate = _read_json(source_dir / "quick_route_gate.json") or {}
    qap_gate = _read_json(source_dir / "qap_route_gate.json") or {}
    trace_gate = _read_json(source_dir / "candidate_trace_replay_gate.json") or {}
    dataset_manifest = _read_json(source_dir / dataset_manifest_name) or {}
    metrics = _read_json(source_dir / metrics_name) or {}
    history = metrics.get("history") or []

    return {
        "id": attempt_id,
        "label": label,
        "source_dir": str(source_dir.relative_to(PROJECT_ROOT)),
        "candidate_model": manifest.get("candidate_model"),
        "decision": manifest.get("decision"),
        "promote_checkpoint": bool(manifest.get("promote_checkpoint")),
        "raw_gate_passed": bool(manifest.get("raw_gate_passed")),
        "quick_gate_passed": bool(manifest.get("quick_gate_passed")),
        "qap_gate_passed": bool(manifest.get("qap_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "raw": {
            "completion_gain": raw_gate.get("completion_gain"),
            "baseline": _summary_from_gate(raw_gate, "baseline"),
            "candidate": _summary_from_gate(raw_gate, "candidate"),
            "improvements": raw_gate.get("improvements", []),
            "regressions": raw_gate.get("regressions", []),
        },
        "quick": {
            "passed": bool(quick_gate.get("passed")),
            "candidate": _summary_from_gate(quick_gate, "candidate"),
            "regressions": quick_gate.get("regressions", []),
        },
        "qap": {
            "passed": bool(qap_gate.get("passed")),
            "candidate": _summary_from_gate(qap_gate, "candidate"),
            "regressions": qap_gate.get("regressions", []),
        },
        "trace_replay": {
            "passed": bool(trace_gate.get("passed")),
            "rows": trace_gate.get("rows"),
            "path": trace_path,
        },
        "dataset": {
            "schema": dataset_manifest.get("schema"),
            "samples": dataset_manifest.get("samples", 0),
            "accepted_traces": dataset_manifest.get("accepted_traces", 0),
            "rejected_traces": dataset_manifest.get("rejected_traces", []),
            "weight_policy": dataset_manifest.get("weight_policy", {}),
            "path": dataset_path,
        },
        "training": {
            "schema": metrics.get("schema"),
            "output_model": metrics.get("output_model"),
            "raw_samples": metrics.get("raw_samples", 0),
            "materialized_samples": metrics.get("materialized_samples", 0),
            "epochs": metrics.get("epochs", 0),
            "lr": metrics.get("lr"),
            "value_loss_weight": metrics.get("value_loss_weight"),
            "last_epoch": history[-1] if history else None,
        },
    }


def _stage9_teacher_scan_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE9_TEACHER_SCAN_DIR / "manifest.json") or {}
    trace_gate = _read_json(STAGE9_TEACHER_SCAN_DIR / "legacy_trace_replay_gate.json") or {}
    examples = manifest.get("examples", [])
    completed = manifest.get("completed_circuits", [])
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage9_teacher_scan",
        "decision": manifest.get("decision"),
        "mapping_strategy": manifest.get("mapping_strategy"),
        "examples": examples,
        "completed_trace_rows": manifest.get("completed_trace_rows", 0),
        "completed_circuits": completed,
        "missing_completed_circuits": manifest.get("missing_completed_circuits", []),
        "coverage": {
            "completed": len(completed),
            "total": len(examples),
        },
        "timeout_rows": manifest.get("timeout_rows", 0),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": trace_gate.get("rows"),
        "trace_path": manifest.get("trace_path"),
        "scan_path": manifest.get("scan_path"),
        "notes": manifest.get("notes", []),
    }


def _stage9_mixed_dataset_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE9_MIXED_DATASET_DIR / "combined_stage9_mixed_samples.manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage9_mixed_dataset",
        "path": "results/npqr_stage9_mixed_dataset/combined_stage9_mixed_samples.jsonl",
        "samples": manifest.get("samples", 0),
        "accepted_traces": manifest.get("accepted_traces", 0),
        "rejected_traces": manifest.get("rejected_traces", []),
        "trace_counts_by_role": manifest.get("trace_counts_by_role", {}),
        "sample_counts_by_role": manifest.get("sample_counts_by_role", {}),
        "trace_counts_by_circuit": manifest.get("trace_counts_by_circuit", {}),
        "weight_policy": manifest.get("weight_policy", {}),
        "notes": manifest.get("notes", []),
    }


def _stage9_extension_scan_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE9_TEACHER_SCAN_EXT_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage9_teacher_scan_ghz_vqe_ext",
        "decision": manifest.get("decision"),
        "examples": manifest.get("examples", []),
        "completed_trace_rows": manifest.get("completed_trace_rows", 0),
        "completed_circuits": manifest.get("completed_circuits", []),
        "missing_completed_circuits": manifest.get("missing_completed_circuits", []),
        "timeout_rows": manifest.get("timeout_rows", 0),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_path": manifest.get("trace_path"),
        "scan_path": manifest.get("scan_path"),
        "notes": manifest.get("notes", []),
    }


def _stage9_mapping_probe_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE9_MAPPING_PROBE_DIR / "manifest.json") or {}
    trace_gate = _read_json(STAGE9_MAPPING_PROBE_DIR / "mapping_probe_trace_replay_gate.json") or {}
    rows = _read_json(STAGE9_MAPPING_PROBE_DIR / "mapping_probe_rows.json") or []
    completed_by_strategy: dict[str, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if not row.get("completed"):
                continue
            strategy = str(row.get("mapping_strategy", "unknown"))
            completed_by_strategy[strategy] = completed_by_strategy.get(strategy, 0) + 1
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage9_mapping_probe",
        "decision": manifest.get("decision"),
        "examples": manifest.get("examples", []),
        "mapping_strategies": manifest.get("mapping_strategies", []),
        "rows": manifest.get("rows", 0),
        "completed_rows": manifest.get("completed_rows", 0),
        "completed_circuits": manifest.get("completed_circuits", []),
        "missing_completed_circuits": manifest.get("missing_completed_circuits", []),
        "completed_by_strategy": completed_by_strategy,
        "timeout_rows": manifest.get("timeout_rows", 0),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": trace_gate.get("rows"),
        "trace_path": manifest.get("trace_path"),
        "rows_path": manifest.get("rows_path"),
        "notes": manifest.get("notes", []),
    }


def _stage10_mapping_selector_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE10_MAPPING_SELECTOR_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage10_mapping_selector",
        "source_rows": manifest.get("source_rows"),
        "decision": manifest.get("decision"),
        "selector": manifest.get("selector", {}),
        "top1_completed": manifest.get("top1_completed", 0),
        "total_circuits": manifest.get("total_circuits", 0),
        "top1_completion_rate": manifest.get("top1_completion_rate", 0.0),
        "circuits": manifest.get("circuits", {}),
        "notes": manifest.get("notes", []),
    }


def _stage11_selector_runtime_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE11_SELECTOR_RUNTIME_DIR / "manifest.json") or {}
    route_gate = _read_json(STAGE11_SELECTOR_RUNTIME_DIR / "route_gate.json") or {}
    quick_gate = _read_json(STAGE11_SELECTOR_RUNTIME_DIR / "quick_route_gate.json") or {}
    trace_gate = _read_json(STAGE11_SELECTOR_RUNTIME_DIR / "trace_replay_gate.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage11_selector_runtime",
        "decision": manifest.get("decision"),
        "passed": bool(manifest.get("passed")),
        "mapping_strategy": manifest.get("mapping_strategy"),
        "selector_role": manifest.get("selector_role"),
        "selector_top_k": manifest.get("selector_top_k"),
        "route_gate_passed": bool(manifest.get("route_gate_passed")),
        "quick_gate_passed": bool(manifest.get("quick_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "expected_trace_replay_rows": manifest.get("expected_trace_replay_rows"),
        "route": {
            "completion_gain": route_gate.get("completion_gain"),
            "baseline": _summary_from_gate(route_gate, "baseline"),
            "candidate": _summary_from_gate(route_gate, "candidate"),
            "improvements": route_gate.get("improvements", []),
            "regressions": route_gate.get("regressions", []),
        },
        "quick": {
            "passed": bool(quick_gate.get("passed")),
            "candidate": _summary_from_gate(quick_gate, "candidate"),
            "regressions": quick_gate.get("regressions", []),
        },
        "trace_replay": {
            "passed": bool(trace_gate.get("passed")),
            "rows": trace_gate.get("rows"),
            "path": "results/npqr_stage11_selector_runtime/selector_completed_traces.jsonl",
        },
        "notes": manifest.get("notes", []),
    }


def _selector_boundary_summary(source_dir: Path, source_label: str) -> dict[str, Any]:
    manifest = _read_json(source_dir / "manifest.json") or {}
    route_gate = _read_json(source_dir / "route_gate.json") or {}
    trace_gate = _read_json(source_dir / "trace_replay_gate.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": source_label,
        "decision": manifest.get("decision"),
        "passed": bool(manifest.get("passed")),
        "circuits": manifest.get("circuits", []),
        "mqt_available": manifest.get("mqt_available"),
        "circuit_metadata": manifest.get("circuit_metadata", {}),
        "selector_top_k": manifest.get("selector_top_k"),
        "route_gate_passed": bool(manifest.get("route_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "expected_trace_replay_rows": manifest.get("expected_trace_replay_rows"),
        "route": {
            "completion_gain": route_gate.get("completion_gain"),
            "baseline": _summary_from_gate(route_gate, "baseline"),
            "candidate": _summary_from_gate(route_gate, "candidate"),
            "improvements": route_gate.get("improvements", []),
            "regressions": route_gate.get("regressions", []),
        },
        "trace_replay": {
            "passed": bool(trace_gate.get("passed")),
            "rows": trace_gate.get("rows"),
            "path": f"{source_label}/selector_completed_traces.jsonl",
        },
        "notes": manifest.get("notes", []),
    }


def _stage12_selector_boundary_summary() -> dict[str, Any]:
    full_top2 = _selector_boundary_summary(
        STAGE12_SELECTOR_BOUNDARY_DIR,
        "results/npqr_stage12_selector_boundary",
    )
    vqe10_top4 = _selector_boundary_summary(
        STAGE12_SELECTOR_VQE10_TOP4_DIR,
        "results/npqr_stage12_selector_boundary_vqe10_top4",
    )
    return {
        "schema": "npqr_stage12_selector_boundary_summary_v1",
        "decision": (
            "hold_full_boundary_use_adaptive_topk"
            if full_top2.get("decision") == "hold_expanded_selector_boundary"
            and vqe10_top4.get("passed")
            else full_top2.get("decision")
        ),
        "full_top2": full_top2,
        "vqe10_top4_rescue": vqe10_top4,
        "notes": [
            "The full 8-circuit MQT/fallback boundary is stronger than checked7.",
            "Selector top-2 is not promoted on the full boundary because vqe_10 regresses.",
            "Targeted vqe_10 top-4 passes, suggesting adaptive top-k should be tested before broader claims.",
        ],
    }


def _stage13_adaptive_selector_boundary_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE13_ADAPTIVE_SELECTOR_DIR / "manifest.json") or {}
    route_gate = _read_json(STAGE13_ADAPTIVE_SELECTOR_DIR / "route_gate.json") or {}
    top2_gate = _read_json(STAGE13_ADAPTIVE_SELECTOR_DIR / "top2_route_gate.json") or {}
    top4_gate = _read_json(STAGE13_ADAPTIVE_SELECTOR_DIR / "top4_route_gate.json") or {}
    trace_gate = _read_json(STAGE13_ADAPTIVE_SELECTOR_DIR / "trace_replay_gate.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage13_adaptive_selector_boundary",
        "decision": manifest.get("decision"),
        "passed": bool(manifest.get("passed")),
        "circuits": manifest.get("circuits", []),
        "mqt_available": manifest.get("mqt_available"),
        "circuit_metadata": manifest.get("circuit_metadata", {}),
        "mapping_strategy": manifest.get("mapping_strategy"),
        "adaptive_rule": manifest.get("adaptive_rule"),
        "selector_top_k_primary": manifest.get("selector_top_k_primary"),
        "selector_top_k_rescue": manifest.get("selector_top_k_rescue"),
        "selection": manifest.get("selection", {}),
        "route_gate_passed": bool(manifest.get("route_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "expected_trace_replay_rows": manifest.get("expected_trace_replay_rows"),
        "top2": {
            "passed": bool(top2_gate.get("passed")),
            "completion_gain": top2_gate.get("completion_gain"),
            "candidate": _summary_from_gate(top2_gate, "candidate"),
            "regressions": top2_gate.get("regressions", []),
        },
        "top4": {
            "passed": bool(top4_gate.get("passed")),
            "completion_gain": top4_gate.get("completion_gain"),
            "candidate": _summary_from_gate(top4_gate, "candidate"),
            "regressions": top4_gate.get("regressions", []),
        },
        "route": {
            "completion_gain": route_gate.get("completion_gain"),
            "baseline": _summary_from_gate(route_gate, "baseline"),
            "candidate": _summary_from_gate(route_gate, "candidate"),
            "improvements": route_gate.get("improvements", []),
            "regressions": route_gate.get("regressions", []),
        },
        "trace_replay": {
            "passed": bool(trace_gate.get("passed")),
            "rows": trace_gate.get("rows"),
            "path": "results/npqr_stage13_adaptive_selector_boundary/adaptive_completed_traces.jsonl",
        },
        "notes": manifest.get("notes", []),
    }


def _stage14_adaptive_dataset_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE14_ADAPTIVE_DATASET_DIR / "combined_stage14_adaptive_samples.manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage14_adaptive_dataset",
        "path": "results/npqr_stage14_adaptive_dataset/combined_stage14_adaptive_samples.jsonl",
        "accepted_traces": manifest.get("accepted_traces", 0),
        "rejected_traces": manifest.get("rejected_traces", []),
        "samples": manifest.get("samples", 0),
        "trace_inputs": manifest.get("trace_inputs", []),
        "trace_counts_by_role": manifest.get("trace_counts_by_role", {}),
        "sample_counts_by_role": manifest.get("sample_counts_by_role", {}),
        "trace_counts_by_circuit": manifest.get("trace_counts_by_circuit", {}),
        "trace_counts_by_stage": manifest.get("trace_counts_by_stage", {}),
        "weight_policy": manifest.get("weight_policy", {}),
        "stage13_boundary": manifest.get("stage13_boundary", {}),
        "notes": manifest.get("notes", []),
    }


def _stage15_finetune_gate_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE15_METRIC_SPRINT_DIR / "manifest.json") or {}
    training = (manifest.get("candidate") or {}).get("training") or {}
    history = training.get("history") or []
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage15_metric_sprint",
        "dataset": "results/npqr_stage14_adaptive_dataset/combined_stage14_adaptive_samples.jsonl",
        "base_model": manifest.get("base_model"),
        "candidate_model": (manifest.get("candidate") or {}).get("model"),
        "decision": manifest.get("decision"),
        "promote_checkpoint": bool(manifest.get("promote_checkpoint")),
        "quick_gate_passed": bool(manifest.get("quick_gate_passed")),
        "checked7_gate_passed": bool(manifest.get("checked7_gate_passed")),
        "expanded_gate_passed": bool(manifest.get("expanded_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "quick": {
            "completion_gain": (manifest.get("quick") or {}).get("completion_gain"),
            "candidate": (manifest.get("quick") or {}).get("candidate", {}),
            "regressions": (manifest.get("quick") or {}).get("regressions", []),
        },
        "checked7": {
            "completion_gain": (manifest.get("checked7") or {}).get("completion_gain"),
            "baseline": (manifest.get("checked7") or {}).get("baseline", {}),
            "candidate": (manifest.get("checked7") or {}).get("candidate", {}),
            "regressions": (manifest.get("checked7") or {}).get("regressions", []),
        },
        "expanded_adaptive": {
            "completion_gain": (manifest.get("expanded_adaptive") or {}).get("completion_gain"),
            "baseline": (manifest.get("expanded_adaptive") or {}).get("baseline", {}),
            "candidate": (manifest.get("expanded_adaptive") or {}).get("candidate", {}),
            "regressions": (manifest.get("expanded_adaptive") or {}).get("regressions", []),
        },
        "training": {
            "schema": training.get("schema"),
            "raw_samples": training.get("raw_samples", 0),
            "materialized_samples": training.get("materialized_samples", 0),
            "epochs": training.get("epochs", 0),
            "lr": training.get("lr"),
            "value_loss_weight": training.get("value_loss_weight"),
            "last_epoch": history[-1] if history else None,
        },
        "notes": manifest.get("notes", []),
    }


def _stage16_hardcase_scout_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE16_HARDCASE_SCOUT_DIR / "manifest.json") or {}
    trace_gate = _read_json(STAGE16_HARDCASE_SCOUT_DIR / "trace_replay_gate.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage16_hardcase_scout",
        "decision": manifest.get("decision"),
        "circuits": manifest.get("circuits", []),
        "config_ids": manifest.get("config_ids", []),
        "rows": manifest.get("rows", 0),
        "timeout_rows": manifest.get("timeout_rows", 0),
        "completed_rows": manifest.get("completed_rows", 0),
        "completed_circuits": manifest.get("completed_circuits", []),
        "missing_completed_circuits": manifest.get("missing_completed_circuits", []),
        "best_by_circuit": manifest.get("best_by_circuit", {}),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "trace_replay": {
            "passed": bool(trace_gate.get("passed")),
            "rows": trace_gate.get("rows"),
            "path": "results/npqr_stage16_hardcase_scout/stage16_completed_traces.jsonl",
            "results": trace_gate.get("results", []),
        },
        "notes": manifest.get("notes", []),
    }


def _stage17_hardcase_dataset_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE17_HARDCASE_DATASET_DIR / "combined_stage17_hardcase_samples.manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage17_hardcase_dataset",
        "path": "results/npqr_stage17_hardcase_dataset/combined_stage17_hardcase_samples.jsonl",
        "accepted_traces": manifest.get("accepted_traces", 0),
        "rejected_traces": manifest.get("rejected_traces", []),
        "samples": manifest.get("samples", 0),
        "trace_inputs": manifest.get("trace_inputs", []),
        "trace_counts_by_role": manifest.get("trace_counts_by_role", {}),
        "sample_counts_by_role": manifest.get("sample_counts_by_role", {}),
        "trace_counts_by_circuit": manifest.get("trace_counts_by_circuit", {}),
        "trace_counts_by_stage": manifest.get("trace_counts_by_stage", {}),
        "weight_policy": manifest.get("weight_policy", {}),
        "stage16_scout": manifest.get("stage16_scout", {}),
        "notes": manifest.get("notes", []),
    }


def _stage18_finetune_gate_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE18_METRIC_SPRINT_DIR / "manifest.json") or {}
    training = (manifest.get("candidate") or {}).get("training") or {}
    history = training.get("history") or []
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage18_metric_sprint",
        "dataset": "results/npqr_stage17_hardcase_dataset/combined_stage17_hardcase_samples.jsonl",
        "base_model": manifest.get("base_model"),
        "candidate_model": (manifest.get("candidate") or {}).get("model"),
        "decision": manifest.get("decision"),
        "promote_checkpoint": bool(manifest.get("promote_checkpoint")),
        "quick_gate_passed": bool(manifest.get("quick_gate_passed")),
        "checked7_gate_passed": bool(manifest.get("checked7_gate_passed")),
        "expanded_gate_passed": bool(manifest.get("expanded_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "quick": {
            "completion_gain": (manifest.get("quick") or {}).get("completion_gain"),
            "candidate": (manifest.get("quick") or {}).get("candidate", {}),
            "regressions": (manifest.get("quick") or {}).get("regressions", []),
        },
        "checked7": {
            "completion_gain": (manifest.get("checked7") or {}).get("completion_gain"),
            "baseline": (manifest.get("checked7") or {}).get("baseline", {}),
            "candidate": (manifest.get("checked7") or {}).get("candidate", {}),
            "regressions": (manifest.get("checked7") or {}).get("regressions", []),
        },
        "expanded_adaptive": {
            "completion_gain": (manifest.get("expanded_adaptive") or {}).get("completion_gain"),
            "baseline": (manifest.get("expanded_adaptive") or {}).get("baseline", {}),
            "candidate": (manifest.get("expanded_adaptive") or {}).get("candidate", {}),
            "regressions": (manifest.get("expanded_adaptive") or {}).get("regressions", []),
        },
        "training": {
            "schema": training.get("schema"),
            "raw_samples": training.get("raw_samples", 0),
            "materialized_samples": training.get("materialized_samples", 0),
            "epochs": training.get("epochs", 0),
            "lr": training.get("lr"),
            "value_loss_weight": training.get("value_loss_weight"),
            "last_epoch": history[-1] if history else None,
        },
        "notes": manifest.get("notes", []),
    }


def _stage19_training_diagnostics_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE19_TRAINING_DIAGNOSTICS_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage19_training_diagnostics",
        "dataset": "results/npqr_stage17_hardcase_dataset/combined_stage17_hardcase_samples.jsonl",
        "base_model": manifest.get("base_model"),
        "stage18_model": manifest.get("stage18_model"),
        "raw_samples": manifest.get("raw_samples", 0),
        "materialized_samples": manifest.get("materialized_samples", 0),
        "base_teacher_fit": manifest.get("base_teacher_fit", {}),
        "stage18_teacher_fit": manifest.get("stage18_teacher_fit", {}),
        "stage18_delta": manifest.get("stage18_delta", {}),
        "tiny_overfit": manifest.get("tiny_overfit", {}),
        "gpu_decision": manifest.get("gpu_decision", {}),
        "notes": manifest.get("notes", []),
    }


def _stage20_ghz10_stall_diagnostics_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE20_GHZ10_STALL_DIAGNOSTICS_DIR / "manifest.json") or {}
    suffix = manifest.get("suffix_search") or {}
    diagnosis = manifest.get("diagnosis") or {}
    selected = manifest.get("selected_row") or {}
    stall = manifest.get("stall_state") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage20_ghz10_stall_diagnostics",
        "source_rows": manifest.get("source_rows"),
        "circuit": manifest.get("circuit"),
        "selected_config_id": selected.get("config_id"),
        "selected_action_trace": selected.get("action_trace", []),
        "selected_executed_gates": selected.get("executed_gates"),
        "selected_swaps": selected.get("swaps"),
        "stall_remaining_gates": diagnosis.get("remaining_gates"),
        "stall_remaining_two_qubit_gates": diagnosis.get("remaining_two_qubit_gates"),
        "stall_front": stall.get("front", []),
        "short_suffix_exists": bool(diagnosis.get("short_suffix_exists")),
        "suffix": suffix.get("suffix", []),
        "suffix_len": suffix.get("suffix_len"),
        "suffix_nodes": suffix.get("nodes"),
        "gpu_recommended": bool(diagnosis.get("gpu_recommended")),
        "decision": diagnosis.get("decision"),
        "notes": manifest.get("notes", []),
    }


def _stage21_suffix_repair_gate_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE21_SUFFIX_REPAIR_GATE_DIR / "manifest.json") or {}
    route_gate = _read_json(STAGE21_SUFFIX_REPAIR_GATE_DIR / "route_gate.json") or {}
    trace_gate = _read_json(STAGE21_SUFFIX_REPAIR_GATE_DIR / "trace_replay_gate.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage21_suffix_repair_gate",
        "source_stage13_dir": "results/npqr_stage13_adaptive_selector_boundary",
        "source_stage20_dir": "results/npqr_stage20_ghz10_stall_diagnostics",
        "decision": manifest.get("decision"),
        "passed": bool(manifest.get("passed")),
        "algorithm": manifest.get("algorithm"),
        "circuits": manifest.get("circuits", []),
        "baseline_completed": manifest.get("baseline_completed"),
        "candidate_completed": manifest.get("candidate_completed"),
        "completion_gain": manifest.get("completion_gain"),
        "repaired_circuits": manifest.get("repaired_circuits", []),
        "unresolved_circuits": manifest.get("unresolved_circuits", []),
        "route_gate_passed": bool(manifest.get("route_gate_passed")),
        "trace_replay_passed": bool(manifest.get("trace_replay_passed")),
        "trace_replay_rows": manifest.get("trace_replay_rows"),
        "expected_trace_replay_rows": manifest.get("expected_trace_replay_rows"),
        "gpu_recommended": bool(manifest.get("gpu_recommended")),
        "route": {
            "completion_gain": route_gate.get("completion_gain"),
            "baseline": _summary_from_gate(route_gate, "baseline"),
            "candidate": _summary_from_gate(route_gate, "candidate"),
            "improvements": route_gate.get("improvements", []),
            "regressions": route_gate.get("regressions", []),
        },
        "trace_replay": {
            "passed": bool(trace_gate.get("passed")),
            "rows": trace_gate.get("rows"),
            "path": "results/npqr_stage21_suffix_repair_gate/suffix_repair_completed_traces.jsonl",
            "results": trace_gate.get("results", []),
        },
        "repair_diagnostics": manifest.get("repair_diagnostics", []),
        "notes": manifest.get("notes", []),
    }


def _stage22_suffix_training_readiness_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE22_SUFFIX_TRAINING_READINESS_DIR / "manifest.json") or {}
    dataset_manifest = manifest.get("dataset_manifest") or {}
    gpu_decision = manifest.get("gpu_decision") or {}
    tiny = manifest.get("tiny_overfit") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage22_suffix_training_readiness",
        "dataset": "results/npqr_stage22_suffix_training_readiness/combined_stage22_suffix_samples.jsonl",
        "base_model": manifest.get("base_model"),
        "raw_samples": manifest.get("raw_samples", 0),
        "materialized_samples": manifest.get("materialized_samples", 0),
        "accepted_traces": dataset_manifest.get("accepted_traces", 0),
        "rejected_traces": dataset_manifest.get("rejected_traces", []),
        "samples": dataset_manifest.get("samples", 0),
        "repaired_suffix_samples": dataset_manifest.get("repaired_suffix_samples", 0),
        "trace_counts_by_stage": dataset_manifest.get("trace_counts_by_stage", {}),
        "sample_counts_by_role": dataset_manifest.get("sample_counts_by_role", {}),
        "stage21_suffix_repair": dataset_manifest.get("stage21_suffix_repair", {}),
        "base_teacher_fit": manifest.get("base_teacher_fit", {}),
        "stage21_base_teacher_fit": manifest.get("stage21_base_teacher_fit", {}),
        "tiny_overfit": {
            "samples": tiny.get("samples"),
            "epochs": tiny.get("epochs"),
            "lr": tiny.get("lr"),
            "before": tiny.get("before", {}),
            "after": tiny.get("after", {}),
            "delta": tiny.get("delta", {}),
        },
        "gpu_decision": {
            "gpu_recommended": bool(gpu_decision.get("gpu_recommended")),
            "long_training_recommended": bool(gpu_decision.get("long_training_recommended")),
            "decision": gpu_decision.get("decision"),
            "reason": gpu_decision.get("reason"),
        },
        "notes": manifest.get("notes", []),
    }


def _stage23_gpu_sweep_plan_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE23_GPU_SWEEP_PLAN_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage23_gpu_sweep_plan",
        "runner": "results/npqr_stage23_gpu_sweep_plan/run_stage23_gpu_sweep.sh",
        "stage22_manifest": "results/npqr_stage22_suffix_training_readiness/manifest.json",
        "dataset": "results/npqr_stage22_suffix_training_readiness/combined_stage22_suffix_samples.jsonl",
        "decision": manifest.get("decision"),
        "allowed_by_stage22": bool(manifest.get("allowed_by_stage22")),
        "long_training_allowed": bool(manifest.get("long_training_allowed")),
        "overnight_allowed": bool(manifest.get("overnight_allowed")),
        "candidate_count": manifest.get("candidate_count", 0),
        "candidates": manifest.get("candidates", []),
        "promotion_policy": manifest.get("promotion_policy", {}),
        "stop_rules": manifest.get("stop_rules", []),
        "stage22_gpu_decision": manifest.get("stage22_gpu_decision", {}),
        "notes": manifest.get("notes", []),
    }


def _stage23_gpu_sweep_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE23_GPU_SWEEP_SUMMARY_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage23_gpu_sweep_summary",
        "plan": "results/npqr_stage23_gpu_sweep_plan/manifest.json",
        "sweep_root": manifest.get("sweep_root"),
        "decision": manifest.get("decision"),
        "candidate_count": manifest.get("candidate_count", 0),
        "pending_count": manifest.get("pending_count", 0),
        "completed_count": manifest.get("completed_count", 0),
        "promoted_count": manifest.get("promoted_count", 0),
        "invalid_promotion_count": manifest.get("invalid_promotion_count", 0),
        "best_candidate": manifest.get("best_candidate"),
        "candidate_results": manifest.get("candidate_results", []),
        "long_training_allowed": bool(manifest.get("long_training_allowed")),
        "overnight_allowed": bool(manifest.get("overnight_allowed")),
        "promotion_policy": manifest.get("promotion_policy", {}),
        "notes": manifest.get("notes", []),
    }


def _stage24_training_go_no_go_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE24_TRAINING_GO_NO_GO_DIR / "manifest.json") or {}
    preflight = _read_json(STAGE24_TRAINING_GO_NO_GO_DIR / "cloud_preflight.json") or {}
    handoff = manifest.get("cloud_handoff") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage24_training_go_no_go",
        "handoff": "results/npqr_stage24_training_go_no_go/STAGE24_GPU_HANDOFF.md",
        "stage22_manifest": "results/npqr_stage22_suffix_training_readiness/manifest.json",
        "stage23_plan": "results/npqr_stage23_gpu_sweep_plan/manifest.json",
        "stage23_summary": "results/npqr_stage23_gpu_sweep_summary/manifest.json",
        "decision": manifest.get("decision"),
        "reason": manifest.get("reason"),
        "bounded_gpu_sweep_recommended": bool(manifest.get("bounded_gpu_sweep_recommended")),
        "long_training_allowed": bool(manifest.get("long_training_allowed")),
        "overnight_allowed": bool(manifest.get("overnight_allowed")),
        "candidate_count": manifest.get("candidate_count", 0),
        "pending_count": manifest.get("pending_count", 0),
        "promoted_count": manifest.get("promoted_count", 0),
        "training_readiness": manifest.get("training_readiness", {}),
        "cloud_preflight": {
            "path": "results/npqr_stage24_training_go_no_go/cloud_preflight.json",
            "passed": bool(preflight.get("passed")),
            "runner_relative_ok": bool(preflight.get("runner_relative_ok")),
            "bounded_policy_ok": bool(preflight.get("bounded_policy_ok")),
            "command_boundary_ok": bool(preflight.get("command_boundary_ok")),
            "forbidden_hits": preflight.get("forbidden_hits", []),
            "stage23_summary_decision": preflight.get("stage23_summary_decision"),
        },
        "go_no_go_rules": manifest.get("go_no_go_rules", []),
        "next_actions": manifest.get("next_actions", []),
        "cloud_handoff": {
            "required_paths": handoff.get("required_paths", []),
            "script_dependencies": handoff.get("script_dependencies", []),
            "path_policy": handoff.get("path_policy", {}),
            "exclude_patterns": handoff.get("exclude_patterns", []),
            "preflight_commands": handoff.get("preflight_commands", []),
            "run_commands": handoff.get("run_commands", []),
            "fetch_back": handoff.get("fetch_back", []),
            "forbidden_commands": handoff.get("forbidden_commands", []),
        },
        "notes": manifest.get("notes", []),
    }


def _stage25_post_sweep_decision_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE25_POST_SWEEP_DECISION_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage25_post_sweep_decision",
        "decision_doc": "results/npqr_stage25_post_sweep_decision/STAGE25_POST_SWEEP_DECISION.md",
        "stage23_summary": "results/npqr_stage23_gpu_sweep_summary/manifest.json",
        "stage24_manifest": "results/npqr_stage24_training_go_no_go/manifest.json",
        "decision": manifest.get("decision"),
        "reason": manifest.get("reason"),
        "long_training_allowed": bool(manifest.get("long_training_allowed")),
        "overnight_allowed": bool(manifest.get("overnight_allowed")),
        "stage23_pending_count": manifest.get("stage23_pending_count", 0),
        "stage23_promoted_count": manifest.get("stage23_promoted_count", 0),
        "stage23_invalid_promotion_count": manifest.get("stage23_invalid_promotion_count", 0),
        "stage23_seed_valid": bool(manifest.get("stage23_seed_valid")),
        "stage24_decision": manifest.get("stage24_decision"),
        "best_candidate": manifest.get("best_candidate"),
        "bounded_long_training_blueprint": manifest.get("bounded_long_training_blueprint"),
        "next_actions": manifest.get("next_actions", []),
        "notes": manifest.get("notes", []),
    }


def _stage26_distillation_audit_summary() -> dict[str, Any]:
    manifest = _read_json(STAGE26_DISTILLATION_AUDIT_DIR / "manifest.json") or {}
    return {
        "schema": manifest.get("schema"),
        "source_dir": "results/npqr_stage26_distillation_audit",
        "decision_doc": "results/npqr_stage26_distillation_audit/STAGE26_DISTILLATION_AUDIT.md",
        "decision": manifest.get("decision"),
        "reason": manifest.get("reason"),
        "long_training_allowed": bool(manifest.get("long_training_allowed")),
        "overnight_allowed": bool(manifest.get("overnight_allowed")),
        "same_corpus_training_allowed": bool(manifest.get("same_corpus_training_allowed")),
        "checkpoint_promotion_allowed": bool(manifest.get("checkpoint_promotion_allowed")),
        "public_default_backend": manifest.get("public_default_backend"),
        "root_causes": manifest.get("root_causes", {}),
        "hard_case_analysis": manifest.get("hard_case_analysis", {}),
        "recommended_next_experiment": manifest.get("recommended_next_experiment", {}),
        "stage23_failure_summary": manifest.get("stage23_failure_summary", {}),
        "notes": manifest.get("notes", []),
    }


def build_npqr_evidence_manifest() -> dict[str, Any]:
    """Build the current NPQR evidence manifest from checked local JSON gates."""
    stage7_manifest = _read_json(STAGE7_DIR / "manifest.json") or {}
    stage7_gate = _read_json(STAGE7_DIR / "route_gate.json") or {}
    stage7_trace_gate = _read_json(STAGE7_DIR / "trace_replay_gate.json") or {}
    stage8_manifest = _read_json(STAGE8_DIR / "manifest.json") or {}
    stage8_raw_gate = _read_json(STAGE8_DIR / "raw_route_gate.json") or {}
    stage8_quick_gate = _read_json(STAGE8_DIR / "quick_route_gate.json") or {}
    stage8_qap_gate = _read_json(STAGE8_DIR / "qap_route_gate.json") or {}
    stage8_trace_gate = _read_json(STAGE8_DIR / "candidate_trace_replay_gate.json") or {}

    uniform_attempt = _stage8_attempt_summary(
        attempt_id="uniform_bc",
        label="Stage8 uniform BC finetune",
        source_dir=STAGE8_DIR,
        dataset_manifest_name="combined_stage8_samples.manifest.json",
        dataset_path="results/npqr_stage8_metric_sprint/combined_stage8_samples.jsonl",
        metrics_name="stage8_finetune_metrics.json",
        trace_path="results/npqr_stage8_metric_sprint/candidate_qap_completed_traces.jsonl",
    )
    hardcase_attempt = _stage8_attempt_summary(
        attempt_id="hardcase_weighted_bc",
        label="Stage8 hard-case weighted BC finetune",
        source_dir=STAGE8_HARDCASE_DIR,
        dataset_manifest_name="combined_stage8_hardcase_samples.manifest.json",
        dataset_path="results/npqr_stage8_hardcase_sprint/combined_stage8_hardcase_samples.jsonl",
        metrics_name="stage8_hardcase_finetune_metrics.json",
        trace_path="results/npqr_stage8_hardcase_sprint/candidate_qap_completed_traces.jsonl",
    )
    stage9_teacher_scan = _stage9_teacher_scan_summary()
    stage9_mixed_dataset = _stage9_mixed_dataset_summary()
    stage9_extension_scan = _stage9_extension_scan_summary()
    stage9_mapping_probe = _stage9_mapping_probe_summary()
    stage10_mapping_selector = _stage10_mapping_selector_summary()
    stage11_selector_runtime = _stage11_selector_runtime_summary()
    stage12_selector_boundary = _stage12_selector_boundary_summary()
    stage13_adaptive_selector_boundary = _stage13_adaptive_selector_boundary_summary()
    stage14_adaptive_dataset = _stage14_adaptive_dataset_summary()
    stage15_finetune_gate = _stage15_finetune_gate_summary()
    stage16_hardcase_scout = _stage16_hardcase_scout_summary()
    stage17_hardcase_dataset = _stage17_hardcase_dataset_summary()
    stage18_finetune_gate = _stage18_finetune_gate_summary()
    stage19_training_diagnostics = _stage19_training_diagnostics_summary()
    stage20_ghz10_stall_diagnostics = _stage20_ghz10_stall_diagnostics_summary()
    stage21_suffix_repair_gate = _stage21_suffix_repair_gate_summary()
    stage22_suffix_training_readiness = _stage22_suffix_training_readiness_summary()
    stage23_gpu_sweep_plan = _stage23_gpu_sweep_plan_summary()
    stage23_gpu_sweep_summary = _stage23_gpu_sweep_summary()
    stage24_training_go_no_go = _stage24_training_go_no_go_summary()
    stage25_post_sweep_decision = _stage25_post_sweep_decision_summary()
    stage26_distillation_audit = _stage26_distillation_audit_summary()
    stage9_mixed_attempt = _stage8_attempt_summary(
        attempt_id="stage9_mixed_bc",
        label="Stage9 mixed QAP/legacy BC finetune",
        source_dir=STAGE9_METRIC_SPRINT_DIR,
        dataset_manifest_name="../npqr_stage9_mixed_dataset/combined_stage9_mixed_samples.manifest.json",
        dataset_path="results/npqr_stage9_mixed_dataset/combined_stage9_mixed_samples.jsonl",
        metrics_name="stage9_mixed_finetune_metrics.json",
        trace_path="results/npqr_stage9_metric_sprint/candidate_qap_completed_traces.jsonl",
    )

    return {
        "schema": "npqr_evidence_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_claim": (
            "NPQR is the default neural-assisted router for the online API/MCP path; "
            "SABRE remains the comparison baseline and NPQR is not claimed to generally beat SABRE."
        ),
        "stable_public_default": {
            "backend": "npqr",
            "algorithm": "npqr_neural_selector_suffix_v1",
            "components": ["neural_beam", "mapping_selector", "bounded_suffix_repair"],
            "baseline": {
                "backend": "sabre",
                "heuristic": "lookahead",
                "seed": 42,
                "trials": 1,
            },
            "reason": "course-facing route uses the self-developed neural-assisted pipeline while keeping SABRE as the measured baseline",
        },
        "npqr_boundary": {
            "self_developed": [
                "GNN/RL policy-value router",
                "NPQR NeuralBeam planner",
                "trace replay gate",
                "route-gated training loop",
            ],
            "borrowed_or_baseline": [
                "SABRE/LightSABRE as baseline or teacher",
                "QAP-style initial-mapping scoring as lightweight offline assistance",
            ],
            "not_claimed": [
                "NPQR does not generally outperform SABRE on the full benchmark.",
                "QAP-style mapping is not a full QAP solver.",
                "Stage8/Stage15/Stage18 finetunes are not promoted because raw NeuralBeam did not improve.",
            ],
        },
        "stage7": {
            "label": "Stage7 QAP-style mapping + NPQR NeuralBeam",
            "source_dir": "results/npqr_stage7_metric_sprint_recovered_trace",
            "model": stage7_manifest.get("model"),
            "route_gate_passed": bool(stage7_manifest.get("route_gate_passed")),
            "trace_replay_passed": bool(stage7_manifest.get("trace_replay_passed")),
            "trace_replay_rows": stage7_manifest.get("trace_replay_rows"),
            "completion_gain": stage7_gate.get("completion_gain"),
            "baseline": _summary_from_gate(stage7_gate, "baseline"),
            "candidate": _summary_from_gate(stage7_gate, "candidate"),
            "improvements": stage7_gate.get("improvements", []),
            "regressions": stage7_gate.get("regressions", []),
            "trace_replay": {
                "passed": bool(stage7_trace_gate.get("passed")),
                "rows": stage7_trace_gate.get("rows"),
                "path": "results/npqr_stage7_metric_sprint_recovered_trace/qap_completed_traces.jsonl",
            },
        },
        "stage8": {
            "label": "Stage8 short BC finetune",
            "source_dir": "results/npqr_stage8_metric_sprint",
            "candidate_model": stage8_manifest.get("candidate_model"),
            "decision": stage8_manifest.get("decision"),
            "promote_checkpoint": bool(stage8_manifest.get("promote_checkpoint")),
            "raw_gate_passed": bool(stage8_manifest.get("raw_gate_passed")),
            "quick_gate_passed": bool(stage8_manifest.get("quick_gate_passed")),
            "qap_gate_passed": bool(stage8_manifest.get("qap_gate_passed")),
            "trace_replay_passed": bool(stage8_manifest.get("trace_replay_passed")),
            "trace_replay_rows": stage8_manifest.get("trace_replay_rows"),
            "raw": {
                "completion_gain": stage8_raw_gate.get("completion_gain"),
                "baseline": _summary_from_gate(stage8_raw_gate, "baseline"),
                "candidate": _summary_from_gate(stage8_raw_gate, "candidate"),
                "improvements": stage8_raw_gate.get("improvements", []),
                "regressions": stage8_raw_gate.get("regressions", []),
            },
            "quick": {
                "passed": bool(stage8_quick_gate.get("passed")),
                "candidate": _summary_from_gate(stage8_quick_gate, "candidate"),
                "regressions": stage8_quick_gate.get("regressions", []),
            },
            "qap": {
                "passed": bool(stage8_qap_gate.get("passed")),
                "candidate": _summary_from_gate(stage8_qap_gate, "candidate"),
                "regressions": stage8_qap_gate.get("regressions", []),
            },
            "trace_replay": {
                "passed": bool(stage8_trace_gate.get("passed")),
                "rows": stage8_trace_gate.get("rows"),
                "path": "results/npqr_stage8_metric_sprint/candidate_qap_completed_traces.jsonl",
            },
            "dataset": _dataset_summary(),
            "training": _training_summary(),
        },
        "stage8_attempts": [uniform_attempt, hardcase_attempt, stage9_mixed_attempt],
        "stage9_teacher_scan": stage9_teacher_scan,
        "stage9_mixed_dataset": stage9_mixed_dataset,
        "stage9_extension_scan": stage9_extension_scan,
        "stage9_mapping_probe": stage9_mapping_probe,
        "stage10_mapping_selector": stage10_mapping_selector,
        "stage11_selector_runtime": stage11_selector_runtime,
        "stage12_selector_boundary": stage12_selector_boundary,
        "stage13_adaptive_selector_boundary": stage13_adaptive_selector_boundary,
        "stage14_adaptive_dataset": stage14_adaptive_dataset,
        "stage15_finetune_gate": stage15_finetune_gate,
        "stage16_hardcase_scout": stage16_hardcase_scout,
        "stage17_hardcase_dataset": stage17_hardcase_dataset,
        "stage18_finetune_gate": stage18_finetune_gate,
        "stage19_training_diagnostics": stage19_training_diagnostics,
        "stage20_ghz10_stall_diagnostics": stage20_ghz10_stall_diagnostics,
        "stage21_suffix_repair_gate": stage21_suffix_repair_gate,
        "stage22_suffix_training_readiness": stage22_suffix_training_readiness,
        "stage23_gpu_sweep_plan": stage23_gpu_sweep_plan,
        "stage23_gpu_sweep_summary": stage23_gpu_sweep_summary,
        "stage24_training_go_no_go": stage24_training_go_no_go,
        "stage25_post_sweep_decision": stage25_post_sweep_decision,
        "stage26_distillation_audit": stage26_distillation_audit,
        "next_algorithm_focus": [
            "Stage9 mixed BC finetune did not improve raw NeuralBeam completion.",
            "Per-mapping probe found replayable QAP-candidate traces for ghz10 and vqe10, while legacy candidates stayed incomplete.",
            "Stage10 offline selector ranks completed ghz10/vqe10 QAP-candidate mappings top-1 using structure-only distance features.",
            "Stage11 selector runtime top-2 passes checked7, quick, and 7/7 trace replay; it is now one component of the NPQR default route.",
            "Stage12 expanded MQT/fallback boundary holds full top-2 due to vqe_10 regression, while targeted vqe_10 top-4 passes.",
            "Stage13 adaptive selector top-k restores the expanded boundary to raw 6/8 without regressions by using top-4 only after top-2 incomplete routes.",
            "Stage14 combines Stage7 QAP, Stage9 legacy, and Stage13 adaptive replayable traces into a 95-sample teacher corpus.",
            "Stage15 short finetune passes quick, checked7, expanded adaptive, and trace replay gates, but is held because raw checked7 completion stays 5/7.",
            "Stage16 hard-case scout finds replayable qaoa_10 traces with wider selector search, while ghz_10 remains unresolved.",
            "Stage17 converts Stage16 qaoa_10 traces into a weighted hard-case teacher corpus with 18 traces and 154 samples.",
            "Stage18 short finetune from the Stage17 corpus passes all route gates but is held because raw checked7 completion stays 5/7.",
            "Stage19 diagnostics show tiny overfit can learn, but Stage18 barely changes full-corpus teacher fit; do not open GPU for same-form BC.",
            "Stage20 ghz_10 stall diagnostics find a 3-action completion suffix after the 10/12-gate stall; fix search/suffix repair before GPU.",
            "Stage21 bounded suffix repair completes qaoa_10 and ghz_10, moving the expanded boundary from 6/8 to 8/8 with trace replay 8/8.",
            "Stage22 converts Stage21 repaired traces into a 226-sample suffix corpus and supports only a bounded GPU sweep, not overnight long training.",
            "Stage23 prepares a 3-candidate bounded GPU sweep runbook; no checkpoint can be promoted without raw checked7 gain and trace replay.",
            "Stage23 bounded GPU sweep completed all 3 candidates; all were held because raw checked7 stayed 5/7 and expanded adaptive regressed qft_10/qaoa_5.",
            "Stage24 go/no-go allowed only the bounded Stage23 sweep and still forbids overnight or broad long training.",
            "Stage25 post-sweep decision holds long training because Stage23 produced no promoted checkpoint.",
            "Stage26 distillation audit attributes the failure to mapping coverage, teacher-policy divergence, beam scoring, action-only BC, unused value loss, and mixed teacher distributions.",
            "Next useful work is changing objective or architecture, not more same-corpus GPU training.",
        ],
        "mcp_tools": [
            "qcompiler_status",
            "list_examples",
            "compile_npqr",
            "compile_sabre",
            "compile_qasm",
            "get_benchmarks",
            "get_npqr_boundary",
            "get_npqr_stage7_evidence",
        ],
    }


def write_npqr_evidence_manifest(path: Path = DEFAULT_EVIDENCE_PATH) -> dict[str, Any]:
    manifest = build_npqr_evidence_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_npqr_evidence_manifest(path: Path = DEFAULT_EVIDENCE_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_npqr_evidence_manifest()
