"""Remote HTTP MCP entrypoint for the ZJU Quantum Compiler.

The server is intentionally read-only: it exposes bounded SABRE compilation and
checked-in evidence, but never starts training jobs or writes GitHub state.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings
from qiskit import QuantumCircuit, qasm2
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.evidence import load_npqr_evidence_manifest
from src.server.app import (
    DEFAULT_MODEL,
    DEFAULT_NPQR_MODEL,
    EXAMPLES,
    MAX_INLINE_QASM_CHARS,
    PROJECT_ROOT,
    REPORT_JSON,
    _ai_load_status,
    _compile_sabre,
    _load_example,
    _npqr_load_status,
    _npqr_runtime_for,
    _resolve_topology,
    _sabre_summary,
)

Heuristic = Literal["basic", "lookahead", "decay"]
VALID_HEURISTICS = {"basic", "lookahead", "decay"}


def _mcp_port() -> int:
    return int(os.environ.get("PORT", os.environ.get("QCOMPILER_MCP_PORT", "8000")))


mcp = FastMCP(
    name="zju-quantum-compiler",
    instructions=(
        "Read-only quantum circuit routing tools for OpenQASM 2 and checked-in "
        "project evidence. Use compile_qasm for user-provided circuits."
    ),
    host="0.0.0.0",
    port=_mcp_port(),
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _benchmark_payload() -> dict[str, Any]:
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    main_results = [row for row in data["results"] if not row.get("outlier")]
    ai_completed = sum(1 for row in main_results if row.get("ai") and row["ai"].get("completed"))
    sabre_completed = sum(1 for row in main_results if row["sabre"].get("completed"))
    comparable = [
        row
        for row in main_results
        if row.get("ai")
        and row["ai"].get("completed")
        and row["sabre"].get("completed")
        and row["sabre"].get("swaps", 0) > 0
    ]
    ratios = [row["ai"]["swaps"] / row["sabre"]["swaps"] for row in comparable]
    return {
        "metadata": data["metadata"],
        "summary": {
            "sabre_completed": sabre_completed,
            "sabre_total": len(main_results),
            "ai_completed": ai_completed,
            "ai_total": len(main_results),
            "ai_beats_sabre": sum(1 for row in comparable if row["ai"]["swaps"] < row["sabre"]["swaps"]),
            "comparable_rows": len(comparable),
            "mean_ai_sabre_ratio": sum(ratios) / len(ratios) if ratios else None,
        },
        "results": data["results"],
    }


def _coerce_heuristic(heuristic: str) -> Heuristic:
    if heuristic not in VALID_HEURISTICS:
        raise ValueError("heuristic must be one of: basic, lookahead, decay")
    return heuristic  # type: ignore[return-value]


def _compile_sabre_payload(
    *,
    circuit: QuantumCircuit,
    circuit_name: str,
    heuristic: str = "lookahead",
    topology: str = "tokyo",
) -> dict[str, Any]:
    sabre_heuristic = _coerce_heuristic(heuristic)
    topo_name, coupling_map = _resolve_topology(topology)
    if circuit.num_qubits > coupling_map.size():
        raise ValueError(f"Circuit has {circuit.num_qubits} qubits, topology has {coupling_map.size()}.")

    input_cx = dict(circuit.count_ops()).get("cx", 0)
    started = time.perf_counter()
    compiled = _compile_sabre(circuit, coupling_map, sabre_heuristic)
    ops = dict(compiled.count_ops())
    return {
        "status": "OK",
        "backend": "sabre",
        "heuristic": sabre_heuristic,
        "topology": topo_name,
        "circuit_name": circuit_name,
        "input_qubits": circuit.num_qubits,
        "input_cx": input_cx,
        "swaps": ops.get("swap", 0),
        "depth": compiled.depth(),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "compiled_qasm": qasm2.dumps(compiled),
        "limits": {
            "max_inline_qasm_chars": MAX_INLINE_QASM_CHARS,
            "max_qubits": coupling_map.size(),
        },
    }


def _compile_npqr_payload(
    *,
    circuit: QuantumCircuit,
    circuit_name: str,
    heuristic: str = "lookahead",
    topology: str = "tokyo",
    max_steps: int = 45,
) -> dict[str, Any]:
    sabre_heuristic = _coerce_heuristic(heuristic)
    topo_name, coupling_map = _resolve_topology(topology)
    if circuit.num_qubits > coupling_map.size():
        raise ValueError(f"Circuit has {circuit.num_qubits} qubits, topology has {coupling_map.size()}.")
    input_cx = dict(circuit.count_ops()).get("cx", 0)
    baseline = _sabre_summary(circuit, coupling_map, sabre_heuristic)
    if not DEFAULT_NPQR_MODEL.exists():
        return {
            "status": "N/A",
            "backend": "npqr",
            "algorithm": "npqr_neural_selector_suffix_v1",
            "topology": topo_name,
            "circuit_name": circuit_name,
            "input_qubits": circuit.num_qubits,
            "input_cx": input_cx,
            "swaps": None,
            "depth": None,
            "elapsed_ms": 0.0,
            "compiled_qasm": None,
            "model_path": str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
            "components": {
                "neural_beam": False,
                "mapping_selector": True,
                "suffix_repair": False,
                "sabre_fallback": False,
            },
            "baseline": baseline,
            "message": "NPQR checkpoint file is missing; SABRE baseline is comparison only.",
        }
    runtime = _npqr_runtime_for(topology, str(DEFAULT_NPQR_MODEL), int(max_steps))
    result = runtime.compile(circuit)
    compiled_qasm = qasm2.dumps(result.compiled_circuit) if result.completed and result.compiled_circuit else None
    return {
        "status": result.status,
        "backend": "npqr",
        "algorithm": result.algorithm,
        "topology": topo_name,
        "circuit_name": circuit_name,
        "input_qubits": circuit.num_qubits,
        "input_cx": input_cx,
        "swaps": result.total_swaps,
        "depth": result.depth if result.completed else None,
        "elapsed_ms": result.elapsed_ms,
        "compiled_qasm": compiled_qasm,
        "model_path": str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
        "components": result.components,
        "baseline": baseline,
        "message": result.message,
        "limits": {
            "max_inline_qasm_chars": MAX_INLINE_QASM_CHARS,
            "max_qubits": coupling_map.size(),
        },
    }


def compile_qasm_payload(
    *,
    qasm: str,
    heuristic: str = "lookahead",
    topology: str = "tokyo",
    backend: Literal["npqr", "sabre"] = "npqr",
    max_steps: int = 45,
) -> dict[str, Any]:
    """Compile inline OpenQASM 2 text through the bounded public route."""
    if len(qasm) > MAX_INLINE_QASM_CHARS:
        raise ValueError(f"Inline OpenQASM 2 input is limited to {MAX_INLINE_QASM_CHARS} characters.")
    if not qasm.strip():
        raise ValueError("qasm must not be empty")
    try:
        circuit = qasm2.loads(qasm)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid OpenQASM 2 input: {exc}") from exc
    circuit.name = "inline_qasm"
    if backend == "sabre":
        return _compile_sabre_payload(
            circuit=circuit,
            circuit_name="inline_qasm",
            heuristic=heuristic,
            topology=topology,
        )
    return _compile_npqr_payload(
        circuit=circuit,
        circuit_name="inline_qasm",
        heuristic=heuristic,
        topology=topology,
        max_steps=max_steps,
    )


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "zju-quantum-compiler-mcp",
            "mcp_endpoint": "/mcp",
            "mode": "read_only",
        }
    )


@mcp.tool()
def qcompiler_status() -> dict[str, Any]:
    """Return local project status and honest AI boundary."""
    ai_loadable, ai_message = _ai_load_status()
    npqr_loadable, npqr_message = _npqr_load_status()
    manifest = load_npqr_evidence_manifest()
    return {
        "version": "0.14.2",
        "status": "Default public route is NPQR neural-assisted routing; SABRE remains the baseline.",
        "default_topology": "tokyo",
        "default_backend": "npqr",
        "default_model": str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
        "model_exists": DEFAULT_MODEL.exists(),
        "ai_loadable": ai_loadable,
        "ai_status": ai_message,
        "npqr_model": str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
        "npqr_model_exists": DEFAULT_NPQR_MODEL.exists(),
        "npqr_loadable": npqr_loadable,
        "npqr_status": npqr_message,
        "mcp_endpoint": "/mcp",
        "npqr_decision": manifest["stage8"]["decision"],
        "npqr_boundary": manifest["npqr_boundary"],
    }


@mcp.tool()
def list_examples() -> dict[str, Any]:
    """List checked-in QASM examples available to the compiler."""
    return {"examples": [{"id": key, **value} for key, value in sorted(EXAMPLES.items())]}


@mcp.tool()
def compile_sabre(
    example: str = "qft5",
    heuristic: str = "lookahead",
    topology: str = "tokyo",
) -> dict[str, Any]:
    """Compile a checked-in example with SABRE on IBM Tokyo."""
    if example not in EXAMPLES:
        known = ", ".join(sorted(EXAMPLES))
        raise ValueError(f"Unknown example: {example}. Known examples: {known}")
    return _compile_sabre_payload(
        circuit=_load_example(example),
        circuit_name=example,
        heuristic=heuristic,
        topology=topology,
    )


@mcp.tool()
def compile_npqr(
    example: str = "qft5",
    heuristic: str = "lookahead",
    topology: str = "tokyo",
    max_steps: int = 45,
) -> dict[str, Any]:
    """Compile a checked-in example with the NPQR neural-assisted route."""
    if example not in EXAMPLES:
        known = ", ".join(sorted(EXAMPLES))
        raise ValueError(f"Unknown example: {example}. Known examples: {known}")
    return _compile_npqr_payload(
        circuit=_load_example(example),
        circuit_name=example,
        heuristic=heuristic,
        topology=topology,
        max_steps=max_steps,
    )


@mcp.tool()
def compile_qasm(
    qasm: str,
    heuristic: str = "lookahead",
    topology: str = "tokyo",
    backend: Literal["npqr", "sabre"] = "npqr",
    max_steps: int = 45,
) -> dict[str, Any]:
    """Compile user-provided OpenQASM 2 text and return routed QASM."""
    return compile_qasm_payload(
        qasm=qasm,
        heuristic=heuristic,
        topology=topology,
        backend=backend,
        max_steps=max_steps,
    )


@mcp.tool()
def get_benchmarks() -> dict[str, Any]:
    """Return the checked-in V14 P1 benchmark summary."""
    return _benchmark_payload()


@mcp.tool()
def get_npqr_boundary() -> dict[str, Any]:
    """Return explicit NPQR claims, borrowed components, and non-claims."""
    manifest = load_npqr_evidence_manifest()
    return {
        "project_claim": manifest["project_claim"],
        "stable_public_default": manifest["stable_public_default"],
        "npqr_boundary": manifest["npqr_boundary"],
        "source": str(REPORT_JSON.relative_to(PROJECT_ROOT)),
    }


@mcp.tool()
def get_npqr_stage7_evidence() -> dict[str, Any]:
    """Return Stage7 evidence plus route-gated NPQR follow-up evidence."""
    manifest = load_npqr_evidence_manifest()
    return {
        "stage7": manifest["stage7"],
        "stage8": manifest["stage8"],
        "stage8_attempts": manifest.get("stage8_attempts", []),
        "stage9_teacher_scan": manifest.get("stage9_teacher_scan", {}),
        "stage9_mixed_dataset": manifest.get("stage9_mixed_dataset", {}),
        "stage9_extension_scan": manifest.get("stage9_extension_scan", {}),
        "stage9_mapping_probe": manifest.get("stage9_mapping_probe", {}),
        "stage10_mapping_selector": manifest.get("stage10_mapping_selector", {}),
        "stage11_selector_runtime": manifest.get("stage11_selector_runtime", {}),
        "stage12_selector_boundary": manifest.get("stage12_selector_boundary", {}),
        "stage13_adaptive_selector_boundary": manifest.get("stage13_adaptive_selector_boundary", {}),
        "stage14_adaptive_dataset": manifest.get("stage14_adaptive_dataset", {}),
        "stage15_finetune_gate": manifest.get("stage15_finetune_gate", {}),
        "stage16_hardcase_scout": manifest.get("stage16_hardcase_scout", {}),
        "stage17_hardcase_dataset": manifest.get("stage17_hardcase_dataset", {}),
        "stage18_finetune_gate": manifest.get("stage18_finetune_gate", {}),
        "stage19_training_diagnostics": manifest.get("stage19_training_diagnostics", {}),
        "stage20_ghz10_stall_diagnostics": manifest.get("stage20_ghz10_stall_diagnostics", {}),
        "stage21_suffix_repair_gate": manifest.get("stage21_suffix_repair_gate", {}),
        "stage22_suffix_training_readiness": manifest.get("stage22_suffix_training_readiness", {}),
        "stage23_gpu_sweep_plan": manifest.get("stage23_gpu_sweep_plan", {}),
        "stage23_gpu_sweep_summary": manifest.get("stage23_gpu_sweep_summary", {}),
        "stage24_training_go_no_go": manifest.get("stage24_training_go_no_go", {}),
        "stage25_post_sweep_decision": manifest.get("stage25_post_sweep_decision", {}),
        "next_algorithm_focus": manifest.get("next_algorithm_focus", []),
    }


app = mcp.streamable_http_app()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
