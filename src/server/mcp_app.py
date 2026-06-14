"""Remote HTTP MCP entrypoint for the ZJU Quantum Compiler.

The server is intentionally read-only: it exposes bounded SABRE compilation and
checked-in evidence, but never starts training jobs or writes GitHub state.
"""
from __future__ import annotations

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
    FrontierPruningPolicy,
    FrontierTriggerProfile,
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
    include_compiled_qasm: bool = True,
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
        "compiled_qasm": qasm2.dumps(compiled) if include_compiled_qasm else None,
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
    frontier_pruning_policy: FrontierPruningPolicy | None = None,
    frontier_trigger_profile: FrontierTriggerProfile | None = None,
    include_compiled_qasm: bool = True,
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
    runtime = _npqr_runtime_for(
        topology,
        str(DEFAULT_NPQR_MODEL),
        int(max_steps),
        frontier_pruning_policy,
        frontier_trigger_profile,
    )
    result = runtime.compile(circuit)
    compiled_qasm = (
        qasm2.dumps(result.compiled_circuit)
        if include_compiled_qasm and result.completed and result.compiled_circuit
        else None
    )
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
    frontier_pruning_policy: FrontierPruningPolicy | None = None,
    frontier_trigger_profile: FrontierTriggerProfile | None = None,
    include_compiled_qasm: bool = True,
) -> dict[str, Any]:
    """Compile inline OpenQASM 2 text through the bounded public route."""
    if len(qasm) > MAX_INLINE_QASM_CHARS:
        raise ValueError(f"Inline OpenQASM 2 input is limited to {MAX_INLINE_QASM_CHARS} characters.")
    if not qasm.strip():
        raise ValueError("qasm must not be empty")
    try:
        circuit = qasm2.loads(
            qasm,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid OpenQASM 2 input: {exc}") from exc
    circuit.name = "inline_qasm"
    if backend == "sabre":
        return _compile_sabre_payload(
            circuit=circuit,
            circuit_name="inline_qasm",
            heuristic=heuristic,
            topology=topology,
            include_compiled_qasm=include_compiled_qasm,
        )
    return _compile_npqr_payload(
        circuit=circuit,
        circuit_name="inline_qasm",
        heuristic=heuristic,
        topology=topology,
        max_steps=max_steps,
        frontier_pruning_policy=frontier_pruning_policy,
        frontier_trigger_profile=frontier_trigger_profile,
        include_compiled_qasm=include_compiled_qasm,
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
    npqr_loadable, npqr_message = _npqr_load_status()
    manifest = load_npqr_evidence_manifest()
    return {
        "version": "0.14.2",
        "status": "Default public route is NPQR neural-assisted routing; SABRE remains the baseline.",
        "default_topology": "tokyo",
        "default_backend": "npqr",
        "default_model": str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
        "model": str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
        "model_exists": DEFAULT_NPQR_MODEL.exists(),
        "model_loadable": npqr_loadable,
        "model_status": npqr_message,
        "mcp_endpoint": "/mcp",
        "algorithm": manifest["project_claim"],
        "claims": manifest["claims"],
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
    frontier_pruning_policy: FrontierPruningPolicy | None = None,
    frontier_trigger_profile: FrontierTriggerProfile | None = None,
    include_compiled_qasm: bool = True,
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
        frontier_pruning_policy=frontier_pruning_policy,
        frontier_trigger_profile=frontier_trigger_profile,
        include_compiled_qasm=include_compiled_qasm,
    )


@mcp.tool()
def compile_qasm(
    qasm: str,
    heuristic: str = "lookahead",
    topology: str = "tokyo",
    backend: Literal["npqr", "sabre"] = "npqr",
    max_steps: int = 45,
    frontier_pruning_policy: FrontierPruningPolicy | None = None,
    frontier_trigger_profile: FrontierTriggerProfile | None = None,
    include_compiled_qasm: bool = True,
) -> dict[str, Any]:
    """Compile user-provided OpenQASM 2 text and return routed QASM."""
    return compile_qasm_payload(
        qasm=qasm,
        heuristic=heuristic,
        topology=topology,
        backend=backend,
        max_steps=max_steps,
        frontier_pruning_policy=frontier_pruning_policy,
        frontier_trigger_profile=frontier_trigger_profile,
        include_compiled_qasm=include_compiled_qasm,
    )


@mcp.tool()
def get_benchmarks() -> dict[str, Any]:
    """Return public benchmark and claim boundaries."""
    manifest = load_npqr_evidence_manifest()
    return {
        "summary": {
            "default_backend": manifest["default_route"]["backend"],
            "comparison_baseline": manifest["default_route"]["comparison_baseline"],
            "sabre_fallback": manifest["default_route"]["sabre_fallback"],
            "representative_10_20_basic": manifest["representative_10_20_basic"]["summary"],
            "scale_smoke_30_50_basic": manifest["scale_smoke_30_50_basic"]["summary"],
            "known_scale_boundary": manifest["scale_smoke_30_50_basic"]["known_boundary"],
            "large_scale_boundary": {
                "max_completed_qubits": manifest["large_scale_boundary"]["max_completed_qubits"],
                "max_sabre_basic_win_qubits": manifest["large_scale_boundary"][
                    "max_sabre_basic_win_qubits"
                ],
            },
        },
        "algorithm_components": manifest["algorithm_components"],
        "claims": manifest["claims"],
    }


@mcp.tool()
def get_npqr_boundary() -> dict[str, Any]:
    """Return explicit NPQR claims, borrowed components, and non-claims."""
    manifest = load_npqr_evidence_manifest()
    return {
        "project_claim": manifest["project_claim"],
        "default_route": manifest["default_route"],
        "baseline": manifest["baseline"],
        "claims": manifest["claims"],
    }


@mcp.tool()
def get_algorithm_evidence() -> dict[str, Any]:
    """Return public NPQR algorithm evidence and course-report mapping."""
    manifest = load_npqr_evidence_manifest()
    return manifest


app = mcp.streamable_http_app()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
