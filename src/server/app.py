"""
Public local API for the ZJU Quantum Compiler playground.

Run locally:
    uvicorn src.server.app:app --reload --port 8765

The GitHub Pages site works without this server. When the server is running on
localhost, the page can call these endpoints for live SABRE and AI routing.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreSwap,
    TrivialLayout,
)

from src.benchmarks.topologies import get_topology, get_topology_info
from src.cli import _DEFAULT_MODEL, _TOPOLOGY_ALIAS
from src.evidence import load_npqr_evidence_manifest

if TYPE_CHECKING:
    from src.compiler.pass_manager import AIRouter
    from src.compiler.npqr_runtime import NPQRRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = PROJECT_ROOT / "models" / "v14_tokyo20" / "eval_report_mqt.json"
DEFAULT_MODEL = _DEFAULT_MODEL
DEFAULT_NPQR_MODEL = (
    PROJECT_ROOT
    / "models"
    / "npqr_overnight_20260603"
    / "wave2_stage2_e120_lr5e5_s51_h02.pt"
)
BASIS_GATES = ["cx", "id", "rz", "sx", "x", "swap"]
MAX_INLINE_QASM_CHARS = 8000

EXAMPLES: dict[str, dict[str, str]] = {
    "qft5": {
        "name": "QFT 5",
        "path": "examples/qft5.qasm",
        "description": "A compact quantum Fourier transform example.",
    },
    "ghz5": {
        "name": "GHZ 5",
        "path": "examples/ghz5.qasm",
        "description": "A small entanglement-chain circuit.",
    },
    "qaoa5": {
        "name": "QAOA 5",
        "path": "examples/qaoa5.qasm",
        "description": "One QAOA-style optimization layer.",
    },
    "qft10": {
        "name": "QFT 10",
        "path": "examples/qft10.qasm",
        "description": "A larger dense-routing quantum Fourier transform example.",
    },
    "qaoa10": {
        "name": "QAOA 10",
        "path": "examples/qaoa10.qasm",
        "description": "A 10-qubit QAOA-style optimization layer.",
    },
    "ghz10": {
        "name": "GHZ 10",
        "path": "examples/ghz10.qasm",
        "description": "A 10-qubit entanglement-chain circuit.",
    },
    "vqe10": {
        "name": "VQE-like 10",
        "path": "examples/vqe10.qasm",
        "description": "A 10-qubit RealAmplitudes-style variational ansatz.",
    },
}

app = FastAPI(
    title="ZJU Quantum Compiler Playground API",
    version="0.14.2",
    description=(
        "Local API for SABRE compilation, experimental AIRouter checks, "
        "and static benchmark summaries."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StatusResponse(BaseModel):
    version: str
    status: str
    default_topology: str
    default_backend: str
    available_topologies: list[str]
    default_model: str
    model_exists: bool
    ai_loadable: bool
    ai_status: str
    npqr_model: str
    npqr_model_exists: bool
    npqr_loadable: bool
    npqr_status: str
    benchmark_report: str


class ExampleInfo(BaseModel):
    id: str
    name: str
    path: str
    description: str


class CompileRequest(BaseModel):
    example: str | None = Field(default=None, description="Example id, such as qft5.")
    qasm: str | None = Field(default=None, description="Inline OpenQASM 2 source.")
    backend: Literal["npqr", "sabre", "ai"] = "npqr"
    heuristic: Literal["basic", "lookahead", "decay"] = "lookahead"
    topology: str = "tokyo"
    max_steps: int = 2000


class CompileResponse(BaseModel):
    status: Literal["OK", "INCOMPLETE", "N/A"]
    backend: Literal["npqr", "sabre", "ai"]
    algorithm: str | None = None
    heuristic: Literal["basic", "lookahead", "decay"] | None = None
    topology: str
    circuit_name: str
    input_qubits: int
    input_cx: int
    swaps: int | None
    depth: int | None
    elapsed_ms: float
    compiled_qasm: str | None = None
    model_path: str | None = None
    components: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    message: str | None = None


def _resolve_topology(name: str):
    canon = _TOPOLOGY_ALIAS.get(name, name)
    return canon, get_topology(canon)


def _load_example(example_id: str) -> QuantumCircuit:
    spec = EXAMPLES.get(example_id)
    if not spec:
        known = ", ".join(sorted(EXAMPLES))
        raise HTTPException(status_code=404, detail=f"Unknown example: {example_id}. Known: {known}")
    path = PROJECT_ROOT / spec["path"]
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Example file missing: {spec['path']}")
    circuit = QuantumCircuit.from_qasm_file(str(path))
    circuit.name = example_id
    return circuit


def _load_request_circuit(req: CompileRequest) -> QuantumCircuit:
    if req.qasm and req.example:
        raise HTTPException(status_code=400, detail="Provide either qasm or example, not both.")
    if req.qasm:
        if len(req.qasm) > MAX_INLINE_QASM_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Inline OpenQASM 2 input is limited to {MAX_INLINE_QASM_CHARS} characters.",
            )
        try:
            qc = qasm2.loads(req.qasm)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid OpenQASM 2 input: {exc}") from exc
        qc.name = "inline_qasm"
        return qc
    return _load_example(req.example or "qft5")


def _compile_sabre(
    circuit: QuantumCircuit,
    coupling_map,
    heuristic: Literal["basic", "lookahead", "decay"],
) -> QuantumCircuit:
    layout_and_route = PassManager(
        [
            TrivialLayout(coupling_map),
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(coupling_map, heuristic=heuristic, seed=42, trials=1),
        ]
    )
    routed = layout_and_route.run(circuit)
    return transpile(
        routed,
        basis_gates=BASIS_GATES,
        optimization_level=0,
        seed_transpiler=42,
    )


def _sabre_summary(
    circuit: QuantumCircuit,
    coupling_map,
    heuristic: Literal["basic", "lookahead", "decay"],
) -> dict[str, Any]:
    started = time.perf_counter()
    compiled = _compile_sabre(circuit, coupling_map, heuristic)
    ops = dict(compiled.count_ops())
    return {
        "backend": "sabre",
        "heuristic": heuristic,
        "status": "OK",
        "swaps": ops.get("swap", 0),
        "depth": compiled.depth(),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


@lru_cache(maxsize=4)
def _router_for(topology: str, model_path: str) -> "AIRouter":
    from src.compiler.pass_manager import AIRouter

    _, coupling_map = _resolve_topology(topology)
    return AIRouter(coupling_map, model_path=model_path)


@lru_cache(maxsize=4)
def _npqr_runtime_for(topology: str, model_path: str, max_steps: int) -> "NPQRRuntime":
    from src.compiler.npqr_runtime import NPQRRuntime, NPQRRuntimeConfig

    _, coupling_map = _resolve_topology(topology)
    config = NPQRRuntimeConfig(max_steps=max_steps)
    return NPQRRuntime(coupling_map, model_path=model_path, config=config)


def _ai_load_status(topology: str = "tokyo") -> tuple[bool, str]:
    if not DEFAULT_MODEL.exists():
        return False, "V14 checkpoint file is not present locally."
    try:
        router = _router_for(topology, str(DEFAULT_MODEL))
    except ImportError as exc:
        return False, f"AIRouter dependency is not installed in this REST deployment: {exc.name}."
    if router._has_model:
        return True, "V14 checkpoint can be loaded. Routes are still experimental."
    return False, router.model_load_error or "V14 checkpoint could not be loaded."


def _npqr_load_status(topology: str = "tokyo") -> tuple[bool, str]:
    if not DEFAULT_NPQR_MODEL.exists():
        return False, "NPQR checkpoint file is not present locally."
    try:
        runtime = _npqr_runtime_for(topology, str(DEFAULT_NPQR_MODEL), 45)
    except ImportError as exc:
        return False, f"NPQR dependency is not installed in this REST deployment: {exc.name}."
    if runtime.has_model:
        return True, "NPQR neural selector/search/repair runtime can be loaded."
    return False, runtime.model_load_error or "NPQR checkpoint could not be loaded."


@app.get("/api/status", response_model=StatusResponse)
async def api_status() -> StatusResponse:
    """Return public project and model status without running a benchmark."""
    ai_loadable, ai_message = _ai_load_status()
    npqr_loadable, npqr_message = _npqr_load_status()
    return StatusResponse(
        version="0.14.2",
        status=(
            "Default backend is NPQR neural-assisted routing; SABRE remains the "
            "comparison baseline. V14/V15 raw checkpoints have not beaten SABRE on P1."
        ),
        default_topology="tokyo",
        default_backend="npqr",
        available_topologies=sorted(_TOPOLOGY_ALIAS),
        default_model=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
        model_exists=DEFAULT_MODEL.exists(),
        ai_loadable=ai_loadable,
        ai_status=ai_message,
        npqr_model=str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
        npqr_model_exists=DEFAULT_NPQR_MODEL.exists(),
        npqr_loadable=npqr_loadable,
        npqr_status=npqr_message,
        benchmark_report=str(REPORT_JSON.relative_to(PROJECT_ROOT)),
    )


@app.get("/api/examples", response_model=list[ExampleInfo])
async def api_examples() -> list[ExampleInfo]:
    """Return the checked-in QASM examples used by the public page."""
    return [
        ExampleInfo(id=key, **value)
        for key, value in sorted(EXAMPLES.items())
    ]


@app.get("/api/benchmarks")
async def api_benchmarks() -> dict:
    """Return the latest checked-in V14 P1 benchmark summary."""
    if not REPORT_JSON.exists():
        raise HTTPException(status_code=404, detail="Benchmark report JSON is missing.")
    import json

    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    main_results = [row for row in data["results"] if not row.get("outlier")]
    ai_completed = sum(1 for row in main_results if row.get("ai") and row["ai"].get("completed"))
    sabre_completed = sum(1 for row in main_results if row["sabre"].get("completed"))
    comparable = [
        row for row in main_results
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


@app.get("/api/npqr/evidence")
async def api_npqr_evidence() -> dict:
    """Return the unified NPQR evidence manifest without rerunning training."""
    return load_npqr_evidence_manifest()


@app.post("/api/compile", response_model=CompileResponse)
async def api_compile(req: CompileRequest) -> CompileResponse:
    """Compile a QASM example with NPQR, SABRE, or the experimental AIRouter."""
    topo_name, coupling_map = _resolve_topology(req.topology)
    circuit = _load_request_circuit(req)
    if circuit.num_qubits > coupling_map.size():
        raise HTTPException(
            status_code=400,
            detail=f"Circuit has {circuit.num_qubits} qubits, topology has {coupling_map.size()}.",
        )

    input_cx = dict(circuit.count_ops()).get("cx", 0)
    started = time.perf_counter()

    if req.backend == "npqr":
        if not DEFAULT_NPQR_MODEL.exists():
            return CompileResponse(
                status="N/A",
                backend="npqr",
                algorithm="npqr_neural_selector_suffix_v1",
                heuristic=None,
                topology=topo_name,
                circuit_name=circuit.name or req.example or "inline_qasm",
                input_qubits=circuit.num_qubits,
                input_cx=input_cx,
                swaps=None,
                depth=None,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                model_path=str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
                components={
                    "neural_beam": False,
                    "mapping_selector": True,
                    "suffix_repair": False,
                    "sabre_fallback": False,
                },
                baseline=_sabre_summary(circuit, coupling_map, req.heuristic),
                message="NPQR checkpoint file is missing; SABRE baseline is comparison only.",
            )
        try:
            runtime = _npqr_runtime_for(req.topology, str(DEFAULT_NPQR_MODEL), req.max_steps)
        except ImportError as exc:
            return CompileResponse(
                status="N/A",
                backend="npqr",
                algorithm="npqr_neural_selector_suffix_v1",
                heuristic=None,
                topology=topo_name,
                circuit_name=circuit.name or req.example or "inline_qasm",
                input_qubits=circuit.num_qubits,
                input_cx=input_cx,
                swaps=None,
                depth=None,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                model_path=str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
                components={
                    "neural_beam": False,
                    "mapping_selector": True,
                    "suffix_repair": False,
                    "sabre_fallback": False,
                },
                baseline=_sabre_summary(circuit, coupling_map, req.heuristic),
                message=f"NPQR dependency is not installed in this REST deployment: {exc.name}.",
            )
        result = runtime.compile(circuit)
        compiled_qasm = qasm2.dumps(result.compiled_circuit) if result.completed and result.compiled_circuit else None
        return CompileResponse(
            status=result.status,
            backend="npqr",
            algorithm=result.algorithm,
            heuristic=None,
            topology=topo_name,
            circuit_name=circuit.name or req.example or "inline_qasm",
            input_qubits=circuit.num_qubits,
            input_cx=input_cx,
            swaps=result.total_swaps,
            depth=result.depth if result.completed else None,
            elapsed_ms=result.elapsed_ms,
            compiled_qasm=compiled_qasm,
            model_path=str(DEFAULT_NPQR_MODEL.relative_to(PROJECT_ROOT)),
            components=result.components,
            baseline=_sabre_summary(circuit, coupling_map, req.heuristic),
            message=result.message,
        )

    if req.backend == "sabre":
        compiled = _compile_sabre(circuit, coupling_map, req.heuristic)
        ops = dict(compiled.count_ops())
        return CompileResponse(
            status="OK",
            backend="sabre",
            algorithm="qiskit_sabre_swap",
            heuristic=req.heuristic,
            topology=topo_name,
            circuit_name=circuit.name or req.example or "inline_qasm",
            input_qubits=circuit.num_qubits,
            input_cx=input_cx,
            swaps=ops.get("swap", 0),
            depth=compiled.depth(),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            compiled_qasm=qasm2.dumps(compiled),
            message=f"SABRE completed with the {req.heuristic} heuristic.",
        )

    if not DEFAULT_MODEL.exists():
        return CompileResponse(
            status="N/A",
            backend="ai",
            algorithm="v14_airouter",
            heuristic=None,
            topology=topo_name,
            circuit_name=circuit.name or req.example or "inline_qasm",
            input_qubits=circuit.num_qubits,
            input_cx=input_cx,
            swaps=None,
            depth=None,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model_path=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
            message="AI checkpoint file is missing; SABRE remains available.",
        )

    try:
        router = _router_for(req.topology, str(DEFAULT_MODEL))
    except ImportError as exc:
        return CompileResponse(
            status="N/A",
            backend="ai",
            algorithm="v14_airouter",
            heuristic=None,
            topology=topo_name,
            circuit_name=circuit.name or req.example or "inline_qasm",
            input_qubits=circuit.num_qubits,
            input_cx=input_cx,
            swaps=None,
            depth=None,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model_path=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
            message=f"AIRouter dependency is not installed in this REST deployment: {exc.name}.",
        )
    if not router._has_model:
        return CompileResponse(
            status="N/A",
            backend="ai",
            algorithm="v14_airouter",
            heuristic=None,
            topology=topo_name,
            circuit_name=circuit.name or req.example or "inline_qasm",
            input_qubits=circuit.num_qubits,
            input_cx=input_cx,
            swaps=None,
            depth=None,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model_path=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
            message=router.model_load_error or "AI checkpoint could not be loaded.",
        )

    result = router.route_count_only(circuit, max_steps=req.max_steps)
    return CompileResponse(
        status="OK" if result["done"] else "INCOMPLETE",
        backend="ai",
        algorithm="v14_airouter",
        heuristic=None,
        topology=topo_name,
        circuit_name=circuit.name or req.example or "inline_qasm",
        input_qubits=circuit.num_qubits,
        input_cx=input_cx,
        swaps=result["ai_swaps"],
        depth=None,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        model_path=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
        message="AIRouter route_count_only ran; incomplete routes are reported honestly.",
    )


@app.get("/api/topology/{name}")
async def api_topology(name: str) -> dict:
    """Return topology metadata for small UI checks."""
    topo_name, coupling_map = _resolve_topology(name)
    return {
        "name": topo_name,
        "info": get_topology_info(coupling_map),
        "edges": coupling_map.get_edges(),
    }
