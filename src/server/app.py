"""
Public local API for the ZJU Quantum Compiler playground.

Run locally:
    uvicorn src.server.app:app --reload --port 8765

The GitHub Pages site works without this server. When the server is running on
localhost, the page can call these endpoints for live NPQR and SABRE routing.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, TYPE_CHECKING

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
    from src.compiler.npqr_runtime import NPQRRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _DEFAULT_MODEL
DEFAULT_NPQR_MODEL = PROJECT_ROOT / "models" / "default" / "npqr-default.pt"
BASIS_GATES = ["cx", "id", "rz", "sx", "x", "swap"]
MAX_INLINE_QASM_CHARS = 8000
STANDARD_SABRE_HEURISTIC: Literal["basic", "lookahead", "decay"] = "basic"
FrontierPruningPolicy = Literal["frontier_touch_8"]
FrontierTriggerProfile = Literal["u485_d30_r060_c120"]
REFINED_FRONTIER_TRIGGER_PROFILE = "u485_d30_r060_c120"
REFINED_FRONTIER_TRIGGER_POLICY: FrontierPruningPolicy = "frontier_touch_8"
FRONTIER_TRIGGER_PROFILES: dict[str, dict[str, Any]] = {
    REFINED_FRONTIER_TRIGGER_PROFILE: {
        "frontier_min_unique_pair_ratio": 0.485,
        "frontier_max_depth": 30,
        "frontier_max_repeat_pair_ratio": 0.60,
        "frontier_max_cx_like": 120,
    }
}
JOB_TTL_SECONDS = 20 * 60
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="compile-job")
JobStatus = Literal["queued", "running", "completed", "failed"]
JobPhase = Literal["parsing", "mapping", "routing", "output", "done", "error"]
PhaseReporter = Callable[[JobPhase], None]

EXAMPLES: dict[str, dict[str, str]] = {
    "qft5": {
        "name": "QFT 5",
        "path": "examples/qft5.qasm",
        "description": "A compact quantum Fourier transform example.",
        "topology": "tokyo",
    },
    "ghz5": {
        "name": "GHZ 5",
        "path": "examples/ghz5.qasm",
        "description": "A small entanglement-chain circuit.",
        "topology": "tokyo",
    },
    "qaoa5": {
        "name": "QAOA 5",
        "path": "examples/qaoa5.qasm",
        "description": "One QAOA-style optimization layer.",
        "topology": "tokyo",
    },
    "qft10": {
        "name": "QFT 10",
        "path": "examples/qft10.qasm",
        "description": "A larger dense-routing quantum Fourier transform example.",
        "topology": "tokyo",
    },
    "qaoa10": {
        "name": "QAOA 10",
        "path": "examples/qaoa10.qasm",
        "description": "A 10-qubit QAOA-style optimization layer.",
        "topology": "tokyo",
    },
    "ghz10": {
        "name": "GHZ 10",
        "path": "examples/ghz10.qasm",
        "description": "A 10-qubit entanglement-chain circuit.",
        "topology": "tokyo",
    },
    "vqe10": {
        "name": "VQE-like 10",
        "path": "examples/vqe10.qasm",
        "description": "A 10-qubit RealAmplitudes-style variational ansatz.",
        "topology": "tokyo",
    },
    "line_ghz30": {
        "name": "LineGHZ30",
        "path": "examples/line_ghz30.qasm",
        "description": "A 30-qubit GHZ chain for the 5x6 grid topology.",
        "topology": "grid_5x6",
    },
    "random30_d4": {
        "name": "Random30-d4",
        "path": "examples/random30_d4.qasm",
        "description": "A deterministic 30-qubit depth-4 random-style routing case.",
        "topology": "grid_5x6",
    },
    "line_ghz50": {
        "name": "LineGHZ50",
        "path": "examples/line_ghz50.qasm",
        "description": "A 50-qubit GHZ chain for the 5x10 grid topology.",
        "topology": "grid_5x10",
    },
    "ring_sparse50": {
        "name": "RingSparse50",
        "path": "examples/ring_sparse50.qasm",
        "description": "A sparse 50-qubit ring interaction case for the 5x10 grid topology.",
        "topology": "grid_5x10",
    },
}

app = FastAPI(
    title="ZJU Quantum Compiler Playground API",
    version="0.14.2",
    description=(
        "Local API for NPQR compilation, SABRE comparison, and static "
        "benchmark summaries."
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
    model_loadable: bool
    model_status: str


class ExampleInfo(BaseModel):
    id: str
    name: str
    path: str
    description: str
    topology: str


class ValidateRequest(BaseModel):
    qasm: str = Field(description="Inline OpenQASM 2 source to validate.")
    topology: str = "tokyo"


class ValidateResponse(BaseModel):
    status: Literal["OK", "Invalid"]
    message: str
    line: int | None = None
    column: int | None = None
    input_qubits: int | None = None
    gate_count: int | None = None
    cx_count: int | None = None
    supported_gates: list[str]
    warnings: list[str]


class CompileRequest(BaseModel):
    example: str | None = Field(default=None, description="Example id, such as qft5.")
    qasm: str | None = Field(default=None, description="Inline OpenQASM 2 source.")
    backend: Literal["npqr", "sabre"] = "npqr"
    heuristic: Literal["basic", "lookahead", "decay"] = STANDARD_SABRE_HEURISTIC
    topology: str = "tokyo"
    max_steps: int = 45
    npqr_frontier_pruning_policy: FrontierPruningPolicy | None = Field(
        default=None,
        description="Opt-in NPQR staging policy; omitted keeps the public default unchanged.",
    )
    npqr_frontier_trigger_profile: FrontierTriggerProfile | None = Field(
        default=None,
        description="Opt-in NPQR refined trigger staging profile; omitted keeps the public default unchanged.",
    )
    include_route_trace: bool = Field(
        default=True,
        description="Include route_trace events in compile responses; false keeps metrics but reduces payload size.",
    )
    include_compiled_qasm: bool = Field(
        default=True,
        description="Include compiled_qasm in compile responses; false keeps metrics but reduces payload size.",
    )


class RouteTraceEvent(BaseModel):
    kind: Literal["swap", "gate"]
    physical_qubits: list[int]
    logical_qubits: list[int | None] | None = None
    op: str | None = None
    source_line: int | None = None
    source_column: int | None = None
    source_text: str | None = None
    reason: str | None = None
    blocked_gate_index: int | None = None
    next_gate_index: int | None = None
    gate_index: int | None = None
    compiled_index: int | None = None
    insertion_index: int | None = None
    action: int | None = None
    mapping_before: dict[int, int] | None = None
    mapping_after: dict[int, int] | None = None
    mapping: dict[int, int] | None = None


class CompileResponse(BaseModel):
    status: Literal["OK", "INCOMPLETE", "N/A"]
    backend: Literal["npqr", "sabre"]
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
    route_trace: list[RouteTraceEvent] | None = None
    trace_len: int | None = None
    executed_gates: int | None = None
    initial_mapping: dict[int, int] | None = None
    final_mapping: dict[int, int] | None = None
    message: str | None = None


class CompilePhaseTiming(BaseModel):
    phase: JobPhase
    status: Literal["pending", "running", "done", "error"] = "pending"
    started_at: float | None = None
    ended_at: float | None = None
    elapsed_ms: float | None = None


class CompileJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    phase: JobPhase
    phases: list[CompilePhaseTiming]
    result: CompileResponse | None = None
    error: str | None = None
    created_at: float
    updated_at: float
    expires_at: float


class _CompileJobState:
    def __init__(self, job_id: str, req: CompileRequest) -> None:
        now = time.time()
        self.job_id = job_id
        self.req = req
        self.status: JobStatus = "queued"
        self.phase: JobPhase = "parsing"
        self.result: CompileResponse | None = None
        self.error: str | None = None
        self.created_at = now
        self.updated_at = now
        self.expires_at = now + JOB_TTL_SECONDS
        self.phases: dict[JobPhase, CompilePhaseTiming] = {
            phase: CompilePhaseTiming(phase=phase)
            for phase in ("parsing", "mapping", "routing", "output")
        }

    def begin(self, phase: JobPhase) -> None:
        if phase not in self.phases:
            return
        now = time.time()
        self.status = "running"
        self.phase = phase
        timing = self.phases[phase]
        if timing.started_at is None:
            timing.started_at = now
        timing.ended_at = None
        timing.elapsed_ms = None
        timing.status = "running"
        self.updated_at = now
        self.expires_at = now + JOB_TTL_SECONDS

    def finish_open_phases(self, *, errored: bool = False) -> None:
        now = time.time()
        for timing in self.phases.values():
            if timing.status == "running":
                timing.ended_at = now
                timing.elapsed_ms = max(0.0, (now - (timing.started_at or now)) * 1000)
                timing.status = "error" if errored else "done"
        self.updated_at = now
        self.expires_at = now + JOB_TTL_SECONDS

    def complete(self, result: CompileResponse) -> None:
        now = time.time()
        self.finish_open_phases()
        self.status = "completed"
        self.phase = "done"
        self.result = result
        self.updated_at = now
        self.expires_at = now + JOB_TTL_SECONDS

    def fail(self, message: str) -> None:
        now = time.time()
        self.finish_open_phases(errored=True)
        self.status = "failed"
        self.phase = "error"
        self.error = message
        self.updated_at = now
        self.expires_at = now + JOB_TTL_SECONDS

    def response(self) -> CompileJobResponse:
        return CompileJobResponse(
            job_id=self.job_id,
            status=self.status,
            phase=self.phase,
            phases=list(self.phases.values()),
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
        )


_compile_jobs: dict[str, _CompileJobState] = {}
_compile_jobs_lock = Lock()


def _cleanup_compile_jobs() -> None:
    now = time.time()
    expired = [
        job_id
        for job_id, job in _compile_jobs.items()
        if job.expires_at < now
    ]
    for job_id in expired:
        _compile_jobs.pop(job_id, None)


def _job_snapshot(job_id: str) -> CompileJobResponse:
    with _compile_jobs_lock:
        _cleanup_compile_jobs()
        job = _compile_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Unknown compile job: {job_id}")
        return job.response()


def _job_reporter(job_id: str) -> PhaseReporter:
    def report(phase: JobPhase) -> None:
        with _compile_jobs_lock:
            job = _compile_jobs.get(job_id)
            if job:
                job.finish_open_phases()
                job.begin(phase)

    return report


def _run_compile_job(job_id: str) -> None:
    with _compile_jobs_lock:
        job = _compile_jobs.get(job_id)
        if not job:
            return
        req = job.req
    try:
        result = _compile_request(req, phase_reporter=_job_reporter(job_id))
    except HTTPException as exc:
        message = str(exc.detail)
        with _compile_jobs_lock:
            job = _compile_jobs.get(job_id)
            if job:
                job.fail(message)
        return
    except Exception as exc:  # noqa: BLE001
        with _compile_jobs_lock:
            job = _compile_jobs.get(job_id)
            if job:
                job.fail(str(exc))
        return
    with _compile_jobs_lock:
        job = _compile_jobs.get(job_id)
        if job:
            job.complete(result)


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


def _request_qasm_source(req: CompileRequest) -> str | None:
    if req.qasm:
        return req.qasm
    example_id = req.example or "qft5"
    spec = EXAMPLES.get(example_id)
    if not spec:
        return None
    path = PROJECT_ROOT / spec["path"]
    return path.read_text(encoding="utf-8") if path.exists() else None


def _request_topology(req: CompileRequest) -> str:
    if req.topology != "tokyo" or req.qasm:
        return req.topology
    spec = EXAMPLES.get(req.example or "qft5")
    return spec.get("topology", req.topology) if spec else req.topology


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
            qc = qasm2.loads(
                req.qasm,
                custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid OpenQASM 2 input: {exc}") from exc
        qc.name = "inline_qasm"
        return qc
    return _load_example(req.example or "qft5")


def _qasm_source_lines(qasm: str | None) -> list[str]:
    return qasm.splitlines() if qasm else []


def _gate_source_lookup(qasm: str | None) -> dict[int, tuple[int, int, str]]:
    lookup: dict[int, tuple[int, int, str]] = {}
    gate_index = 0
    for line_number, line in enumerate(_qasm_source_lines(qasm), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith(("OPENQASM", "include", "qreg", "creg", "barrier", "measure")):
            continue
        lookup[gate_index] = (line_number, line.find(stripped) + 1, stripped)
        gate_index += 1
    return lookup


def _source_metadata(
    lookup: dict[int, tuple[int, int, str]],
    gate_index: int | None,
) -> tuple[int | None, int | None, str | None]:
    if gate_index is None or gate_index not in lookup:
        return None, None, None
    return lookup[gate_index]


def _extract_error_position(message: str) -> tuple[int | None, int | None]:
    import re

    match = re.search(r"line\s+(\d+)(?:,\s*column\s+(\d+))?", message, flags=re.IGNORECASE)
    if not match:
        return None, None
    line = int(match.group(1))
    column = int(match.group(2)) if match.group(2) else None
    return line, column


def _validate_qasm_source(qasm: str, topology: str = "tokyo") -> ValidateResponse:
    warnings: list[str] = []
    if not qasm.strip():
        return ValidateResponse(
            status="Invalid",
            message="OpenQASM 输入为空。",
            supported_gates=BASIS_GATES,
            warnings=[],
        )
    if len(qasm) > MAX_INLINE_QASM_CHARS:
        return ValidateResponse(
            status="Invalid",
            message=f"OpenQASM 输入超过 {MAX_INLINE_QASM_CHARS} 字符。",
            supported_gates=BASIS_GATES,
            warnings=[],
        )
    if not qasm.lstrip().startswith("OPENQASM 2.0;"):
        warnings.append("建议以 OPENQASM 2.0; 作为第一行。")
    if 'include "qelib1.inc";' not in qasm:
        warnings.append('建议包含 include "qelib1.inc"; 以使用标准门。')
    try:
        circuit = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception as exc:  # noqa: BLE001
        line, column = _extract_error_position(str(exc))
        return ValidateResponse(
            status="Invalid",
            message=f"OpenQASM 解析失败：{exc}",
            line=line,
            column=column,
            supported_gates=BASIS_GATES,
            warnings=warnings,
        )
    _, coupling_map = _resolve_topology(topology)
    if circuit.num_qubits > coupling_map.size():
        return ValidateResponse(
            status="Invalid",
            message=f"电路有 {circuit.num_qubits} 个量子位，当前拓扑只有 {coupling_map.size()} 个物理量子位。",
            input_qubits=circuit.num_qubits,
            gate_count=len(circuit.data),
            cx_count=dict(circuit.count_ops()).get("cx", 0),
            supported_gates=BASIS_GATES,
            warnings=warnings,
        )
    return ValidateResponse(
        status="OK",
        message="OpenQASM 解析通过，可提交编译。",
        input_qubits=circuit.num_qubits,
        gate_count=len(circuit.data),
        cx_count=dict(circuit.count_ops()).get("cx", 0),
        supported_gates=BASIS_GATES,
        warnings=warnings,
    )


def _compile_sabre(
    circuit: QuantumCircuit,
    coupling_map,
    heuristic: Literal["basic", "lookahead", "decay"] = STANDARD_SABRE_HEURISTIC,
) -> QuantumCircuit:
    mapped = _map_for_sabre(circuit, coupling_map)
    return _route_mapped_sabre(mapped, coupling_map, heuristic)


def _map_for_sabre(circuit: QuantumCircuit, coupling_map) -> QuantumCircuit:
    layout_and_route = PassManager(
        [
            TrivialLayout(coupling_map),
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ]
    )
    return layout_and_route.run(circuit)


def _route_mapped_sabre(
    mapped_circuit: QuantumCircuit,
    coupling_map,
    heuristic: Literal["basic", "lookahead", "decay"] = STANDARD_SABRE_HEURISTIC,
) -> QuantumCircuit:
    routed = PassManager(
        [SabreSwap(coupling_map, heuristic=heuristic, seed=42, trials=1)]
    ).run(mapped_circuit)
    return transpile(
        routed,
        basis_gates=BASIS_GATES,
        optimization_level=0,
        seed_transpiler=42,
    )


def _sabre_summary(
    circuit: QuantumCircuit,
    coupling_map,
    heuristic: Literal["basic", "lookahead", "decay"] = STANDARD_SABRE_HEURISTIC,
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


def _logical_at_physical(mapping: dict[int, int], physical: int) -> int | None:
    for logical, mapped_physical in mapping.items():
        if int(mapped_physical) == int(physical):
            return int(logical)
    return None


def _swap_mapping(mapping: dict[int, int], physical_a: int, physical_b: int) -> dict[int, int]:
    updated = dict(mapping)
    for logical, physical in mapping.items():
        if int(physical) == int(physical_a):
            updated[int(logical)] = int(physical_b)
        elif int(physical) == int(physical_b):
            updated[int(logical)] = int(physical_a)
        else:
            updated[int(logical)] = int(physical)
    return updated


def _gate_metadata(circuit: QuantumCircuit, gate_index: int | None) -> tuple[str | None, list[int] | None]:
    if gate_index is None or gate_index < 0 or gate_index >= len(circuit.data):
        return None, None
    instruction = circuit.data[gate_index]
    logical_qubits = [
        int(circuit.find_bit(qubit).index)
        for qubit in instruction.qubits
    ]
    return instruction.operation.name, logical_qubits


def _next_gate_index(events: list[Any], start: int) -> int | None:
    for event in events[start + 1:]:
        if event.kind == "gate":
            return event.gate_index
    return None


def _swap_reason(
    physical_qubits: list[int],
    logical_qubits: list[int | None] | None,
    next_gate_index: int | None,
) -> str:
    edge = "-".join(f"p{qubit}" for qubit in physical_qubits)
    logical = "-".join("--" if qubit is None else f"q{qubit}" for qubit in (logical_qubits or []))
    next_gate = "--" if next_gate_index is None else f"g{next_gate_index}"
    return f"为让后续门 {next_gate} 的逻辑量子位更接近，沿 IBM Tokyo 相邻边 {edge} 交换 {logical}。"


def _gate_reason(op: str | None, logical_qubits: list[int] | None, physical_qubits: list[int]) -> str:
    if len(physical_qubits) == 1:
        return f"{op or '单比特门'} 作用在已映射的物理量子位 p{physical_qubits[0]}。"
    logical = "-".join(f"q{qubit}" for qubit in (logical_qubits or []))
    physical = "-".join(f"p{qubit}" for qubit in physical_qubits)
    return f"{op or '两比特门'} 的逻辑位 {logical} 当前映射到相邻物理边 {physical}，可直接执行。"


def _route_trace_payload(result: Any, circuit: QuantumCircuit, qasm_source: str | None = None) -> list[RouteTraceEvent]:
    if not result.replay:
        return []
    source_lookup = _gate_source_lookup(qasm_source)
    mapping = {
        int(logical): int(physical)
        for logical, physical in (result.initial_mapping or {}).items()
    }
    events = list(result.replay.events)
    payload: list[RouteTraceEvent] = []
    for index, event in enumerate(events):
        physical_qubits = [int(qubit) for qubit in event.physical_qubits]
        before = dict(mapping)
        op, logical_qubits = _gate_metadata(circuit, event.gate_index)
        insertion_index = event.gate_index
        next_gate_index = None
        if event.kind == "swap":
            logical_qubits = [_logical_at_physical(before, qubit) for qubit in physical_qubits]
            op = "swap"
            next_gate_index = _next_gate_index(events, index)
            insertion_index = next_gate_index
            if len(physical_qubits) == 2:
                mapping = _swap_mapping(mapping, physical_qubits[0], physical_qubits[1])
        after = dict(mapping)
        source_gate_index = next_gate_index if event.kind == "swap" else event.gate_index
        source_line, source_column, source_text = _source_metadata(source_lookup, source_gate_index)
        payload.append(
            RouteTraceEvent(
                kind=event.kind,
                physical_qubits=physical_qubits,
                logical_qubits=logical_qubits,
                op=op,
                source_line=source_line,
                source_column=source_column,
                source_text=source_text,
                reason=(
                    _swap_reason(physical_qubits, logical_qubits, next_gate_index)
                    if event.kind == "swap"
                    else _gate_reason(op, logical_qubits, physical_qubits)
                ),
                blocked_gate_index=next_gate_index if event.kind == "swap" else None,
                next_gate_index=next_gate_index,
                gate_index=event.gate_index,
                insertion_index=insertion_index,
                action=event.action,
                mapping_before=before,
                mapping_after=after,
                mapping=after,
            )
        )
    return payload


def _sabre_trace_payload(
    compiled: QuantumCircuit,
    input_qubits: int,
) -> tuple[list[RouteTraceEvent], dict[int, int]]:
    mapping = {logical: logical for logical in range(input_qubits)}
    payload: list[RouteTraceEvent] = []
    two_qubit_index = 0
    for compiled_index, instruction in enumerate(compiled.data):
        physical_qubits = [
            int(compiled.find_bit(qubit).index)
            for qubit in instruction.qubits
        ]
        if len(physical_qubits) != 2:
            continue
        before = dict(mapping)
        logical_qubits = [_logical_at_physical(before, qubit) for qubit in physical_qubits]
        op = instruction.operation.name
        kind: Literal["swap", "gate"] = "swap" if op == "swap" else "gate"
        insertion_index = two_qubit_index
        if kind == "swap":
            mapping = _swap_mapping(mapping, physical_qubits[0], physical_qubits[1])
        else:
            two_qubit_index += 1
        after = dict(mapping)
        reason = (
            _swap_reason(physical_qubits, logical_qubits, insertion_index)
            if kind == "swap"
            else _gate_reason(op, [qubit for qubit in logical_qubits if qubit is not None], physical_qubits)
        )
        payload.append(
            RouteTraceEvent(
                kind=kind,
                physical_qubits=physical_qubits,
                logical_qubits=logical_qubits,
                op=op,
                reason=reason,
                blocked_gate_index=insertion_index if kind == "swap" else None,
                next_gate_index=insertion_index if kind == "swap" else None,
                compiled_index=int(compiled_index),
                insertion_index=int(insertion_index),
                mapping_before=before,
                mapping_after=after,
                mapping=after,
            )
        )
    return payload, mapping


@lru_cache(maxsize=8)
def _npqr_runtime_for(
    topology: str,
    model_path: str,
    max_steps: int,
    frontier_pruning_policy: str | None = None,
    frontier_trigger_profile: str | None = None,
) -> "NPQRRuntime":
    from src.compiler.npqr_runtime import NPQRRuntime, NPQRRuntimeConfig

    _, coupling_map = _resolve_topology(topology)
    profile_config = _frontier_trigger_profile_config(frontier_trigger_profile)
    policy = frontier_pruning_policy
    if frontier_trigger_profile:
        policy = policy or REFINED_FRONTIER_TRIGGER_POLICY
    config = NPQRRuntimeConfig(
        max_steps=max_steps,
        frontier_rescue_enabled=bool(policy),
        frontier_action_pruning_policy=policy,
        **profile_config,
    )
    return NPQRRuntime(coupling_map, model_path=model_path, config=config)


def _frontier_trigger_profile_config(profile: str | None) -> dict[str, Any]:
    if profile is None:
        return {}
    if profile not in FRONTIER_TRIGGER_PROFILES:
        known = ", ".join(sorted(FRONTIER_TRIGGER_PROFILES))
        raise ValueError(f"npqr_frontier_trigger_profile must be one of: {known}")
    return dict(FRONTIER_TRIGGER_PROFILES[profile])


def _npqr_load_status(topology: str = "tokyo") -> tuple[bool, str]:
    if not DEFAULT_NPQR_MODEL.exists():
        return False, "Default NPQR model file is not present locally."
    try:
        runtime = _npqr_runtime_for(topology, str(DEFAULT_NPQR_MODEL), 45, None, None)
    except ImportError as exc:
        return False, f"NPQR dependency is not installed in this REST deployment: {exc.name}."
    if runtime.has_model:
        return True, "NPQR neural selector/search/repair runtime can be loaded."
    return False, runtime.model_load_error or "Default NPQR model could not be loaded."


@app.get("/api/status", response_model=StatusResponse)
async def api_status() -> StatusResponse:
    """Return public project and model status without running a benchmark."""
    npqr_loadable, npqr_message = _npqr_load_status()
    return StatusResponse(
        version="0.14.2",
        status=(
            "Default backend is NPQR neural-assisted routing. SABRE is returned "
            "as a comparison baseline, not as an NPQR fallback."
        ),
        default_topology="tokyo",
        default_backend="npqr",
        available_topologies=sorted(_TOPOLOGY_ALIAS),
        default_model=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)),
        model_exists=DEFAULT_NPQR_MODEL.exists(),
        model_loadable=npqr_loadable,
        model_status=npqr_message,
    )


@app.get("/api/examples", response_model=list[ExampleInfo])
async def api_examples() -> list[ExampleInfo]:
    """Return the checked-in QASM examples used by the public page."""
    return [
        ExampleInfo(id=key, **value)
        for key, value in sorted(EXAMPLES.items())
    ]


@app.post("/api/validate", response_model=ValidateResponse)
async def api_validate(req: ValidateRequest) -> ValidateResponse:
    """Validate inline OpenQASM 2 without running NPQR or SABRE compilation."""
    return _validate_qasm_source(req.qasm, req.topology)


@app.get("/api/benchmarks")
async def api_benchmarks() -> dict:
    """Return public benchmark and claim boundaries without internal logs."""
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


@app.get("/api/npqr/evidence")
async def api_npqr_evidence() -> dict:
    """Return the unified NPQR evidence manifest without rerunning training."""
    return load_npqr_evidence_manifest()


def _compile_request(
    req: CompileRequest,
    phase_reporter: PhaseReporter | None = None,
) -> CompileResponse:
    if phase_reporter:
        phase_reporter("parsing")
    topology = _request_topology(req)
    topo_name, coupling_map = _resolve_topology(topology)
    qasm_source = _request_qasm_source(req)
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
            if phase_reporter:
                phase_reporter("output")
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
                baseline=_sabre_summary(circuit, coupling_map),
                message="NPQR checkpoint file is missing; SABRE baseline is comparison only.",
            )
        try:
            if phase_reporter:
                phase_reporter("mapping")
            runtime = _npqr_runtime_for(
                topology,
                str(DEFAULT_NPQR_MODEL),
                req.max_steps,
                req.npqr_frontier_pruning_policy,
                req.npqr_frontier_trigger_profile,
            )
        except ImportError as exc:
            if phase_reporter:
                phase_reporter("output")
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
                baseline=_sabre_summary(circuit, coupling_map),
                message=f"NPQR dependency is not installed in this REST deployment: {exc.name}.",
            )
        if phase_reporter:
            phase_reporter("routing")
        result = runtime.compile(circuit)
        if phase_reporter:
            phase_reporter("output")
        compiled_qasm = (
            qasm2.dumps(result.compiled_circuit)
            if req.include_compiled_qasm and result.completed and result.compiled_circuit
            else None
        )
        route_trace = _route_trace_payload(result, circuit, qasm_source) if req.include_route_trace else None
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
            baseline=_sabre_summary(circuit, coupling_map),
            route_trace=route_trace,
            trace_len=result.trace_len,
            executed_gates=result.executed_gates,
            initial_mapping=result.initial_mapping,
            final_mapping=result.final_mapping,
            message=result.message,
        )

    if req.backend == "sabre":
        if phase_reporter:
            phase_reporter("mapping")
        mapped = _map_for_sabre(circuit, coupling_map)
        if phase_reporter:
            phase_reporter("routing")
        compiled = _route_mapped_sabre(mapped, coupling_map, req.heuristic)
        if phase_reporter:
            phase_reporter("output")
        route_trace, final_mapping = _sabre_trace_payload(compiled, circuit.num_qubits)
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
            compiled_qasm=qasm2.dumps(compiled) if req.include_compiled_qasm else None,
            route_trace=route_trace if req.include_route_trace else None,
            trace_len=len(route_trace),
            executed_gates=sum(1 for event in route_trace if event.kind == "gate"),
            initial_mapping={logical: logical for logical in range(circuit.num_qubits)},
            final_mapping=final_mapping,
            message=f"SABRE completed with the {req.heuristic} heuristic.",
        )

    raise HTTPException(status_code=400, detail="backend must be npqr or sabre.")


@app.post("/api/compile/jobs", response_model=CompileJobResponse)
async def create_compile_job(req: CompileRequest) -> CompileJobResponse:
    """Create an in-memory compile job and return its current status."""
    job_id = uuid.uuid4().hex
    job = _CompileJobState(job_id, req)
    with _compile_jobs_lock:
        _cleanup_compile_jobs()
        _compile_jobs[job_id] = job
    JOB_EXECUTOR.submit(_run_compile_job, job_id)
    return _job_snapshot(job_id)


@app.get("/api/compile/jobs/{job_id}", response_model=CompileJobResponse)
async def get_compile_job(job_id: str) -> CompileJobResponse:
    """Return the current status, phase timings, and result for a compile job."""
    return _job_snapshot(job_id)


@app.post("/api/compile", response_model=CompileResponse)
async def api_compile(req: CompileRequest) -> CompileResponse:
    """Compile a QASM example with NPQR or SABRE."""
    return _compile_request(req)


@app.get("/api/topology/{name}")
async def api_topology(name: str) -> dict:
    """Return topology metadata for small UI checks."""
    topo_name, coupling_map = _resolve_topology(name)
    return {
        "name": topo_name,
        "info": get_topology_info(coupling_map),
        "edges": [[int(a), int(b)] for a, b in coupling_map.get_edges()],
    }
