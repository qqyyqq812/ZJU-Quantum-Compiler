"""Command-line entry points for the public NPQR compiler."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreSwap,
    TrivialLayout,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "default" / "npqr-default.pt"

_TOPOLOGY_ALIAS: dict[str, str] = {
    "tokyo": "ibm_tokyo",
    "tokyo20": "ibm_tokyo",
    "ibm_tokyo": "ibm_tokyo",
    "linear5": "linear_5",
    "linear_5": "linear_5",
    "grid3x3": "grid_3x3",
    "grid_3x3": "grid_3x3",
}

_EXAMPLES = {
    "ghz5": "examples/ghz5.qasm",
    "qft5": "examples/qft5.qasm",
    "qaoa5": "examples/qaoa5.qasm",
    "qft10": "examples/qft10.qasm",
    "qaoa10": "examples/qaoa10.qasm",
    "ghz10": "examples/ghz10.qasm",
    "vqe10": "examples/vqe10.qasm",
}

_BASIS_GATES = ["cx", "id", "rz", "sx", "x", "swap"]


def _resolve_topology(name: str):
    from src.benchmarks.topologies import get_topology

    canon = _TOPOLOGY_ALIAS.get(name, name)
    return canon, get_topology(canon)


def _compile_sabre(circuit: QuantumCircuit, coupling_map, heuristic: str = "basic") -> QuantumCircuit:
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
        basis_gates=_BASIS_GATES,
        optimization_level=0,
        seed_transpiler=42,
    )


def _compile_npqr(circuit: QuantumCircuit, topology: str, max_steps: int):
    from src.compiler.npqr_runtime import NPQRRuntime, NPQRRuntimeConfig

    if not _DEFAULT_MODEL.exists():
        raise FileNotFoundError(f"default NPQR model is missing: {_DEFAULT_MODEL}")
    _topo_name, coupling_map = _resolve_topology(topology)
    runtime = NPQRRuntime(
        coupling_map,
        model_path=str(_DEFAULT_MODEL),
        config=NPQRRuntimeConfig(max_steps=max_steps),
    )
    return runtime.compile(circuit)


def _cmd_info(_: argparse.Namespace) -> int:
    print("ZJU Quantum Compiler 0.14.2")
    print(f"Project root: {_PROJECT_ROOT}")
    print("Default backend: NPQR neural-assisted routing")
    print(f"Default model: {_DEFAULT_MODEL}")
    print(f"Model exists: {_DEFAULT_MODEL.exists()}")
    print()
    print("Topologies:")
    for alias, canon in sorted(_TOPOLOGY_ALIAS.items()):
        print(f"  {alias:10s} -> {canon}")
    print()
    print("Examples:")
    for name, path in sorted(_EXAMPLES.items()):
        print(f"  {name:8s} {path}")
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    qasm_path = Path(args.qasm_path).expanduser().resolve()
    if not qasm_path.exists():
        print(f"[error] QASM file not found: {qasm_path}", file=sys.stderr)
        return 2

    try:
        topo_name, coupling_map = _resolve_topology(args.topology)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    original_cx = int(dict(circuit.count_ops()).get("cx", 0))
    started = time.perf_counter()

    if args.backend == "npqr":
        try:
            result = _compile_npqr(circuit, args.topology, args.max_steps)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] NPQR compile failed: {exc}", file=sys.stderr)
            return 3
        elapsed_ms = result.elapsed_ms
        compiled = result.compiled_circuit
        status = result.status
        swaps = result.total_swaps
        depth = result.depth
        message = result.message
    else:
        compiled = _compile_sabre(circuit, coupling_map, args.heuristic)
        elapsed_ms = (time.perf_counter() - started) * 1000
        status = "OK"
        swaps = int(dict(compiled.count_ops()).get("swap", 0))
        depth = int(compiled.depth())
        message = f"SABRE completed with the {args.heuristic} heuristic."

    print(f"Circuit:  {qasm_path.name} ({circuit.num_qubits} qubits, cx={original_cx})")
    print(f"Topology: {topo_name} ({coupling_map.size()} qubits)")
    print(f"Backend:  {args.backend}")
    print(f"Status:   {status}")
    print(f"SWAPs:    {swaps}")
    print(f"Depth:    {depth}")
    print(f"Time:     {elapsed_ms:.1f} ms")
    print(f"Message:  {message}")

    if args.output and compiled is not None:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        qasm2.dump(compiled, out_path)
        print(f"Output:   {out_path}")

    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    from scripts.experiment_algorithm_matrix import print_markdown, run_matrix

    rows = run_matrix(quick=args.quick, examples=args.examples)
    print_markdown(rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qcompiler",
        description="ZJU Quantum Compiler public CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compile = sub.add_parser("compile", help="Compile a QASM circuit")
    p_compile.add_argument("qasm_path", help="Path to an OpenQASM 2 file")
    p_compile.add_argument("--topology", default="tokyo", help="Topology alias, default: tokyo")
    p_compile.add_argument("--backend", default="npqr", choices=["npqr", "sabre"])
    p_compile.add_argument(
        "--heuristic",
        default="basic",
        choices=["basic", "lookahead", "decay"],
        help="SABRE heuristic when --backend sabre is used",
    )
    p_compile.add_argument("--max-steps", type=int, default=45, help="NPQR step limit")
    p_compile.add_argument("--output", default=None, help="Write compiled QASM to this path")
    p_compile.set_defaults(func=_cmd_compile)

    p_matrix = sub.add_parser("matrix", help="Print the reproducible algorithm matrix")
    p_matrix.add_argument("--quick", action="store_true", help="Use the small default example subset")
    p_matrix.add_argument("--examples", nargs="+", choices=sorted(_EXAMPLES))
    p_matrix.set_defaults(func=_cmd_matrix)

    p_info = sub.add_parser("info", help="Show model, topology, and example status")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
