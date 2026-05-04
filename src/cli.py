"""
qcompiler — ZJU Quantum Compiler CLI
====================================

Three subcommands:
  qcompiler compile <qasm_path> [--topology ...] [--backend ai|sabre] [--output ...]
  qcompiler eval [--model path.pt] [--circuits qft_5,qaoa_5,ghz_5]
  qcompiler info
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "v14_tokyo20" / "checkpoint_ep25333.pt"

# Friendly aliases → internal topology names (src/benchmarks/topologies.py)
_TOPOLOGY_ALIAS: dict[str, str] = {
    "tokyo": "ibm_tokyo",
    "tokyo20": "ibm_tokyo",
    "ibm_tokyo": "ibm_tokyo",
    "linear5": "linear_5",
    "linear_5": "linear_5",
    "grid3x3": "grid_3x3",
    "grid_3x3": "grid_3x3",
}


def _resolve_topology(name: str):
    from src.benchmarks.topologies import get_topology
    canon = _TOPOLOGY_ALIAS.get(name, name)
    return canon, get_topology(canon)


def _load_qasm(path: Path):
    from qiskit import QuantumCircuit
    return QuantumCircuit.from_qasm_file(str(path))


def _build_ghz(n_qubits: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def _build_vqe(n_qubits: int):
    from src.benchmarks.mqt_bench import _vqe_fallback
    return _vqe_fallback(n_qubits)


def _build_named_circuit(tag: str):
    from src.benchmarks.circuits import (
        generate_grover,
        generate_qaoa,
        generate_qft,
    )

    gens = {
        "qft": generate_qft,
        "qaoa": generate_qaoa,
        "grover": generate_grover,
        "ghz": _build_ghz,
        "vqe": _build_vqe,
    }
    kind, n_str = tag.split("_")
    return gens[kind](int(n_str))


def _count_swap_overhead(original_cx: int, compiled_cx: int) -> int:
    return max(0, (compiled_cx - original_cx)) // 3


def _cmd_info(_: argparse.Namespace) -> int:
    print("ZJU Quantum Compiler — qcompiler 0.14.2")
    print(f"Project root: {_PROJECT_ROOT}")
    print("Current status: V14/V15 not yet better than SABRE; demo is evidence-first.")
    print()
    print("Available topologies (alias → internal):")
    for alias, canon in sorted(_TOPOLOGY_ALIAS.items()):
        if alias == canon:
            continue
        print(f"  {alias:10s} → {canon}")
    print()
    print("Model weights:")
    models_dir = _PROJECT_ROOT / "models"
    if models_dir.exists():
        for sub in sorted(models_dir.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                pt_files = sorted(sub.glob("*.pt"))
                if pt_files:
                    best = next((p for p in pt_files if "best" in p.name), pt_files[0])
                    size_mb = best.stat().st_size / 1024 / 1024
                    print(f"  {sub.name:24s} {best.name} ({size_mb:.1f} MB)")
    else:
        print("  (models/ not found)")
    print()
    print(f"Default model: {_DEFAULT_MODEL}")
    print(f"  exists: {_DEFAULT_MODEL.exists()}")
    print("Default AI model is used for status/evaluation only; incomplete routes are reported honestly.")
    return 0


def _compile_sabre(circuit, coupling_map):
    from qiskit import transpile

    basis_gates = ["cx", "id", "rz", "sx", "x", "swap"]
    return transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=0,
        layout_method="trivial",
        routing_method="sabre",
        seed_transpiler=42,
    )


def _compile_ai(circuit, coupling_map, model_path: Path, max_steps: int):
    if not model_path.exists():
        print(
            f"[error] AI model weights not found: {model_path}\n"
            f"        Download from README:\n"
            f"        https://github.com/qqyyqq812/ZJU-Quantum-Compiler#models",
            file=sys.stderr,
        )
        return None
    try:
        from src.compiler.pass_manager import AIRouter
    except Exception as exc:  # noqa: BLE001
        print(f"[error] AI backend import failed: {exc}", file=sys.stderr)
        return None

    router = AIRouter(coupling_map, model_path=str(model_path))
    if not router._has_model:
        print("[error] AI model could not be loaded; evaluation will not pretend it worked.", file=sys.stderr)
        if router.model_load_error:
            print(f"[error] model_load_error: {router.model_load_error.splitlines()[0]}", file=sys.stderr)
        return None
    return router.route(circuit, max_steps=max_steps)


def _cmd_compile(args: argparse.Namespace) -> int:
    qasm_path = Path(args.qasm_path).expanduser().resolve()
    if not qasm_path.exists():
        print(f"[error] qasm file not found: {qasm_path}", file=sys.stderr)
        return 2

    try:
        topo_name, coupling_map = _resolve_topology(args.topology)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    circuit = _load_qasm(qasm_path)
    original_cx = dict(circuit.count_ops()).get("cx", 0)
    original_depth = circuit.depth()

    print(f"Circuit:    {qasm_path.name}  ({circuit.num_qubits} qubits, "
          f"depth={original_depth}, cx={original_cx})")
    print(f"Topology:   {topo_name}  ({coupling_map.size()} qubits)")
    print(f"Backend:    {args.backend}")

    t0 = time.perf_counter()
    route_info = None
    if args.backend == "sabre":
        compiled = _compile_sabre(circuit, coupling_map)
    elif args.backend == "ai":
        model_path = Path(args.model).expanduser().resolve() if args.model else _DEFAULT_MODEL
        ai_result = _compile_ai(circuit, coupling_map, model_path, args.max_steps)
        if ai_result is None:
            return 3
        compiled, route_info = ai_result
    else:
        print(f"[error] unknown backend: {args.backend}", file=sys.stderr)
        return 2
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ops = dict(compiled.count_ops())
    compiled_cx = ops.get("cx", 0)
    extra_swap = ops.get("swap", _count_swap_overhead(original_cx, compiled_cx))
    print()
    print(f"Compiled:   depth={compiled.depth()}, cx={compiled_cx}, "
          f"extra_swap={extra_swap}, time={elapsed_ms:.1f}ms")
    if route_info is not None:
        status = "OK" if route_info.get("done") else "INCOMPLETE"
        print(f"AI status:  {status}, executed_gates={route_info.get('executed_gates')}, "
              f"trace_events={route_info.get('trace_events')}")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from qiskit import qasm2
        qasm2.dump(compiled, out_path)
        print(f"Output:     {out_path}")

    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from src.compiler.pass_manager import AIRouter

    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    topo_name, coupling_map = _resolve_topology(args.topology)
    model_path = Path(args.model).expanduser().resolve() if args.model else _DEFAULT_MODEL
    router = None
    if not args.no_ai and model_path.exists():
        router = AIRouter(coupling_map, model_path=str(model_path))
        if not router._has_model:
            router = None

    print(f"Topology: {topo_name}  ({coupling_map.size()} qubits)")
    print(f"AI model: {model_path if router else 'N/A'}")
    print()
    header = (
        f"{'circuit':<14} {'cx_in':>6} {'SABRE':>6} {'AI status':>11} "
        f"{'AI':>6} {'AI gates':>8} {'ratio':>8} {'time_ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for tag in circuits:
        try:
            qc = _build_named_circuit(tag)
        except (KeyError, ValueError):
            print(f"{tag:<14} [skip: bad spec, expect <qft|qaoa|grover|ghz|vqe>_<n>]")
            continue
        in_cx = dict(qc.count_ops()).get("cx", 0)
        t0 = time.perf_counter()
        out = _compile_sabre(qc, coupling_map)
        sabre_ms = (time.perf_counter() - t0) * 1000
        sabre_swap = dict(out.count_ops()).get("swap", 0)

        ai_status = "N/A"
        ai_swap: int | str = "N/A"
        ai_gates: int | str = "N/A"
        ratio = "N/A"
        ai_ms = 0.0
        if router is not None:
            t0 = time.perf_counter()
            ai = router.route_count_only(qc, max_steps=args.max_steps)
            ai_ms = (time.perf_counter() - t0) * 1000
            ai_status = "OK" if ai["done"] else "INCOMPLETE"
            ai_swap = ai["ai_swaps"]
            ai_gates = ai["executed_gates"]
            if sabre_swap > 0:
                ratio = f"{ai['ai_swaps'] / sabre_swap:.3f}"
            elif sabre_swap == 0 and ai["ai_swaps"] == 0:
                ratio = "1.000"

        total_ms = sabre_ms + ai_ms
        print(
            f"{tag:<14} {in_cx:>6d} {sabre_swap:>6d} {ai_status:>11} "
            f"{ai_swap:>6} {ai_gates:>8} {ratio:>8} {total_ms:>9.1f}"
        )
    if router is None:
        print("\n[note] AI model unavailable or --no-ai was set; SABRE rows are still valid.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qcompiler",
        description="ZJU Quantum Circuit AI Compiler",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compile = sub.add_parser("compile", help="Compile a QASM circuit")
    p_compile.add_argument("qasm_path", help="Path to .qasm file")
    p_compile.add_argument("--topology", default="tokyo",
                           help="Topology alias: tokyo|linear5|grid3x3 (default tokyo)")
    p_compile.add_argument("--backend", default="sabre", choices=["ai", "sabre"],
                           help="Compilation backend (default sabre)")
    p_compile.add_argument("--model", default=None,
                           help="AI model .pt path (default: models/v14_tokyo20/checkpoint_ep25333.pt)")
    p_compile.add_argument("--max-steps", type=int, default=2000,
                           help="AI route step limit (default 2000)")
    p_compile.add_argument("--output", default=None, help="Write compiled QASM to file")
    p_compile.set_defaults(func=_cmd_compile)

    p_eval = sub.add_parser("eval", help="Evaluate compilers on benchmark circuits")
    p_eval.add_argument("--model", default=None, help="AI model .pt path")
    p_eval.add_argument("--circuits", default="qft_5,qaoa_5,ghz_5",
                        help="Comma-separated circuits (e.g. qft_5,qaoa_5,ghz_5)")
    p_eval.add_argument("--topology", default="tokyo", help="Topology alias")
    p_eval.add_argument("--max-steps", type=int, default=2000,
                        help="AI route step limit (default 2000)")
    p_eval.add_argument("--no-ai", action="store_true",
                        help="Skip AI routing and only report SABRE")
    p_eval.set_defaults(func=_cmd_eval)

    p_info = sub.add_parser("info", help="Show available models and topologies")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
