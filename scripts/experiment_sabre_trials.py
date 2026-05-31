"""Compare SabreSwap heuristic quality and runtime across trial counts.

This experiment keeps the same layout pre-passes and seed as the public API,
then varies only the SabreSwap ``trials`` parameter. It is local-only and prints
a Markdown table for the project board.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreSwap,
    TrivialLayout,
)

from src.benchmarks.topologies import get_topology

BASIS_GATES = ["cx", "id", "rz", "sx", "x", "swap"]
EXAMPLES = {
    "qft5": "examples/qft5.qasm",
    "qaoa5": "examples/qaoa5.qasm",
    "ghz5": "examples/ghz5.qasm",
    "qft10": "examples/qft10.qasm",
    "qaoa10": "examples/qaoa10.qasm",
    "ghz10": "examples/ghz10.qasm",
    "vqe10": "examples/vqe10.qasm",
}
HEURISTICS = ("basic", "lookahead", "decay")


@dataclass
class TrialResult:
    example: str
    heuristic: str
    trials: int
    swaps: int
    depth: int
    runtime_ms: float


def compile_sabre(
    circuit: QuantumCircuit,
    coupling_map,
    heuristic: str,
    trials: int,
) -> tuple[int, int]:
    pass_manager = PassManager(
        [
            TrivialLayout(coupling_map),
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(coupling_map, heuristic=heuristic, seed=42, trials=trials),
        ]
    )
    routed = pass_manager.run(circuit)
    compiled = transpile(
        routed,
        basis_gates=BASIS_GATES,
        optimization_level=0,
        seed_transpiler=42,
    )
    ops = dict(compiled.count_ops())
    return int(ops.get("swap", 0)), int(compiled.depth())


def run_experiment(examples: list[str], trials_values: list[int]) -> list[TrialResult]:
    coupling_map = get_topology("ibm_tokyo")
    results: list[TrialResult] = []
    for example in examples:
        circuit = QuantumCircuit.from_qasm_file(str(Path(EXAMPLES[example])))
        for heuristic in HEURISTICS:
            for trials in trials_values:
                started = time.perf_counter()
                swaps, depth = compile_sabre(circuit, coupling_map, heuristic, trials)
                runtime_ms = (time.perf_counter() - started) * 1000
                results.append(
                    TrialResult(
                        example=example,
                        heuristic=heuristic,
                        trials=trials,
                        swaps=swaps,
                        depth=depth,
                        runtime_ms=runtime_ms,
                    )
                )
    return results


def print_results(results: list[TrialResult]) -> None:
    print("| example | heuristic | trials | swaps | depth | ms |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in results:
        print(
            f"| {row.example} | {row.heuristic} | {row.trials} | "
            f"{row.swaps} | {row.depth} | {row.runtime_ms:.1f} |"
        )


def print_summary(results: list[TrialResult]) -> None:
    grouped: dict[tuple[str, str], list[TrialResult]] = {}
    for row in results:
        grouped.setdefault((row.example, row.heuristic), []).append(row)

    print()
    print("| example | heuristic | swaps@first | swaps@best | ms@first | ms@best | decision |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row.trials)
        first = rows[0]
        best = min(rows, key=lambda row: (row.swaps, row.runtime_ms))
        if best.swaps < first.swaps:
            decision = f"multi-trial improves by {first.swaps - best.swaps}"
        else:
            decision = "single-trial enough"
        print(
            f"| {key[0]} | {key[1]} | {first.swaps} | {best.swaps} | "
            f"{first.runtime_ms:.1f} | {best.runtime_ms:.1f} | {decision} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        nargs="+",
        default=list(EXAMPLES),
        choices=sorted(EXAMPLES),
    )
    parser.add_argument("--trials", nargs="+", type=int, default=[1, 4, 8])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args.examples, args.trials)
    print_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
