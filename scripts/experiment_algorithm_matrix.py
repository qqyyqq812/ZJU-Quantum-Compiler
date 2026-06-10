"""Generate a reproducible algorithm matrix for routing candidates.

The matrix is intentionally conservative: the default quick mode evaluates the
same SabreSwap heuristics used by the public website before any new algorithm is
allowed to enter the website default path.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import TextIO

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_sabre_trials import EXAMPLES, HEURISTICS, compile_sabre
from src.benchmarks.topologies import get_topology

QUICK_EXAMPLES = ("ghz5", "qft5", "qft10")


@dataclass(frozen=True)
class MatrixRow:
    circuit: str
    algorithm: str
    swap: int
    depth: int
    completed: bool
    runtime_ms: float


def _rows_as_dicts(rows: list[MatrixRow]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]


def run_matrix(quick: bool = False, examples: list[str] | None = None) -> list[MatrixRow]:
    selected_examples = examples or list(QUICK_EXAMPLES if quick else EXAMPLES)
    coupling_map = get_topology("ibm_tokyo")
    rows: list[MatrixRow] = []

    for circuit_name in selected_examples:
        circuit = QuantumCircuit.from_qasm_file(str(Path(EXAMPLES[circuit_name])))
        for heuristic in HEURISTICS:
            started = time.perf_counter()
            try:
                swaps, depth = compile_sabre(
                    circuit,
                    coupling_map,
                    heuristic=heuristic,
                    trials=1,
                )
                completed = True
            except Exception:
                swaps = -1
                depth = -1
                completed = False
            runtime_ms = (time.perf_counter() - started) * 1000
            rows.append(
                MatrixRow(
                    circuit=circuit_name,
                    algorithm=heuristic,
                    swap=swaps,
                    depth=depth,
                    completed=completed,
                    runtime_ms=runtime_ms,
                )
            )
    return rows


def write_json(rows: list[MatrixRow], output: TextIO) -> None:
    json.dump(_rows_as_dicts(rows), output, ensure_ascii=False, indent=2)
    output.write("\n")


def write_csv(rows: list[MatrixRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["circuit", "algorithm", "swap", "depth", "completed", "runtime_ms"],
        )
        writer.writeheader()
        writer.writerows(_rows_as_dicts(rows))


def print_markdown(rows: list[MatrixRow]) -> None:
    print("| circuit | algorithm | swap | depth | completed | runtime_ms |")
    print("| --- | --- | ---: | ---: | --- | ---: |")
    for row in rows:
        print(
            f"| {row.circuit} | {row.algorithm} | {row.swap} | {row.depth} | "
            f"{str(row.completed).lower()} | {row.runtime_ms:.1f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Evaluate a small representative subset: ghz5, qft5, qft10.",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=sorted(EXAMPLES),
        help="Explicit checked-in examples to evaluate.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    output_group.add_argument("--csv", type=Path, help="Write CSV to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_matrix(quick=args.quick, examples=args.examples)
    if args.json:
        import sys

        write_json(rows, sys.stdout)
    elif args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
