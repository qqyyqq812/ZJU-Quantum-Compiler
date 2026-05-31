"""Small bounded-search experiment for quantum routing.

This script is intentionally standalone and local-only. It compares Qiskit
SabreSwap heuristics with a deterministic, non-neural bounded look-ahead search
prototype on the public checked-in QASM examples.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import SwapGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreSwap,
    TrivialLayout,
)

from src.benchmarks.topologies import get_topology
from src.compiler.dag import CircuitDAG
from src.compiler.env import QuantumRoutingEnv

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


@dataclass
class Result:
    example: str
    method: str
    completed: bool
    swaps: int
    depth: int | None
    two_qubit_gates: int | None
    runtime_ms: float
    trace_len: int = 0
    best_reason: str = "-"


def _compile_sabre(circuit: QuantumCircuit, coupling_map, heuristic: str) -> QuantumCircuit:
    pass_manager = PassManager(
        [
            TrivialLayout(coupling_map),
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(coupling_map, heuristic=heuristic, seed=42, trials=1),
        ]
    )
    routed = pass_manager.run(circuit)
    return transpile(
        routed,
        basis_gates=BASIS_GATES,
        optimization_level=0,
        seed_transpiler=42,
    )


def _front_distance(env: QuantumRoutingEnv) -> float:
    return float(env._compute_front_distance())


def _extended_distance(env: QuantumRoutingEnv) -> float:
    return float(env._compute_extended_distance())


def _distance_for_mapping(env: QuantumRoutingEnv, mapping: dict[int, int], extended: bool) -> float:
    if env._dag is None:
        return 0.0
    gates = env._dag.get_extended_front(depth=2) if extended else env._dag.get_two_qubit_front()
    total = 0.0
    for gate in gates:
        p0 = mapping.get(gate.qubits[0], gate.qubits[0])
        p1 = mapping.get(gate.qubits[1], gate.qubits[1])
        if p0 < env.n_physical and p1 < env.n_physical:
            total += env._dist_matrix[p0][p1]
    return float(total)


def _executable_front_count(env: QuantumRoutingEnv, mapping: dict[int, int]) -> int:
    if env._dag is None:
        return 0
    edges = set(tuple(edge) for edge in env.coupling_map.get_edges())
    count = 0
    for gate in env._dag.get_front_layer():
        if not gate.is_two_qubit:
            count += 1
            continue
        p0 = mapping[gate.qubits[0]]
        p1 = mapping[gate.qubits[1]]
        if (p0, p1) in edges or (p1, p0) in edges:
            count += 1
    return count


def _static_action_score(env: QuantumRoutingEnv, action: int) -> tuple[float, str]:
    p1, p2 = env.swap_edges[action]
    new_mapping = CircuitDAG.apply_swap(p1, p2, env._mapping)
    front_before = _distance_for_mapping(env, env._mapping, extended=False)
    ext_before = _distance_for_mapping(env, env._mapping, extended=True)
    front_after = _distance_for_mapping(env, new_mapping, extended=False)
    ext_after = _distance_for_mapping(env, new_mapping, extended=True)
    executable_gain = _executable_front_count(env, new_mapping)
    score = (
        executable_gain * 100.0
        + (front_before - front_after) * 10.0
        + (ext_before - ext_after) * 2.0
        - 10.0
    )
    reason = (
        f"swap {(p1, p2)}, path_len=1, score={score:.1f}, "
        f"exec_front={executable_gain}, front={front_after:.1f}"
    )
    return score, reason


def _valid_swap_actions(env: QuantumRoutingEnv, branch_factor: int) -> list[int]:
    candidate_actions: set[int] = set()
    front = env._dag.get_extended_front(depth=2) if env._dag is not None else []
    for gate in front:
        if not gate.is_two_qubit:
            continue
        p0 = env._mapping.get(gate.qubits[0], gate.qubits[0])
        p1 = env._mapping.get(gate.qubits[1], gate.qubits[1])
        current = env._dist_matrix[p0][p1]
        for action, (s1, s2) in enumerate(env.swap_edges):
            new_p0 = s2 if p0 == s1 else (s1 if p0 == s2 else p0)
            new_p1 = s2 if p1 == s1 else (s1 if p1 == s2 else p1)
            if env._dist_matrix[new_p0][new_p1] <= current:
                candidate_actions.add(action)
    if not candidate_actions:
        candidate_actions = set(range(env.n_swap_actions))

    scored: list[tuple[float, int]] = []
    for action in candidate_actions:
        score, _ = _static_action_score(env, action)
        scored.append((score, action))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [action for _, action in scored[:branch_factor]]


def _state_score(
    env: QuantumRoutingEnv,
    executed_gate_gain: int,
    added_swaps: int,
) -> float:
    remaining = env._dag.remaining_gates() if env._dag is not None else 0
    return (
        executed_gate_gain * 100.0
        - added_swaps * 10.0
        - _front_distance(env)
        - 0.5 * _extended_distance(env)
        - remaining * 0.1
    )


def _step_pass_until_blocked(env: QuantumRoutingEnv) -> int:
    executed_total = 0
    while env._dag is not None and not env._dag.is_done():
        before = env._total_gates_executed
        _, _, terminated, truncated, _ = env.step(env.PASS_ACTION)
        executed = env._total_gates_executed - before
        executed_total += executed
        if terminated or truncated or executed == 0:
            break
    return executed_total


def _execute_ready_gates(env: QuantumRoutingEnv, trace: list[tuple]) -> int:
    if env._dag is None:
        return 0
    executed = env._dag.execute_executable(env._mapping, env.coupling_map)
    env._total_gates_executed += len(executed)
    for gate in executed:
        trace.append(("gate", gate.gate_id))
    return len(executed)


def _choose_bounded_action(
    env: QuantumRoutingEnv,
    branch_factor: int,
    max_depth: int,
) -> tuple[int | None, str]:
    candidates = _valid_swap_actions(env, branch_factor)
    if not candidates:
        return None, "no valid swap action"
    if max_depth <= 1:
        action = candidates[0]
        _, reason = _static_action_score(env, action)
        return action, reason

    beam: list[tuple[float, QuantumRoutingEnv, list[int], int]] = []
    for action in candidates:
        candidate = copy.deepcopy(env)
        before_executed = candidate._total_gates_executed
        candidate.step(action)
        executed = candidate._total_gates_executed - before_executed
        score = _state_score(candidate, executed, added_swaps=1)
        beam.append((score, candidate, [action], executed))

    best_score, best_env, best_path, best_executed = max(beam, key=lambda row: row[0])
    for depth in range(2, max_depth + 1):
        next_beam: list[tuple[float, QuantumRoutingEnv, list[int], int]] = []
        for _, current_env, path, executed_so_far in beam:
            for action in _valid_swap_actions(current_env, branch_factor):
                candidate = copy.deepcopy(current_env)
                before_executed = candidate._total_gates_executed
                candidate.step(action)
                executed = candidate._total_gates_executed - before_executed
                total_executed = executed_so_far + executed
                score = _state_score(candidate, total_executed, added_swaps=len(path) + 1)
                next_beam.append((score, candidate, [*path, action], total_executed))
        if not next_beam:
            break
        next_beam.sort(key=lambda row: row[0], reverse=True)
        beam = next_beam[:branch_factor]
        depth_best = beam[0]
        if depth_best[0] > best_score:
            best_score, best_env, best_path, best_executed = depth_best

    action = best_path[0]
    edge = env.swap_edges[action]
    reason = (
        f"swap {edge}, path_len={len(best_path)}, "
        f"score={best_score:.1f}, executed_gain={best_executed}, "
        f"front={_front_distance(best_env):.1f}"
    )
    return action, reason


def run_bounded_search(
    example: str,
    circuit: QuantumCircuit,
    coupling_map,
    branch_factor: int,
    max_depth: int,
    max_steps: int,
) -> Result:
    started = time.perf_counter()
    env = QuantumRoutingEnv(
        coupling_map=coupling_map,
        max_steps=max_steps,
        soft_mask=False,
        use_sabre_reward=False,
        tabu_size=0,
    )
    env.set_circuit(circuit, topology_name="ibm_tokyo")
    env.reset(seed=42)
    trace: list[tuple] = []
    reasons: list[str] = []
    if env._dag is not None:
        for gate in env._dag._gates.values():
            if gate.executed:
                trace.append(("gate", gate.gate_id))

    while env._dag is not None and not env._dag.is_done() and env._step_count < max_steps:
        _execute_ready_gates(env, trace)
        if env._dag.is_done():
            break
        action, reason = _choose_bounded_action(env, branch_factor, max_depth)
        if action is None:
            reasons.append(reason)
            break
        p1, p2 = env.swap_edges[action]
        env._mapping = CircuitDAG.apply_swap(p1, p2, env._mapping)
        env._total_swaps += 1
        env._step_count += 1
        trace.append(("swap", p1, p2))
        _execute_ready_gates(env, trace)
        reasons.append(reason)

    elapsed = (time.perf_counter() - started) * 1000
    completed = bool(env._dag is not None and env._dag.is_done())
    routed = _build_routed_circuit(circuit, trace, coupling_map.size())
    routed_ops = dict(routed.count_ops())
    return Result(
        example=example,
        method="bounded",
        completed=completed,
        swaps=env._total_swaps,
        depth=routed.depth(),
        two_qubit_gates=int(routed_ops.get("cx", 0) + routed_ops.get("swap", 0) * 3),
        runtime_ms=elapsed,
        trace_len=len(reasons),
        best_reason=reasons[0] if reasons else "completed without swaps",
    )


def _build_routed_circuit(
    original: QuantumCircuit,
    trace: list[tuple],
    n_physical: int,
) -> QuantumCircuit:
    original_ops = list(original.data)
    compiled = QuantumCircuit(n_physical, original.num_clbits)
    mapping: dict[int, int] = {i: i for i in range(original.num_qubits)}
    for event in trace:
        if event[0] == "swap":
            _, p1, p2 = event
            compiled.append(SwapGate(), [p1, p2])
            mapping = CircuitDAG.apply_swap(p1, p2, mapping)
        elif event[0] == "gate":
            gid = event[1]
            if gid >= len(original_ops):
                continue
            instruction = original_ops[gid]
            operation = instruction.operation
            phys_qubits = [
                mapping.get(original.find_bit(qubit).index, original.find_bit(qubit).index)
                for qubit in instruction.qubits
            ]
            clbits = [original.find_bit(clbit).index for clbit in instruction.clbits]
            compiled.append(operation, phys_qubits, clbits)
    return compiled


def run_sabre(example: str, circuit: QuantumCircuit, coupling_map, heuristic: str) -> Result:
    started = time.perf_counter()
    compiled = _compile_sabre(circuit, coupling_map, heuristic)
    elapsed = (time.perf_counter() - started) * 1000
    ops = dict(compiled.count_ops())
    return Result(
        example=example,
        method=heuristic,
        completed=True,
        swaps=int(ops.get("swap", 0)),
        depth=compiled.depth(),
        two_qubit_gates=int(ops.get("cx", 0) + ops.get("swap", 0) * 3),
        runtime_ms=elapsed,
    )


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def print_results(results: list[Result]) -> None:
    print(
        "| example | method | completed | swaps | depth | 2q | ms | trace | reason |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        depth = "-" if row.depth is None else str(row.depth)
        twoq = "-" if row.two_qubit_gates is None else str(row.two_qubit_gates)
        print(
            f"| {row.example} | {row.method} | {_format_bool(row.completed)} "
            f"| {row.swaps} | {depth} | {twoq} | {row.runtime_ms:.1f} "
            f"| {row.trace_len} | {row.best_reason} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        nargs="+",
        default=["qft5", "qaoa5", "ghz5"],
        choices=sorted(EXAMPLES),
    )
    parser.add_argument("--branch-factor", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coupling_map = get_topology("ibm_tokyo")
    results: list[Result] = []
    for example in args.examples:
        circuit = QuantumCircuit.from_qasm_file(str(Path(EXAMPLES[example])))
        for heuristic in ("basic", "lookahead", "decay"):
            results.append(run_sabre(example, circuit, coupling_map, heuristic))
        results.append(
            run_bounded_search(
                example,
                circuit,
                coupling_map,
                branch_factor=args.branch_factor,
                max_depth=args.max_depth,
                max_steps=args.max_steps,
            )
        )
    print_results(results)


if __name__ == "__main__":
    main()
