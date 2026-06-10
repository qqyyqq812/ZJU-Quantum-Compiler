"""Replay and verify NPQR action traces."""
from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.env import QuantumRoutingEnv
from src.compiler.light_env import LightweightEnv


@dataclass(frozen=True)
class ReplayEvent:
    kind: str
    physical_qubits: tuple[int, ...]
    gate_index: int | None = None
    action: int | None = None


@dataclass(frozen=True)
class ReplayResult:
    status: str
    completed: bool
    total_swaps: int
    executed_gates: int
    trace_len: int
    events: tuple[ReplayEvent, ...]
    final_mapping: dict[int, int]
    message: str


@dataclass(frozen=True)
class TopologyResult:
    passed: bool
    invalid_ops: tuple[dict, ...]


def _build_env(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    initial_mapping: dict[int, int],
    max_steps: int,
) -> LightweightEnv:
    def mapping_fn(_circuit, _coupling_map):
        return dict(initial_mapping)

    base_env = QuantumRoutingEnv(
        coupling_map=coupling_map,
        max_steps=max_steps,
        soft_mask=True,
        initial_mapping_fn=mapping_fn,
    )
    base_env.set_circuit(circuit)
    base_env.reset()
    return LightweightEnv(base_env)


def replay_action_trace(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    initial_mapping: dict[int, int],
    action_trace: list[int],
    max_steps: int,
) -> ReplayResult:
    """Replay an NPQR trace without invoking SABRE or fallback."""
    env = _build_env(circuit, coupling_map, initial_mapping, max_steps=max_steps)
    events: list[ReplayEvent] = []
    previous_executed = env.executed.copy()

    for action in action_trace:
        if action < 0 or action >= env.n_actions:
            return ReplayResult(
                status="INVALID_ACTION",
                completed=False,
                total_swaps=int(env._total_swaps),
                executed_gates=int(env._total_gates_executed),
                trace_len=len(action_trace),
                events=tuple(events),
                final_mapping=dict(env._mapping),
                message=f"Action {action} is outside the action space.",
            )
        if action < env.n_swap_actions:
            events.append(
                ReplayEvent(
                    kind="swap",
                    physical_qubits=tuple(int(q) for q in env.swap_edges[action]),
                    action=int(action),
                )
            )
        _, _, terminated, truncated, _ = env.step(int(action))
        newly_executed = [
            gid
            for gid, done in enumerate(env.executed)
            if bool(done) and not bool(previous_executed[gid])
        ]
        for gid in newly_executed:
            logical = env.gate_qubits[gid]
            physical = tuple(int(env._mapping[q]) for q in logical)
            events.append(ReplayEvent(kind="gate", physical_qubits=physical, gate_index=int(gid)))
        previous_executed = env.executed.copy()
        if terminated or truncated:
            break

    completed = bool(env.is_done())
    return ReplayResult(
        status="OK" if completed else "INCOMPLETE",
        completed=completed,
        total_swaps=int(env._total_swaps),
        executed_gates=int(env._total_gates_executed),
        trace_len=len(action_trace),
        events=tuple(events),
        final_mapping=dict(env._mapping),
        message=(
            "Trace replay completed the circuit."
            if completed
            else "Trace replay stopped before completing the circuit."
        ),
    )


def build_routed_circuit_from_replay(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    replay: ReplayResult,
) -> QuantumCircuit:
    routed = QuantumCircuit(coupling_map.size(), name=f"{circuit.name or 'circuit'}_npqr_replay")
    for event in replay.events:
        if event.kind == "swap":
            p0, p1 = event.physical_qubits
            routed.swap(p0, p1)
        elif event.kind == "gate" and event.gate_index is not None:
            operation = circuit.data[event.gate_index].operation
            routed.append(operation.copy(), list(event.physical_qubits), [])
    return routed


def verify_routed_circuit_topology(
    routed: QuantumCircuit,
    coupling_map: CouplingMap,
) -> TopologyResult:
    edges = {tuple(edge) for edge in coupling_map.get_edges()}
    undirected = edges | {(b, a) for a, b in edges}
    invalid = []
    for index, instruction in enumerate(routed.data):
        qubits = [routed.find_bit(qubit).index for qubit in instruction.qubits]
        if len(qubits) != 2:
            continue
        pair = (int(qubits[0]), int(qubits[1]))
        if pair not in undirected:
            invalid.append(
                {
                    "index": index,
                    "operation": instruction.operation.name,
                    "qubits": list(pair),
                }
            )
    return TopologyResult(passed=not invalid, invalid_ops=tuple(invalid))
