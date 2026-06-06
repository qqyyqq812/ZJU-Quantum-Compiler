"""Online NPQR runtime: neural selector beam plus bounded suffix repair."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.env import QuantumRoutingEnv
from src.compiler.light_env import LightweightEnv
from src.compiler.neural_planner import NeuralPlannerConfig, NeuralPlanningRouter
from src.compiler.npqr_trace import (
    ReplayResult,
    build_routed_circuit_from_replay,
    replay_action_trace,
    verify_routed_circuit_topology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPQR_MODEL = (
    PROJECT_ROOT
    / "models"
    / "npqr_overnight_20260603"
    / "wave2_stage2_e120_lr5e5_s51_h02.pt"
)


@dataclass(frozen=True)
class NPQRRuntimeConfig:
    """Bounded runtime settings for the deployed NPQR route."""

    max_steps: int = 45
    beam_width: int = 4
    branch_factor: int = 3
    perturbation_count: int = 6
    qap_local_search_rounds: int = 2
    primary_selector_top_k: int = 2
    rescue_selector_top_k: int = 4
    suffix_depth: int = 6
    suffix_max_nodes: int = 4000
    suffix_action_limit: int = 12


@dataclass(frozen=True)
class SuffixRepairResult:
    found: bool
    suffix_actions: tuple[int, ...]
    nodes: int
    total_swaps: int
    executed_gates: int
    trace_len: int


@dataclass(frozen=True)
class NPQRRuntimeResult:
    algorithm: str
    status: str
    completed: bool
    total_swaps: int | None
    depth: int | None
    executed_gates: int
    trace_len: int
    elapsed_ms: float
    model_path: str
    message: str
    compiled_circuit: QuantumCircuit | None
    compiled_qasm: str | None
    replay: ReplayResult | None
    components: dict[str, bool | int | None] = field(default_factory=dict)
    initial_mapping: dict[int, int] = field(default_factory=dict)
    final_mapping: dict[int, int] = field(default_factory=dict)
    action_trace: tuple[int, ...] = ()


def _planner_config(config: NPQRRuntimeConfig, *, selector_top_k: int) -> NeuralPlannerConfig:
    return NeuralPlannerConfig(
        max_steps=config.max_steps,
        beam_width=config.beam_width,
        branch_factor=config.branch_factor,
        perturbation_count=config.perturbation_count,
        mapping_strategy="selector",
        qap_local_search_rounds=config.qap_local_search_rounds,
        selector_top_k=selector_top_k,
    )


def _build_light_env(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    initial_mapping: dict[int, int],
    max_steps: int,
) -> LightweightEnv:
    def mapping_fn(_circuit: QuantumCircuit, _coupling_map: CouplingMap) -> dict[int, int]:
        return dict(initial_mapping)

    base_env = QuantumRoutingEnv(
        coupling_map=coupling_map,
        max_steps=max_steps,
        soft_mask=True,
        initial_mapping_fn=mapping_fn,
        use_sabre_reward=False,
    )
    base_env.set_circuit(circuit)
    base_env.reset()
    return LightweightEnv(base_env)


def _replay_to_light_env(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    initial_mapping: dict[int, int],
    action_trace: tuple[int, ...],
    max_steps: int,
) -> LightweightEnv:
    env = _build_light_env(circuit, coupling_map, initial_mapping, max_steps=max_steps)
    for action in action_trace:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return env


def _front_distance_sum(env: LightweightEnv) -> float:
    total = 0.0
    for gid in env.two_qubit_front:
        q0, q1 = env.gate_qubits[gid]
        p0 = env._mapping.get(q0, q0)
        p1 = env._mapping.get(q1, q1)
        if p0 < env.n_physical and p1 < env.n_physical:
            total += float(env._dist_matrix[p0][p1])
    return total


def _valid_suffix_actions(env: LightweightEnv, *, limit: int) -> list[int]:
    rows: list[tuple[tuple[bool, int, float, int, int], int]] = []
    mask = env.get_action_mask()
    for action, valid in enumerate(mask):
        if valid <= 0:
            continue
        probe = env.clone()
        before_gates = int(probe._total_gates_executed)
        before_swaps = int(probe._total_swaps)
        before_front = _front_distance_sum(probe)
        _, _, terminated, _truncated, _ = probe.step(int(action))
        key = (
            not bool(terminated),
            -(int(probe._total_gates_executed) - before_gates),
            _front_distance_sum(probe) - before_front,
            int(probe._total_swaps) - before_swaps,
            int(action),
        )
        rows.append((key, int(action)))
    rows.sort(key=lambda item: item[0])
    return [action for _, action in rows[:limit]]


def _state_key(env: LightweightEnv) -> tuple[tuple[tuple[int, int], ...], tuple[bool, ...], int]:
    return (
        tuple(sorted((int(k), int(v)) for k, v in env._mapping.items())),
        tuple(bool(done) for done in env.executed),
        int(env._total_swaps),
    )


def search_completion_suffix(
    start_env: LightweightEnv,
    *,
    max_depth: int,
    max_nodes: int,
    action_limit: int,
) -> SuffixRepairResult:
    """Bounded BFS suffix repair used after the neural beam stalls."""
    queue = deque([(start_env.clone(), tuple())])
    seen = {_state_key(start_env)}
    nodes = 0
    best = start_env
    while queue and nodes < max_nodes:
        env, suffix = queue.popleft()
        nodes += 1
        if env.is_done():
            return SuffixRepairResult(
                found=True,
                suffix_actions=tuple(int(action) for action in suffix),
                nodes=nodes,
                total_swaps=int(env._total_swaps),
                executed_gates=int(env._total_gates_executed),
                trace_len=len(suffix),
            )
        if int(env._total_gates_executed) > int(best._total_gates_executed):
            best = env
        if len(suffix) >= max_depth:
            continue
        for action in _valid_suffix_actions(env, limit=action_limit):
            child = env.clone()
            child.step(int(action))
            key = _state_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, (*suffix, int(action))))
    return SuffixRepairResult(
        found=False,
        suffix_actions=(),
        nodes=nodes,
        total_swaps=int(best._total_swaps),
        executed_gates=int(best._total_gates_executed),
        trace_len=0,
    )


class NPQRRuntime:
    """Product runtime for the neural-assisted NPQR router."""

    def __init__(
        self,
        coupling_map: CouplingMap,
        model_path: str | Path = DEFAULT_NPQR_MODEL,
        config: NPQRRuntimeConfig | None = None,
    ) -> None:
        self.coupling_map = coupling_map
        self.model_path = Path(model_path)
        self.config = config or NPQRRuntimeConfig()
        self.primary_router = NeuralPlanningRouter(
            coupling_map,
            model_path=str(self.model_path),
            config=_planner_config(self.config, selector_top_k=self.config.primary_selector_top_k),
        )
        self.rescue_router = NeuralPlanningRouter(
            coupling_map,
            model_path=str(self.model_path),
            config=_planner_config(self.config, selector_top_k=self.config.rescue_selector_top_k),
        )

    @property
    def has_model(self) -> bool:
        return bool(self.primary_router._has_model and self.rescue_router._has_model)

    @property
    def model_load_error(self) -> str | None:
        return self.primary_router.model_load_error or self.rescue_router.model_load_error

    def compile(self, circuit: QuantumCircuit) -> NPQRRuntimeResult:
        started = time.perf_counter()
        if not self.has_model:
            return NPQRRuntimeResult(
                algorithm="npqr_neural_selector_suffix_v1",
                status="N/A",
                completed=False,
                total_swaps=None,
                depth=None,
                executed_gates=0,
                trace_len=0,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                model_path=str(self.model_path),
                message=self.model_load_error or "NPQR model is not loadable.",
                compiled_circuit=None,
                compiled_qasm=None,
                replay=None,
                components={
                    "neural_beam": False,
                    "mapping_selector": True,
                    "suffix_repair": False,
                    "sabre_fallback": False,
                    "primary_selector_top_k": self.config.primary_selector_top_k,
                    "rescue_selector_top_k": self.config.rescue_selector_top_k,
                },
            )

        primary = self.primary_router.route_count_only(circuit)
        selected = primary
        selected_top_k = self.config.primary_selector_top_k
        rescue_used = False
        if not primary.completed:
            rescue = self.rescue_router.route_count_only(circuit)
            selected = rescue
            selected_top_k = self.config.rescue_selector_top_k
            rescue_used = True

        suffix = SuffixRepairResult(
            found=False,
            suffix_actions=(),
            nodes=0,
            total_swaps=int(selected.total_swaps if selected.total_swaps >= 0 else 0),
            executed_gates=int(selected.executed_gates),
            trace_len=0,
        )
        final_trace = tuple(int(action) for action in selected.action_trace)
        suffix_attempted = False
        if not selected.completed and selected.initial_mapping and final_trace:
            suffix_attempted = True
            env = _replay_to_light_env(
                circuit,
                self.coupling_map,
                selected.initial_mapping,
                final_trace,
                max_steps=max(self.config.max_steps, len(final_trace) + self.config.suffix_depth),
            )
            suffix = search_completion_suffix(
                env,
                max_depth=self.config.suffix_depth,
                max_nodes=self.config.suffix_max_nodes,
                action_limit=self.config.suffix_action_limit,
            )
            if suffix.found:
                final_trace = (*final_trace, *suffix.suffix_actions)

        replay = replay_action_trace(
            circuit,
            self.coupling_map,
            selected.initial_mapping,
            list(final_trace),
            max_steps=max(self.config.max_steps, len(final_trace) + 1),
        )
        compiled_circuit = build_routed_circuit_from_replay(circuit, self.coupling_map, replay)
        topology_check = verify_routed_circuit_topology(compiled_circuit, self.coupling_map)
        if not topology_check.passed:
            return NPQRRuntimeResult(
                algorithm="npqr_neural_selector_suffix_v1",
                status="INCOMPLETE",
                completed=False,
                total_swaps=replay.total_swaps,
                depth=None,
                executed_gates=replay.executed_gates,
                trace_len=len(final_trace),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                model_path=str(self.model_path),
                message="NPQR replay produced an invalid topology; result withheld.",
                compiled_circuit=None,
                compiled_qasm=None,
                replay=replay,
                components=self._components(selected_top_k, rescue_used, suffix_attempted, suffix),
                initial_mapping=selected.initial_mapping,
                final_mapping=replay.final_mapping,
                action_trace=final_trace,
            )

        return NPQRRuntimeResult(
            algorithm="npqr_neural_selector_suffix_v1",
            status="OK" if replay.completed else "INCOMPLETE",
            completed=bool(replay.completed),
            total_swaps=int(replay.total_swaps),
            depth=compiled_circuit.depth(),
            executed_gates=int(replay.executed_gates),
            trace_len=len(final_trace),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model_path=str(self.model_path),
            message=(
                "NPQR neural-assisted router completed without SABRE fallback."
                if replay.completed
                else "NPQR neural-assisted router stopped incomplete; SABRE baseline is comparison only."
            ),
            compiled_circuit=compiled_circuit,
            compiled_qasm=None,
            replay=replay,
            components=self._components(selected_top_k, rescue_used, suffix_attempted, suffix),
            initial_mapping=selected.initial_mapping,
            final_mapping=replay.final_mapping,
            action_trace=final_trace,
        )

    def _components(
        self,
        selected_top_k: int,
        rescue_used: bool,
        suffix_attempted: bool,
        suffix: SuffixRepairResult,
    ) -> dict[str, bool | int | None]:
        return {
            "neural_beam": True,
            "mapping_selector": True,
            "suffix_repair": bool(suffix.found),
            "suffix_repair_attempted": bool(suffix_attempted),
            "suffix_nodes": int(suffix.nodes),
            "selector_rescue_used": bool(rescue_used),
            "selected_selector_top_k": int(selected_top_k),
            "sabre_fallback": False,
        }
