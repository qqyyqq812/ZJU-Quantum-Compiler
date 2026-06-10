"""Online NPQR runtime: neural selector beam plus bounded suffix repair."""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.env import QuantumRoutingEnv
from src.compiler.light_env import LightweightEnv
from src.compiler.neural_planner import (
    NeuralPlannerConfig,
    NeuralPlanningRouter,
    generate_initial_mapping_candidates,
)
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
    / "default"
    / "npqr-default.pt"
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
    frontier_rescue_enabled: bool = False
    frontier_rescue_precheck_enabled: bool = True
    frontier_max_steps: int = 120
    frontier_action_limit: int = 60
    frontier_action_pruning_policy: str | None = None
    frontier_max_candidates: int = 2
    frontier_selector_top_k: int = 2
    frontier_min_qubits: int = 10
    frontier_min_cx_like: int = 40
    frontier_min_unique_pairs: int = 25
    frontier_min_unique_pair_ratio: float = 0.55
    frontier_min_pair_entropy: float = 4.0
    frontier_max_depth: int = 30
    frontier_max_cx_like: int = 100
    frontier_max_repeat_pair_ratio: float = 0.55


@dataclass(frozen=True)
class SuffixRepairResult:
    found: bool
    suffix_actions: tuple[int, ...]
    nodes: int
    total_swaps: int
    executed_gates: int
    trace_len: int


@dataclass(frozen=True)
class FrontierRescueResult:
    completed: bool
    total_swaps: int
    executed_gates: int
    trace_len: int
    steps: int
    initial_mapping: dict[int, int]
    final_mapping: dict[int, int]
    action_trace: tuple[int, ...]
    candidate_count: int
    message: str


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
    components: dict[str, Any] = field(default_factory=dict)
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


def _frontier_state_key(env: LightweightEnv) -> tuple[tuple[tuple[int, int], ...], tuple[bool, ...]]:
    return (
        tuple(sorted((int(k), int(v)) for k, v in env._mapping.items())),
        tuple(bool(done) for done in env.executed),
    )


def _two_qubit_pairs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in {"cx", "cz", "cy", "cp", "swap"} or len(instruction.qubits) != 2:
            continue
        q0 = circuit.find_bit(instruction.qubits[0]).index
        q1 = circuit.find_bit(instruction.qubits[1]).index
        pairs.append(tuple(sorted((int(q0), int(q1)))))
    return pairs


def circuit_interaction_metrics(circuit: QuantumCircuit) -> dict[str, float | int]:
    pairs = _two_qubit_pairs(circuit)
    counts = Counter(pairs)
    total = len(pairs)
    entropy = 0.0
    if total:
        entropy = sum(-(count / total) * log2(count / total) for count in counts.values())
    return {
        "qubits": int(circuit.num_qubits),
        "size": int(circuit.size()),
        "depth": int(circuit.depth()),
        "cx_like_count": int(total),
        "unique_two_qubit_pairs": int(len(counts)),
        "pair_entropy": round(float(entropy), 6),
        "repeat_pair_ratio": round(1.0 - len(counts) / total, 6) if total else 0.0,
    }


def should_precheck_frontier_rescue(
    circuit: QuantumCircuit,
    config: NPQRRuntimeConfig,
) -> tuple[bool, dict[str, float | int | bool]]:
    metrics = circuit_interaction_metrics(circuit)
    unique_pair_ratio = (
        float(metrics["unique_two_qubit_pairs"]) / float(metrics["cx_like_count"])
        if int(metrics["cx_like_count"]) > 0
        else 0.0
    )
    triggered = (
        int(metrics["qubits"]) >= config.frontier_min_qubits
        and int(metrics["cx_like_count"]) >= config.frontier_min_cx_like
        and int(metrics["cx_like_count"]) <= config.frontier_max_cx_like
        and int(metrics["unique_two_qubit_pairs"]) >= config.frontier_min_unique_pairs
        and unique_pair_ratio >= config.frontier_min_unique_pair_ratio
        and float(metrics["pair_entropy"]) >= config.frontier_min_pair_entropy
        and int(metrics["depth"]) <= config.frontier_max_depth
        and float(metrics["repeat_pair_ratio"]) <= config.frontier_max_repeat_pair_ratio
    )
    return bool(triggered), {
        **metrics,
        "unique_pair_ratio": round(unique_pair_ratio, 6),
        "triggered": bool(triggered),
        "min_qubits": config.frontier_min_qubits,
        "min_cx_like": config.frontier_min_cx_like,
        "max_cx_like": config.frontier_max_cx_like,
        "min_unique_two_qubit_pairs": config.frontier_min_unique_pairs,
        "min_unique_pair_ratio": config.frontier_min_unique_pair_ratio,
        "min_pair_entropy": config.frontier_min_pair_entropy,
        "max_depth": config.frontier_max_depth,
        "max_repeat_pair_ratio": config.frontier_max_repeat_pair_ratio,
    }


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
        frontier_attempted = False
        frontier_precheck_triggered = False
        frontier_candidate_count = 0
        frontier_precheck: dict[str, float | int | bool] | None = None
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
                    "frontier_rescue": False,
                    "frontier_rescue_attempted": False,
                    "frontier_precheck_triggered": False,
                    "frontier_action_pruning": False,
                    "frontier_action_pruning_policy": self.config.frontier_action_pruning_policy,
                    "sabre_fallback": False,
                    "primary_selector_top_k": self.config.primary_selector_top_k,
                    "rescue_selector_top_k": self.config.rescue_selector_top_k,
                },
            )

        if self.config.frontier_rescue_enabled and self.config.frontier_rescue_precheck_enabled:
            frontier_precheck_triggered, frontier_precheck = should_precheck_frontier_rescue(
                circuit,
                self.config,
            )
            if frontier_precheck_triggered:
                frontier_attempted = True
                frontier = self._run_frontier_rescue(circuit)
                frontier_candidate_count = frontier.candidate_count
                if frontier.completed:
                    return self._result_from_trace(
                        algorithm="npqr_neural_selector_suffix_frontier_v1",
                        started=started,
                        circuit=circuit,
                        initial_mapping=frontier.initial_mapping,
                        final_trace=frontier.action_trace,
                        message="NPQR frontier-objective rescue completed without SABRE fallback.",
                        components=self._components(
                            selected_top_k=self.config.frontier_selector_top_k,
                            rescue_used=False,
                            suffix_attempted=False,
                            suffix=SuffixRepairResult(False, (), 0, 0, 0, 0),
                            frontier_attempted=True,
                            frontier_precheck_triggered=True,
                            frontier_completed=True,
                            frontier_candidate_count=frontier_candidate_count,
                            frontier_precheck=frontier_precheck,
                        ),
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
                components=self._components(
                    selected_top_k,
                    rescue_used,
                    suffix_attempted,
                    suffix,
                    frontier_attempted=frontier_attempted,
                    frontier_precheck_triggered=frontier_precheck_triggered,
                    frontier_completed=False,
                    frontier_candidate_count=frontier_candidate_count,
                    frontier_precheck=frontier_precheck,
                ),
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
            components=self._components(
                selected_top_k,
                rescue_used,
                suffix_attempted,
                suffix,
                frontier_attempted=frontier_attempted,
                frontier_precheck_triggered=frontier_precheck_triggered,
                frontier_completed=False,
                frontier_candidate_count=frontier_candidate_count,
                frontier_precheck=frontier_precheck,
            ),
            initial_mapping=selected.initial_mapping,
            final_mapping=replay.final_mapping,
            action_trace=final_trace,
        )

    def _run_frontier_rescue(self, circuit: QuantumCircuit) -> FrontierRescueResult:
        mappings = generate_initial_mapping_candidates(
            circuit,
            self.coupling_map,
            perturbation_count=self.config.perturbation_count,
            strategy="selector",
            qap_local_search_rounds=self.config.qap_local_search_rounds,
            selector_top_k=self.config.frontier_selector_top_k,
        )[: max(1, self.config.frontier_max_candidates)]
        best: FrontierRescueResult | None = None
        for mapping in mappings:
            result = self._route_mapping_with_frontier_objective(circuit, mapping)
            if self._is_better_frontier_result(result, best):
                best = result
        if best is None:
            return FrontierRescueResult(
                completed=False,
                total_swaps=0,
                executed_gates=0,
                trace_len=0,
                steps=0,
                initial_mapping={},
                final_mapping={},
                action_trace=(),
                candidate_count=0,
                message="No frontier rescue candidate mapping was generated.",
            )
        return FrontierRescueResult(
            completed=best.completed,
            total_swaps=best.total_swaps,
            executed_gates=best.executed_gates,
            trace_len=best.trace_len,
            steps=best.steps,
            initial_mapping=best.initial_mapping,
            final_mapping=best.final_mapping,
            action_trace=best.action_trace,
            candidate_count=len(mappings),
            message=best.message,
        )

    def _route_mapping_with_frontier_objective(
        self,
        circuit: QuantumCircuit,
        initial_mapping: dict[int, int],
    ) -> FrontierRescueResult:
        env = _build_light_env(
            circuit,
            self.coupling_map,
            initial_mapping,
            max_steps=self.config.frontier_max_steps,
        )
        trace: list[int] = []
        seen_states = {_frontier_state_key(env)}
        recent_actions: deque[int] = deque(maxlen=6)
        while not env.is_done() and env._step_count < env.max_steps:
            self._advance_frontier_productive_passes(env, trace)
            seen_states.add(_frontier_state_key(env))
            if env.is_done() or env._step_count >= env.max_steps:
                break
            actions = self._valid_frontier_swap_actions(env)
            if not actions:
                break
            action = min(
                actions,
                key=lambda candidate: self._score_frontier_action(
                    env,
                    candidate,
                    seen_states=seen_states,
                    recent_actions=recent_actions,
                ),
            )
            _, _, terminated, truncated, _ = env.step(int(action))
            trace.append(int(action))
            recent_actions.append(int(action))
            seen_states.add(_frontier_state_key(env))
            if terminated or truncated:
                break
        return FrontierRescueResult(
            completed=bool(env.is_done()),
            total_swaps=int(env._total_swaps),
            executed_gates=int(env._total_gates_executed),
            trace_len=len(trace),
            steps=int(env._step_count),
            initial_mapping=dict(initial_mapping),
            final_mapping=dict(env._mapping),
            action_trace=tuple(int(action) for action in trace),
            candidate_count=1,
            message=(
                "Frontier-objective rescue completed."
                if env.is_done()
                else "Frontier-objective rescue stopped incomplete."
            ),
        )

    def _advance_frontier_productive_passes(self, env: LightweightEnv, trace: list[int]) -> None:
        while self._frontier_pass_would_progress(env):
            _, _, terminated, truncated, _ = env.step(env.PASS_ACTION)
            trace.append(int(env.PASS_ACTION))
            if terminated or truncated:
                break

    def _frontier_pass_would_progress(self, env: LightweightEnv) -> bool:
        if env.is_done() or env._step_count >= env.max_steps:
            return False
        probe = env.clone()
        before = int(probe._total_gates_executed)
        probe.step(probe.PASS_ACTION)
        return int(probe._total_gates_executed) > before

    def _valid_frontier_swap_actions(self, env: LightweightEnv) -> list[int]:
        mask = env.get_action_mask()
        actions = [int(action) for action, valid in enumerate(mask[: env.n_swap_actions]) if valid > 0]
        actions = actions[: max(1, int(self.config.frontier_action_limit))]
        return self._prune_frontier_swap_actions(env, actions)

    def _frontier_action_pruning_spec(self) -> tuple[str, int] | None:
        policy = self.config.frontier_action_pruning_policy
        if not policy:
            return None
        parts = policy.split("_")
        if len(parts) != 3 or parts[1] != "touch" or parts[0] not in {"frontier", "extended"}:
            raise ValueError(
                "frontier_action_pruning_policy must be like 'frontier_touch_8' "
                "or 'extended_touch_12'"
            )
        return parts[0], max(1, int(parts[2]))

    def _prune_frontier_swap_actions(self, env: LightweightEnv, actions: list[int]) -> list[int]:
        spec = self._frontier_action_pruning_spec()
        if spec is None:
            return actions
        source, max_scored_actions = spec
        if len(actions) <= max_scored_actions:
            return actions
        touched = self._frontier_touched_physical_qubits(env, source)
        ranked: list[tuple[int, float, int]] = []
        for action in actions:
            s1, s2 = env.swap_edges[action]
            touches_frontier = int(s1 in touched or s2 in touched)
            ranked.append(
                (
                    -touches_frontier,
                    self._best_frontier_distance_delta(env, int(action), source),
                    int(action),
                )
            )
        ranked.sort()
        return [action for _touch, _delta, action in ranked[:max_scored_actions]]

    def _frontier_touched_physical_qubits(self, env: LightweightEnv, source: str) -> set[int]:
        physical: set[int] = set()
        for q0, q1 in self._frontier_logical_pairs(env, source):
            physical.add(int(env._mapping.get(q0, q0)))
            physical.add(int(env._mapping.get(q1, q1)))
        return physical

    def _frontier_logical_pairs(self, env: LightweightEnv, source: str) -> list[tuple[int, int]]:
        gate_ids = env.two_qubit_front if source == "frontier" else env.extended_two_qubit_front
        pairs: list[tuple[int, int]] = []
        for gid in gate_ids:
            q0, q1 = env.gate_qubits[gid]
            pairs.append((int(q0), int(q1)))
        return pairs

    def _best_frontier_distance_delta(self, env: LightweightEnv, action: int, source: str) -> float:
        pairs = self._frontier_logical_pairs(env, source)
        if not pairs:
            return 0.0
        s1, s2 = env.swap_edges[action]
        best_delta = float("inf")
        for q0, q1 in pairs:
            p0 = int(env._mapping.get(q0, q0))
            p1 = int(env._mapping.get(q1, q1))
            if p0 >= env.n_physical or p1 >= env.n_physical:
                continue
            before = float(env._dist_matrix[p0][p1])
            next_p0 = s2 if p0 == s1 else (s1 if p0 == s2 else p0)
            next_p1 = s2 if p1 == s1 else (s1 if p1 == s2 else p1)
            best_delta = min(best_delta, float(env._dist_matrix[next_p0][next_p1]) - before)
        return 0.0 if best_delta == float("inf") else best_delta

    def _score_frontier_action(
        self,
        env: LightweightEnv,
        action: int,
        *,
        seen_states: set[tuple[tuple[tuple[int, int], ...], tuple[bool, ...]]],
        recent_actions: deque[int],
    ) -> tuple[float, ...]:
        probe = env.clone()
        before_gates = int(probe._total_gates_executed)
        before_swaps = int(probe._total_swaps)
        _, _, _terminated, _truncated, _ = probe.step(int(action))
        self._advance_frontier_productive_passes(probe, [])
        gates_delta = int(probe._total_gates_executed) - before_gates
        return (
            float(_frontier_state_key(probe) in seen_states),
            float(gates_delta <= 0),
            float(action in recent_actions),
            float(-gates_delta),
            float(probe._compute_front_distance()),
            float(probe._compute_extended_distance()),
            float(int(probe._total_swaps) - before_swaps),
            float(action),
        )

    def _is_better_frontier_result(
        self,
        candidate: FrontierRescueResult,
        current: FrontierRescueResult | None,
    ) -> bool:
        if current is None:
            return True
        if candidate.completed != current.completed:
            return candidate.completed
        if candidate.completed:
            if candidate.total_swaps != current.total_swaps:
                return candidate.total_swaps < current.total_swaps
            return candidate.trace_len < current.trace_len
        if candidate.executed_gates != current.executed_gates:
            return candidate.executed_gates > current.executed_gates
        if candidate.total_swaps != current.total_swaps:
            return candidate.total_swaps < current.total_swaps
        return candidate.trace_len < current.trace_len

    def _result_from_trace(
        self,
        *,
        algorithm: str,
        started: float,
        circuit: QuantumCircuit,
        initial_mapping: dict[int, int],
        final_trace: tuple[int, ...],
        message: str,
        components: dict[str, bool | int | float | None | dict[str, float | int | bool]],
    ) -> NPQRRuntimeResult:
        replay = replay_action_trace(
            circuit,
            self.coupling_map,
            initial_mapping,
            list(final_trace),
            max_steps=max(self.config.max_steps, self.config.frontier_max_steps, len(final_trace) + 1),
        )
        compiled_circuit = build_routed_circuit_from_replay(circuit, self.coupling_map, replay)
        topology_check = verify_routed_circuit_topology(compiled_circuit, self.coupling_map)
        if not topology_check.passed:
            return NPQRRuntimeResult(
                algorithm=algorithm,
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
                components=components,
                initial_mapping=initial_mapping,
                final_mapping=replay.final_mapping,
                action_trace=final_trace,
            )
        return NPQRRuntimeResult(
            algorithm=algorithm,
            status="OK" if replay.completed else "INCOMPLETE",
            completed=bool(replay.completed),
            total_swaps=int(replay.total_swaps),
            depth=compiled_circuit.depth(),
            executed_gates=int(replay.executed_gates),
            trace_len=len(final_trace),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model_path=str(self.model_path),
            message=message if replay.completed else "NPQR frontier rescue stopped incomplete.",
            compiled_circuit=compiled_circuit,
            compiled_qasm=None,
            replay=replay,
            components=components,
            initial_mapping=initial_mapping,
            final_mapping=replay.final_mapping,
            action_trace=final_trace,
        )

    def _components(
        self,
        selected_top_k: int,
        rescue_used: bool,
        suffix_attempted: bool,
        suffix: SuffixRepairResult,
        *,
        frontier_attempted: bool = False,
        frontier_precheck_triggered: bool = False,
        frontier_completed: bool = False,
        frontier_candidate_count: int = 0,
        frontier_precheck: dict[str, float | int | bool] | None = None,
    ) -> dict[str, bool | int | float | None | dict[str, float | int | bool]]:
        return {
            "neural_beam": True,
            "mapping_selector": True,
            "suffix_repair": bool(suffix.found),
            "suffix_repair_attempted": bool(suffix_attempted),
            "suffix_nodes": int(suffix.nodes),
            "selector_rescue_used": bool(rescue_used),
            "selected_selector_top_k": int(selected_top_k),
            "frontier_rescue": bool(frontier_completed),
            "frontier_rescue_attempted": bool(frontier_attempted),
            "frontier_precheck_triggered": bool(frontier_precheck_triggered),
            "frontier_candidate_count": int(frontier_candidate_count),
            "frontier_action_limit": int(self.config.frontier_action_limit),
            "frontier_action_pruning": bool(self.config.frontier_action_pruning_policy),
            "frontier_action_pruning_policy": self.config.frontier_action_pruning_policy,
            "frontier_precheck": frontier_precheck,
            "sabre_fallback": False,
        }
