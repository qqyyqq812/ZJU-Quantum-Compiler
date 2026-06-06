"""NPQR v1: neural-planning quantum router."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.env import QuantumRoutingEnv
from src.compiler.gnn_extractor import GraphBatch
from src.compiler.initial_mapping import (
    compute_initial_mapping,
    compute_qap_initial_mapping,
    improve_mapping_qap_local_search,
    selector_mapping_score,
)
from src.compiler.light_env import LightweightEnv
from src.compiler.pass_manager import _load_policy_state_dict
from src.compiler.policy import PolicyNetwork


@dataclass(frozen=True)
class NeuralPlannerConfig:
    beam_width: int = 4
    branch_factor: int = 4
    max_steps: int = 2000
    perturbation_count: int = 2
    soft_mask: bool = True
    policy_weight: float = 1.0
    value_weight: float = 1.0
    progress_weight: float = 0.05
    swap_penalty: float = 0.05
    step_penalty: float = 0.01
    completion_bonus: float = 1000.0
    execute_gates_first: bool = True
    drop_unproductive_pass: bool = True
    mapping_strategy: str = "legacy"
    qap_local_search_rounds: int = 2
    selector_top_k: int = 1


@dataclass(frozen=True)
class NeuralPlanResult:
    algorithm: str
    status: str
    completed: bool
    total_swaps: int
    executed_gates: int
    trace_len: int
    steps: int
    initial_mapping: dict[int, int]
    final_mapping: dict[int, int]
    has_model: bool
    elapsed_ms: float
    message: str
    action_trace: tuple[int, ...] = ()


@dataclass(frozen=True)
class _BeamState:
    env: LightweightEnv
    score: float
    log_policy: float
    value: float
    trace: tuple[int, ...]
    initial_mapping: dict[int, int]


def _mapping_key(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def _swap_physical_positions(mapping: dict[int, int], p1: int, p2: int) -> dict[int, int]:
    out = dict(mapping)
    for logical, physical in mapping.items():
        if physical == p1:
            out[logical] = p2
        elif physical == p2:
            out[logical] = p1
    return out


def _append_unique(
    mappings: list[dict[int, int]],
    seen: set[tuple[tuple[int, int], ...]],
    mapping: dict[int, int],
    n_logical: int,
    n_physical: int,
) -> None:
    if set(mapping) != set(range(n_logical)):
        return
    values = list(mapping.values())
    if len(values) != len(set(values)):
        return
    if any(value < 0 or value >= n_physical for value in values):
        return
    key = _mapping_key(mapping)
    if key not in seen:
        mappings.append(dict(mapping))
        seen.add(key)


def generate_initial_mapping_candidates(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    perturbation_count: int = 2,
    strategy: str = "legacy",
    qap_local_search_rounds: int = 2,
    selector_top_k: int = 1,
) -> list[dict[int, int]]:
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()
    identity = {i: i for i in range(n_logical)}
    mappings: list[dict[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    _append_unique(mappings, seen, identity, n_logical, n_physical)

    try:
        greedy = compute_initial_mapping(circuit, coupling_map)
        _append_unique(mappings, seen, greedy, n_logical, n_physical)
    except Exception:
        greedy = identity

    if strategy in {"qap", "selector"}:
        try:
            qap = compute_qap_initial_mapping(
                circuit,
                coupling_map,
                local_search_rounds=qap_local_search_rounds,
            )
            _append_unique(mappings, seen, qap, n_logical, n_physical)
        except Exception:
            qap = greedy
    else:
        qap = greedy

    edges = sorted({tuple(sorted(edge)) for edge in coupling_map.get_edges()})
    for base in [identity, greedy, qap]:
        for p1, p2 in edges:
            if len(mappings) >= 2 + perturbation_count:
                if strategy == "selector":
                    return _select_top_mapping_candidates(circuit, coupling_map, mappings, selector_top_k)
                return mappings
            candidate = _swap_physical_positions(base, p1, p2)
            if strategy in {"qap", "selector"}:
                candidate = improve_mapping_qap_local_search(
                    circuit,
                    coupling_map,
                    candidate,
                    rounds=qap_local_search_rounds,
                )
            _append_unique(mappings, seen, candidate, n_logical, n_physical)
    if strategy == "selector":
        return _select_top_mapping_candidates(circuit, coupling_map, mappings, selector_top_k)
    return mappings


def _select_top_mapping_candidates(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    mappings: list[dict[int, int]],
    top_k: int = 1,
) -> list[dict[int, int]]:
    if not mappings:
        return []
    limit = max(1, int(top_k))
    ranked = sorted(
        enumerate(mappings),
        key=lambda item: (selector_mapping_score(circuit, coupling_map, item[1]), item[0]),
    )
    return [dict(mapping) for _, mapping in ranked[:limit]]


class NeuralPlanningRouter:
    def __init__(
        self,
        coupling_map: CouplingMap,
        model_path: str | None = None,
        config: NeuralPlannerConfig | None = None,
    ) -> None:
        self.coupling_map = coupling_map
        self.config = config or NeuralPlannerConfig()
        self.model_path = model_path
        self.model_load_error: str | None = None
        template_env = QuantumRoutingEnv(
            coupling_map=coupling_map,
            max_steps=self.config.max_steps,
            soft_mask=self.config.soft_mask,
            initial_mapping_fn=None,
            use_sabre_reward=False,
        )
        self._obs_dim = int(template_env.observation_space.shape[0])
        self._max_front_gates = int(template_env._max_front_gates)
        self._max_swap_edges = int(template_env._max_swap_edges)
        self.swap_edges = list(template_env.swap_edges)
        self.pass_action = int(template_env.PASS_ACTION)
        self.policy = PolicyNetwork(obs_dim=self._obs_dim, n_actions=template_env.action_space.n)
        self._has_model = False
        if model_path is None:
            self.model_load_error = "No AI model path was provided."
            return
        if not Path(model_path).exists():
            self.model_load_error = f"AI model file is missing: {model_path}"
            return
        try:
            state_dict = _load_policy_state_dict(model_path)
            self.policy.load_state_dict(state_dict)
            self.policy.eval()
            self._has_model = True
        except (RuntimeError, ValueError, TypeError) as exc:
            self.model_load_error = str(exc)

    def route_count_only(self, circuit: QuantumCircuit) -> NeuralPlanResult:
        started = time.perf_counter()
        if not self._has_model:
            return NeuralPlanResult(
                algorithm="npqr_neural_beam",
                status="N/A",
                completed=False,
                total_swaps=-1,
                executed_gates=0,
                trace_len=0,
                steps=0,
                initial_mapping={},
                final_mapping={},
                has_model=False,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                message=self.model_load_error or "AI model is not loadable.",
            )

        candidates = generate_initial_mapping_candidates(
            circuit,
            self.coupling_map,
            perturbation_count=self.config.perturbation_count,
            strategy=self.config.mapping_strategy,
            qap_local_search_rounds=self.config.qap_local_search_rounds,
            selector_top_k=self.config.selector_top_k,
        )
        best: _BeamState | None = None
        for mapping in candidates:
            state = self._run_beam_for_mapping(circuit, mapping)
            if self._is_better_state(state, best):
                best = state
            if best.env.is_done() and best.env._total_swaps == 0:
                break

        if best is None:
            return NeuralPlanResult(
                algorithm="npqr_neural_beam",
                status="INCOMPLETE",
                completed=False,
                total_swaps=-1,
                executed_gates=0,
                trace_len=0,
                steps=0,
                initial_mapping={},
                final_mapping={},
                has_model=True,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                message="NPQR did not produce any beam state.",
            )

        completed = bool(best.env.is_done())
        return NeuralPlanResult(
            algorithm="npqr_neural_beam",
            status="OK" if completed else "INCOMPLETE",
            completed=completed,
            total_swaps=int(best.env._total_swaps),
            executed_gates=int(best.env._total_gates_executed),
            trace_len=len(best.trace),
            steps=int(best.env._step_count),
            initial_mapping=best.initial_mapping,
            final_mapping=dict(best.env._mapping),
            has_model=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            message=(
                "NPQR-NeuralBeam completed the route."
                if completed
                else "NPQR-NeuralBeam stopped with an honest incomplete route."
            ),
            action_trace=best.trace,
        )

    def _run_beam_for_mapping(self, circuit: QuantumCircuit, initial_mapping: dict[int, int]) -> _BeamState:
        base_env = self._build_base_env(circuit, initial_mapping)
        root_env = LightweightEnv(base_env)
        root_trace: tuple[int, ...] = ()
        if self.config.execute_gates_first:
            root_trace = self._advance_productive_passes(root_env, root_trace)
        root_value = 1.0 if root_env.is_done() else self._estimate_value(root_env)
        root = _BeamState(
            env=root_env,
            score=self._score_state(root_env, 0.0, root_value),
            log_policy=0.0,
            value=root_value,
            trace=root_trace,
            initial_mapping=dict(initial_mapping),
        )
        beam = [root]
        best = root
        for _ in range(self.config.max_steps):
            expanded: list[_BeamState] = []
            for state in beam:
                if state.env.is_done():
                    if self._is_better_state(state, best):
                        best = state
                    expanded.append(state)
                    continue
                for action, log_prob in self._top_policy_actions(state.env):
                    child_env = state.env.clone()
                    _, _, terminated, truncated, _ = child_env.step(action)
                    child_trace = (*state.trace, action)
                    if self.config.execute_gates_first and not terminated and not truncated:
                        child_trace = self._advance_productive_passes(child_env, child_trace)
                        terminated = child_env.is_done()
                        truncated = child_env._step_count >= child_env.max_steps
                    child_value = 1.0 if child_env.is_done() else self._estimate_value(child_env)
                    if truncated and not child_env.is_done():
                        child_value = -1.0
                    child = _BeamState(
                        env=child_env,
                        score=self._score_state(child_env, state.log_policy + log_prob, child_value),
                        log_policy=state.log_policy + log_prob,
                        value=child_value,
                        trace=child_trace,
                        initial_mapping=state.initial_mapping,
                    )
                    expanded.append(child)
                    if self._is_better_state(child, best):
                        best = child
            if not expanded:
                break
            expanded.sort(key=lambda item: item.score, reverse=True)
            beam = expanded[: self.config.beam_width]
            if all(state.env.is_done() for state in beam):
                break
        return best

    def _advance_productive_passes(self, env: LightweightEnv, trace: tuple[int, ...]) -> tuple[int, ...]:
        while not env.is_done() and env._step_count < env.max_steps and self._pass_would_progress(env):
            _, _, terminated, truncated, _ = env.step(env.PASS_ACTION)
            trace = (*trace, env.PASS_ACTION)
            if terminated or truncated:
                break
        return trace

    def _pass_would_progress(self, env: LightweightEnv) -> bool:
        if env.is_done() or env._step_count >= env.max_steps:
            return False
        probe = env.clone()
        before = int(probe._total_gates_executed)
        probe.step(probe.PASS_ACTION)
        return int(probe._total_gates_executed) > before

    def _build_base_env(self, circuit: QuantumCircuit, initial_mapping: dict[int, int]) -> QuantumRoutingEnv:
        def mapping_fn(_circuit, _coupling_map):
            return dict(initial_mapping)

        env = QuantumRoutingEnv(
            coupling_map=self.coupling_map,
            max_steps=self.config.max_steps,
            soft_mask=self.config.soft_mask,
            initial_mapping_fn=mapping_fn,
            use_sabre_reward=False,
        )
        env.set_circuit(circuit)
        env.reset()
        return env

    def _top_policy_actions(self, env: LightweightEnv) -> list[tuple[int, float]]:
        device = next(self.policy.parameters()).device
        obs = torch.tensor(self._fixed_obs(env), dtype=torch.float32, device=device).unsqueeze(0)
        info = env._get_info()
        gnn_input = info["gnn_input"]
        graph_batch = GraphBatch.from_data_list([gnn_input["graph"]]).to(device)
        action_mask = torch.tensor(env.get_action_mask(), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, _ = self.policy.forward(obs, graph_batch, [gnn_input["swap_edges"]])
            logits = dist.logits.squeeze(0)
            logits = logits.masked_fill(action_mask == 0, -1e9)
            log_probs = torch.log_softmax(logits, dim=-1)
            valid = torch.where(action_mask > 0)[0]
            if valid.numel() == 0:
                return []
            if self.config.drop_unproductive_pass and not self._pass_would_progress(env):
                swap_valid = valid[valid < env.n_swap_actions]
                if swap_valid.numel() > 0:
                    valid = swap_valid
            k = min(self.config.branch_factor, int(valid.numel()))
            top = torch.topk(log_probs[valid], k=k)
        return [(int(valid[idx].item()), float(score.item())) for score, idx in zip(top.values.cpu(), top.indices.cpu(), strict=True)]

    def _estimate_value(self, env: LightweightEnv) -> float:
        device = next(self.policy.parameters()).device
        obs = torch.tensor(self._fixed_obs(env), dtype=torch.float32, device=device).unsqueeze(0)
        info = env._get_info()
        gnn_input = info["gnn_input"]
        graph_batch = GraphBatch.from_data_list([gnn_input["graph"]]).to(device)
        with torch.no_grad():
            _, values = self.policy.forward(obs, graph_batch, [gnn_input["swap_edges"]])
        return float(values.squeeze(0).item())

    def _fixed_obs(self, env: LightweightEnv) -> np.ndarray:
        distances = np.zeros(self._max_front_gates, dtype=np.float32)
        front_pairs: list[tuple[int, int]] = []
        for i, gid in enumerate(env.two_qubit_front[: self._max_front_gates]):
            q0, q1 = env.gate_qubits[gid]
            p0 = env._mapping.get(q0, q0)
            p1 = env._mapping.get(q1, q1)
            if p0 < env.n_physical and p1 < env.n_physical:
                distances[i] = env._dist_matrix[p0][p1]
                front_pairs.append((p0, p1))
        ext_distances = np.zeros(self._max_front_gates, dtype=np.float32)
        front_ids = set(env.two_qubit_front)
        j = 0
        for gid in env.extended_two_qubit_front:
            if gid in front_ids:
                continue
            if j >= self._max_front_gates:
                break
            q0, q1 = env.gate_qubits[gid]
            p0 = env._mapping.get(q0, q0)
            p1 = env._mapping.get(q1, q1)
            if p0 < env.n_physical and p1 < env.n_physical:
                ext_distances[j] = env._dist_matrix[p0][p1]
            j += 1
        edge_features = np.zeros(self._max_swap_edges * 4, dtype=np.float32)
        for idx, (s1, s2) in enumerate(env.swap_edges[: self._max_swap_edges]):
            if not front_pairs:
                continue
            deltas = []
            for p0, p1 in front_pairs:
                d_now = env._dist_matrix[p0][p1]
                new_p0 = s2 if p0 == s1 else (s1 if p0 == s2 else p0)
                new_p1 = s2 if p1 == s1 else (s1 if p1 == s2 else p1)
                d_after = env._dist_matrix[new_p0][new_p1]
                deltas.append(d_after - d_now)
            base = idx * 4
            edge_features[base] = min(deltas)
            edge_features[base + 1] = max(deltas)
            edge_features[base + 2] = sum(deltas) / len(deltas)
            edge_features[base + 3] = sum(1 for delta in deltas if delta < 0)
        stats = np.zeros(10, dtype=np.float32)
        valid_dists = distances[distances > 0]
        stats[0] = len(front_pairs)
        stats[1] = float(np.sum(valid_dists)) if len(valid_dists) > 0 else 0.0
        stats[2] = float(np.mean(valid_dists)) if len(valid_dists) > 0 else 0.0
        stats[3] = float(np.max(valid_dists)) if len(valid_dists) > 0 else 0.0
        stats[4] = float(np.min(valid_dists)) if len(valid_dists) > 0 else 0.0
        stats[5] = len(env.swap_edges)
        stats[6] = env.n_physical
        stats[7] = env._total_swaps
        stats[8] = env._step_count
        stats[9] = int(((~env.executed) & env.gate_is_two_qubit).sum())
        progress = np.array([int((~env.executed).sum()) / max(env._total_gates, 1)], dtype=np.float32)
        obs = np.concatenate([edge_features, distances, ext_distances, stats, progress])
        if obs.shape[0] < self._obs_dim:
            obs = np.pad(obs, (0, self._obs_dim - obs.shape[0]))
        elif obs.shape[0] > self._obs_dim:
            obs = obs[: self._obs_dim]
        return obs

    def _score_state(self, env: LightweightEnv, log_policy: float, value: float) -> float:
        score = (
            self.config.policy_weight * log_policy
            + self.config.value_weight * value
            + self.config.progress_weight * float(env._total_gates_executed)
            - self.config.swap_penalty * float(env._total_swaps)
            - self.config.step_penalty * float(env._step_count)
        )
        if env.is_done():
            score += self.config.completion_bonus
        return float(score)

    def _is_better_state(self, candidate: _BeamState, current: _BeamState | None) -> bool:
        if current is None:
            return True
        c_done = candidate.env.is_done()
        b_done = current.env.is_done()
        if c_done != b_done:
            return c_done
        if c_done and b_done:
            if candidate.env._total_swaps != current.env._total_swaps:
                return candidate.env._total_swaps < current.env._total_swaps
            if candidate.env._step_count != current.env._step_count:
                return candidate.env._step_count < current.env._step_count
            return candidate.score > current.score
        if candidate.env._total_gates_executed != current.env._total_gates_executed:
            return candidate.env._total_gates_executed > current.env._total_gates_executed
        if candidate.env._total_swaps != current.env._total_swaps:
            return candidate.env._total_swaps < current.env._total_swaps
        if candidate.env._step_count != current.env._step_count:
            return candidate.env._step_count < current.env._step_count
        return candidate.score > current.score
