"""V15 self-play — generate (state, π_MCTS, z) tuples for one circuit.

Outline:
    play_one_episode(circuit, network, mcts_cfg, ...) →
        for step in range(max_steps):
            run_mcts(env, network) → (visit_counts, root)
            π = visits_to_policy(visit_counts, temperature)
            record (state, π, env_state)
            sample action from π (or argmax if T=0)
            env.step(action)
            if done: break
        z = compute_outcome(total_swaps, sabre_baseline)
        backfill all records with z
        return list[Sample]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.compiler.env import QuantumRoutingEnv
from src.compiler.light_env import LightweightEnv
from src.compiler.v15.replay import Sample
from src.compiler.v15.tree import MCTSConfig, run_mcts, visits_to_policy

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    max_steps_per_game: int = 800
    temperature_warmup: float = 1.0
    temperature_play: float = 0.0
    temperature_threshold: int = 30
    reward_scheme: str = "relative_sabre"  # "relative_sabre" | "absolute"


def _compute_outcome(
    total_swaps: int,
    sabre_swaps: int,
    completed: bool,
    scheme: str,
    max_steps: int,
) -> float:
    """Map final game state to a scalar in [-1, 1] for value head training."""
    if not completed:
        return -1.0  # Truncated → strong negative signal

    if scheme == "relative_sabre":
        if sabre_swaps <= 0:
            return 1.0  # SABRE itself produced 0 SWAP — anything we did with 0 also wins
        # +1 means we achieved 0 SWAP; 0 means equal to SABRE; -1 means 2× SABRE
        relative = (sabre_swaps - total_swaps) / sabre_swaps
        return float(np.clip(relative, -1.0, 1.0))

    # absolute scheme
    return float(np.clip(1.0 - (total_swaps / max(max_steps, 1)) * 2, -1.0, 1.0))


def play_one_episode(
    base_env: QuantumRoutingEnv,
    network,  # PolicyValueNet
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    sabre_swaps: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[Sample], dict]:
    """Play one self-play game on `base_env`.

    Returns:
        samples: list[Sample] one per environment step
        info: dict with 'total_swaps', 'completed', 'steps', 'outcome_z'
    """
    if rng is None:
        rng = np.random.default_rng()
    env = LightweightEnv(base_env)

    pending: list[Sample] = []
    step = 0

    for step in range(sp_cfg.max_steps_per_game):
        if env.is_done():
            break

        visit_counts, root = run_mcts(env, network, mcts_cfg, rng=rng)
        T = (
            sp_cfg.temperature_warmup
            if step < sp_cfg.temperature_threshold
            else sp_cfg.temperature_play
        )
        policy = visits_to_policy(visit_counts, T)

        # Capture state at decision time (for training)
        info = env._get_info()  # type: ignore[attr-defined]
        graph = info["gnn_input"]["graph"]
        x = graph["x"] if isinstance(graph, dict) else graph.x
        edge_index = (
            graph["edge_index"] if isinstance(graph, dict) else graph.edge_index
        )
        x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
        ei_np = (
            edge_index
            if isinstance(edge_index, np.ndarray)
            else edge_index.detach().cpu().numpy()
        )

        pending.append(
            Sample(
                x=x_np.copy(),
                edge_index=ei_np.copy(),
                policy=policy.copy(),
                value=0.0,  # backfilled below
            )
        )

        # Sample action
        if T < 1e-6:
            action = int(policy.argmax())
        else:
            action = int(rng.choice(len(policy), p=policy / policy.sum()))

        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    completed = bool(env.is_done())
    z = _compute_outcome(
        env._total_swaps,
        sabre_swaps,
        completed,
        sp_cfg.reward_scheme,
        sp_cfg.max_steps_per_game,
    )

    # Backfill outcome
    for s in pending:
        s.value = z

    info_out = {
        "total_swaps": int(env._total_swaps),
        "completed": completed,
        "steps": int(env._step_count),
        "outcome_z": float(z),
        "sabre_swaps": int(sabre_swaps),
    }
    return pending, info_out
