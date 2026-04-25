"""V15 smoke tests — verify imports and basic forward pass on tiny inputs.

These are CPU-only, run in seconds, and assert architectural invariants
without any actual training. Good for CI / pre-commit.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from src.benchmarks.circuits import generate_qft
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.light_env import LightweightEnv
from src.compiler.v15.network import NetworkConfig, PolicyValueNet
from src.compiler.v15.replay import ReplayBuffer, Sample
from src.compiler.v15.selfplay import SelfPlayConfig, play_one_episode
from src.compiler.v15.tree import MCTSConfig, run_mcts, visits_to_policy


@pytest.fixture
def tiny_env() -> QuantumRoutingEnv:
    cm = get_topology("ibm_tokyo")
    env = QuantumRoutingEnv(coupling_map=cm, max_steps=200, soft_mask=False)
    env.set_circuit(generate_qft(3), topology_name="ibm_tokyo")
    env.set_curriculum_stage(0)
    env.reset()
    return env


@pytest.fixture
def tiny_net(tiny_env: QuantumRoutingEnv) -> PolicyValueNet:
    cfg = NetworkConfig(
        gnn_in_channels=9,
        gnn_hidden=32,
        gnn_out=32,
        graph_pool_dim=64,  # 2 * gnn_out
        policy_head_hidden=64,
        value_head_hidden=32,
        n_actions=int(tiny_env.n_actions),
        dropout=0.0,
    )
    return PolicyValueNet(cfg)


@pytest.mark.unit
def test_yaml_loads_and_has_required_keys() -> None:
    with open("configs/v15_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    for top in ("topology", "network", "mcts", "selfplay", "replay", "training", "environment"):
        assert top in cfg, f"missing top-level section: {top}"
    assert cfg["algorithm"].startswith("AlphaZero")


@pytest.mark.unit
def test_network_forward_shapes(tiny_env: QuantumRoutingEnv, tiny_net: PolicyValueNet) -> None:
    info = tiny_env._get_info()
    graph = info["gnn_input"]["graph"]
    x, ei = graph.x, graph.edge_index
    logits, value = tiny_net(x, ei, batch=None)
    assert logits.shape == (1, tiny_env.n_actions)
    assert value.shape == (1,)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


@pytest.mark.unit
def test_network_predict_masks_invalid_actions(
    tiny_env: QuantumRoutingEnv, tiny_net: PolicyValueNet
) -> None:
    info = tiny_env._get_info()
    graph = info["gnn_input"]["graph"]
    mask = tiny_env.get_action_mask()
    probs, value = tiny_net.predict(
        graph.x, graph.edge_index, torch.tensor(mask, dtype=torch.float32)
    )
    invalid_idx = np.where(mask == 0)[0]
    if len(invalid_idx) > 0:
        assert probs[invalid_idx].sum().item() < 1e-6
    assert abs(probs.sum().item() - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0


@pytest.mark.unit
def test_lightweight_env_clone_preserves_state(tiny_env: QuantumRoutingEnv) -> None:
    le = LightweightEnv(tiny_env)
    clone = le.clone()
    # Mutate clone, original must stay
    clone.step(0)
    assert clone._step_count == 1
    assert le._step_count == 0


@pytest.mark.unit
def test_mcts_runs_with_low_simulations(
    tiny_env: QuantumRoutingEnv, tiny_net: PolicyValueNet
) -> None:
    le = LightweightEnv(tiny_env)
    cfg = MCTSConfig(n_simulations=4, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)
    visits, root = run_mcts(le, tiny_net, cfg)
    assert visits.shape == (tiny_env.n_actions,)
    assert visits.sum() > 0
    assert root.N > 0


@pytest.mark.unit
def test_visits_to_policy_temperatures() -> None:
    visits = np.array([1.0, 5.0, 2.0])
    p_greedy = visits_to_policy(visits, temperature=0.0)
    assert p_greedy.argmax() == 1
    assert abs(p_greedy.sum() - 1.0) < 1e-6
    p_soft = visits_to_policy(visits, temperature=1.0)
    assert abs(p_soft.sum() - 1.0) < 1e-6
    # softer than one-hot
    assert p_soft[0] > 0


@pytest.mark.unit
def test_replay_buffer_basic() -> None:
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(
            Sample(
                x=np.zeros((20, 9), dtype=np.float32),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                policy=np.ones(40, dtype=np.float32) / 40,
                value=float(i) * 0.1,
            )
        )
    assert len(buf) == 5
    xs, edges, policies, values = buf.sample_batch(3)
    assert len(xs) == 3
    assert policies.shape == (3, 40)
    assert values.shape == (3,)


@pytest.mark.integration
def test_full_selfplay_one_episode(
    tiny_env: QuantumRoutingEnv, tiny_net: PolicyValueNet
) -> None:
    """Integration: complete one self-play episode end-to-end.

    Use very low simulations to keep test fast (< 30 s).
    """
    mcts_cfg = MCTSConfig(n_simulations=4)
    sp_cfg = SelfPlayConfig(max_steps_per_game=30, temperature_threshold=5)
    samples, info = play_one_episode(
        tiny_env, tiny_net, mcts_cfg, sp_cfg, sabre_swaps=2
    )
    assert len(samples) > 0
    assert all(s.policy.sum() <= 1.0 + 1e-4 for s in samples)
    # Outcome z must be backfilled identically across all samples
    z_set = {s.value for s in samples}
    assert len(z_set) == 1
    assert -1.0 <= info["outcome_z"] <= 1.0
