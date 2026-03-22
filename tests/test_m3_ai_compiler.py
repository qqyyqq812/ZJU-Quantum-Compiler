"""M3 测试：AI 编译器核心模块"""

import pytest
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork, PPOTrainer, RolloutBuffer
from src.compiler.gnn_encoder import GATEncoder, CircuitEncoder
from src.compiler.pass_manager import AIRouter, compile_with_ai


class TestGNNEncoder:
    def test_gat_encoder_forward(self):
        encoder = GATEncoder(in_channels=8, hidden_channels=16, out_channels=8)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = encoder(x, edge_index)
        assert out.shape == (5, 8)  # 无 batch 时返回节点嵌入

    def test_gat_encoder_with_pooling(self):
        encoder = GATEncoder(in_channels=8, hidden_channels=16, out_channels=8)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        batch = torch.zeros(5, dtype=torch.long)
        out = encoder(x, edge_index, batch)
        assert out.shape == (1, 8)  # 池化后 1 个图嵌入

    def test_circuit_encoder(self):
        from torch_geometric.data import Data
        encoder = CircuitEncoder(dag_node_features=8, coupling_node_features=4, embed_dim=32)
        dag_data = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.zeros(5, dtype=torch.long),
        )
        coupling_data = Data(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.zeros(4, dtype=torch.long),
        )
        out = encoder(dag_data, coupling_data)
        assert out.shape == (1, 32)


class TestRLEnv:
    def _make_env(self):
        cm = CouplingMap.from_line(5)
        return QuantumRoutingEnv(coupling_map=cm)

    def _make_circuit(self):
        qc = QuantumCircuit(5)
        qc.cx(0, 4)  # 不相邻，需要 SWAP
        qc.cx(1, 3)
        return qc

    def test_env_creation(self):
        env = self._make_env()
        assert env.n_actions > 0
        assert env.observation_space.shape[0] > 0

    def test_env_reset(self):
        env = self._make_env()
        env.set_circuit(self._make_circuit())
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert 'total_swaps' in info

    def test_env_step(self):
        env = self._make_env()
        env.set_circuit(self._make_circuit())
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert isinstance(reward, float)

    def test_env_episode(self):
        env = self._make_env()
        qc = QuantumCircuit(3)
        qc.cx(0, 1)  # 相邻，可能不需要 SWAP
        env.set_circuit(qc)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1


class TestPolicy:
    def test_policy_creation(self):
        policy = PolicyNetwork(obs_dim=50, n_actions=10)
        assert policy is not None

    def test_policy_get_action(self):
        policy = PolicyNetwork(obs_dim=50, n_actions=10)
        obs = np.random.randn(50).astype(np.float32)
        action, log_prob, value = policy.get_action(obs)
        assert 0 <= action < 10
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_rollout_buffer(self):
        buffer = RolloutBuffer.create()
        for _ in range(10):
            buffer.add(
                np.zeros(5), 0, 1.0, -0.5, 0.8, False
            )
        assert len(buffer) == 10
        advantages, returns = buffer.compute_returns()
        assert len(advantages) == 10

    def test_ppo_update(self):
        policy = PolicyNetwork(obs_dim=20, n_actions=4)
        trainer = PPOTrainer(policy, epochs_per_update=2)
        buffer = RolloutBuffer.create()
        for _ in range(16):
            obs = np.random.randn(20).astype(np.float32)
            action, log_prob, value = policy.get_action(obs)
            buffer.add(obs, action, np.random.randn(), log_prob, value, False)
        metrics = trainer.update(buffer)
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics


class TestAIRouter:
    def test_router_creation(self):
        cm = CouplingMap.from_line(5)
        router = AIRouter(cm)
        assert not router._has_model

    def test_router_route(self):
        cm = CouplingMap.from_line(5)
        router = AIRouter(cm)
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        compiled, info = router.route(qc)
        assert isinstance(compiled, QuantumCircuit)
        assert 'total_swaps' in info

    def test_compile_with_ai(self):
        cm = CouplingMap.from_line(5)
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        compiled = compile_with_ai(qc, cm)
        assert isinstance(compiled, QuantumCircuit)
