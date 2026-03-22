"""
训练管线
========
端到端训练量子电路路由 RL Agent。

用法:
    python -m src.compiler.train --episodes 1000 --topology grid_3x3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit.transpiler import CouplingMap

from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork, PPOTrainer, RolloutBuffer


def make_training_circuits(n_qubits: int, n_circuits: int = 20, seed: int = 42) -> list:
    """生成训练用电路集合。"""
    circuits = []
    for i in range(n_circuits):
        circuits.append(generate_random(n_qubits, depth=n_qubits, seed=seed + i))
    # 加入结构化电路
    circuits.append(generate_qft(n_qubits))
    circuits.append(generate_qaoa(n_qubits, p=1))
    return circuits


def train(
    topology_name: str = "grid_3x3",
    n_qubits: int = 5,
    n_episodes: int = 500,
    rollout_steps: int = 128,
    log_interval: int = 50,
    save_dir: str = "models",
) -> dict:
    """训练路由 Agent。

    Args:
        topology_name: 目标拓扑名称
        n_qubits: 训练电路的比特数
        n_episodes: 训练 episode 数
        rollout_steps: 每次 rollout 的步数
        log_interval: 日志打印间隔
        save_dir: 模型保存目录
    Returns:
        训练历史
    """
    cm = get_topology(topology_name)
    env = QuantumRoutingEnv(coupling_map=cm)
    circuits = make_training_circuits(n_qubits)

    policy = PolicyNetwork(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
    )
    trainer = PPOTrainer(policy)

    history = {
        'episode_rewards': [],
        'episode_swaps': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
    }

    print(f"🚀 开始训练: {topology_name} ({cm.size()}q), 电路 {n_qubits}q, {n_episodes} episodes")
    t0 = time.time()

    for episode in range(n_episodes):
        # 随机选一个电路
        circuit = circuits[episode % len(circuits)]
        env.set_circuit(circuit)
        obs, info = env.reset()

        buffer = RolloutBuffer.create()
        episode_reward = 0.0

        # Rollout
        for _ in range(rollout_steps):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer.add(obs, action, reward, log_prob, value, terminated or truncated)
            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        # PPO 更新
        if len(buffer) > 0:
            metrics = trainer.update(buffer)
            history['policy_losses'].append(metrics['policy_loss'])
            history['value_losses'].append(metrics['value_loss'])

        history['episode_rewards'].append(episode_reward)
        history['episode_swaps'].append(info['total_swaps'])
        history['episode_lengths'].append(info['step_count'])

        # 日志
        if (episode + 1) % log_interval == 0:
            recent = slice(-log_interval, None)
            avg_reward = np.mean(history['episode_rewards'][recent])
            avg_swaps = np.mean(history['episode_swaps'][recent])
            elapsed = time.time() - t0
            print(f"  [{episode+1}/{n_episodes}] "
                  f"avg_reward={avg_reward:.1f} avg_swaps={avg_swaps:.1f} "
                  f"time={elapsed:.0f}s")

    # 保存模型
    model_path = Path(save_dir) / f"router_{topology_name}_{n_qubits}q.pt"
    trainer.save(str(model_path))
    print(f"✅ 模型已保存: {model_path}")

    # 保存训练历史
    history_path = Path(save_dir) / f"history_{topology_name}_{n_qubits}q.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ 训练历史已保存: {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="训练量子电路路由 RL Agent")
    parser.add_argument('--topology', default='grid_3x3', help='目标拓扑')
    parser.add_argument('--qubits', type=int, default=5, help='电路比特数')
    parser.add_argument('--episodes', type=int, default=500, help='训练 episode 数')
    parser.add_argument('--save-dir', default='models', help='模型保存目录')
    args = parser.parse_args()

    train(
        topology_name=args.topology,
        n_qubits=args.qubits,
        n_episodes=args.episodes,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
