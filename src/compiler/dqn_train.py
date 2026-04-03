"""
V11 图强化学习 DQN 训练管线
=============================
基于 DQNNetwork 与 ReplayBuffer 的循环执行逻辑结构
"""
from __future__ import annotations

import torch
import argparse
import json
import time
from pathlib import Path
from collections import deque
import numpy as np

from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.curriculum import CurriculumScheduler
from src.compiler.dqn_policy import DQNNetwork, ReplayBuffer, DQNTrainer


def evaluate_dqn(q_net: DQNNetwork, cm, circuits: list, n_eval: int = 5, tabu_size: int = 4) -> dict:
    env = QuantumRoutingEnv(coupling_map=cm, soft_mask=False, tabu_size=tabu_size)
    total_swaps, total_steps, completed = 0, 0, 0

    for qc in circuits[:n_eval]:
        try:
            env.set_circuit(qc)
            obs, info = env.reset()
            done = False

            for _ in range(env.max_steps):
                mask = env.get_action_mask()
                # 测试期完全贪心， epsilon = 0.0
                action, _ = q_net.get_action(obs, action_mask=mask, gnn_input=info.get('gnn_input'), epsilon=0.0)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done:
                    break

            total_swaps += info.get('total_swaps', 0)
            total_steps += info.get('step_count', 0)
            if terminated:
                completed += 1
        except Exception:
            pass

    n = min(n_eval, len(circuits))
    return {
        'avg_swaps': total_swaps / max(n, 1),
        'avg_steps': total_steps / max(n, 1),
        'completion_rate': completed / max(n, 1),
    }


def train_dqn(
    topology_name: str = "ibm_tokyo",
    n_episodes: int = 50000,
    save_dir: str = "models",
    lr: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 10000,
    target_update_tau: float = 0.005,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 20000,
):
    cm = get_topology(topology_name)
    
    # 强制硬掩码设置
    env = QuantumRoutingEnv(
        coupling_map=cm,
        reward_gate=1.0,
        penalty_swap=-1.0,
        reward_done=20.0,
        distance_reward_coef=0.5,
        soft_mask=False,
        tabu_size=4,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  训练设备: {device} (V11 DQN)")
    
    q_net = DQNNetwork(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n).to(device)
    trainer = DQNTrainer(q_net, lr=lr, gamma=gamma, tau=target_update_tau)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    
    scheduler = CurriculumScheduler(max_n_qubits=cm.size())
    circuits = scheduler.circuits
    eval_circuits = circuits  # 可以精细化
    
    best_eval_swaps = float('inf')
    loss_window = deque(maxlen=100)
    swap_window = deque(maxlen=100)
    
    print(f"🚀 V11 DQN 启动: {topology_name} ({cm.size()}Q)")
    t0 = time.time()
    
    for episode in range(n_episodes):
        circuit = circuits[episode % len(circuits)]
        env.set_circuit(circuit)
        obs, info = env.reset()
        
        # Epsilon 衰减
        eps = epsilon_end + (epsilon_start - epsilon_end) * max(0, (epsilon_decay_episodes - episode) / epsilon_decay_episodes)
        
        ep_loss = 0.0
        updates = 0
        
        while True:
            mask = env.get_action_mask()
            action, _ = q_net.get_action(obs, action_mask=mask, gnn_input=info.get('gnn_input'), epsilon=eps)
            
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_mask = env.get_action_mask()
            
            # 使用 DQN Q-learning 特有的基于状态动作价值的密集奖励训练，因此存入 Buffer 前可以做 Reward Clipping
            buffer.add(
                obs, action, reward, next_obs, terminated or truncated, 
                mask, next_mask, info.get('gnn_input'), next_info.get('gnn_input')
            )
            
            # 更新模型
            loss = trainer.update(buffer, batch_size=batch_size)
            if loss > 0:
                ep_loss += loss
                updates += 1
                
            obs = next_obs
            info = next_info
            
            if terminated or truncated:
                break
                
        swap_window.append(info.get('total_swaps', 0))
        if updates > 0:
            loss_window.append(ep_loss / updates)
            
        # 课程汇报
        if scheduler.report_episode(info.get('total_swaps', 0)):
             circuits = scheduler.circuits
             print(f"\n  🎓 课程升级! → 阶段 {scheduler.current_stage}")
             
        if (episode + 1) % 100 == 0:
            avg_swap = np.mean(swap_window) if swap_window else 0
            avg_loss = np.mean(loss_window) if loss_window else 0
            ep_s = (episode + 1) / (time.time() - t0)
            print(f"  [{episode+1:>6}/{n_episodes}] SWAP={avg_swap:>5.1f} Loss={avg_loss:.3f} Eps={eps:.3f} ({ep_s:.1f} ep/s)")
            
        if (episode + 1) % 1000 == 0:
            res = evaluate_dqn(q_net, cm, eval_circuits, tabu_size=4)
            print(f"  📊 EVAL: avg_swap={res['avg_swaps']:.1f} done={res['completion_rate']:.0%}")
            if res['avg_swaps'] < best_eval_swaps and res['completion_rate'] == 1.0:
                best_eval_swaps = res['avg_swaps']
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                trainer.save(str(Path(save_dir) / f"v11_dqn_{topology_name}_best.pt"))
                
    trainer.save(str(Path(save_dir) / f"v11_dqn_{topology_name}_final.pt"))
    print("✅ 训练完成")

if __name__ == '__main__':
    train_dqn()
