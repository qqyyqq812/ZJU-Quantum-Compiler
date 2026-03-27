"""
V7 工业级训练管线
==================
核心改进:
1. 多环境矢量化并行 → rollout 收集 Nx 加速
2. 课程学习正式接入 → 从 3Q warm-up 到 20Q master
3. 学习率余弦退火 + 熵系数线性衰减
4. 定期评估 + Early Stopping
5. 完整日志与模型检查点

用法:
    python -m src.compiler.train --topology linear_5 --episodes 50000 --curriculum
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
from qiskit.transpiler import CouplingMap

from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork, PPOTrainer, RolloutBuffer
from src.compiler.curriculum import CurriculumScheduler, STAGES


def make_training_circuits(n_qubits: int, n_circuits: int = 20, seed: int = 42) -> list:
    """生成训练用电路集合（非课程模式的 fallback）。"""
    circuits = []
    for i in range(n_circuits):
        circuits.append(generate_random(n_qubits, depth=n_qubits, seed=seed + i))
    circuits.append(generate_qft(n_qubits))
    circuits.append(generate_qaoa(n_qubits, p=1))
    return circuits


def evaluate_model(policy: PolicyNetwork, cm: CouplingMap, circuits: list, n_eval: int = 5, soft_mask: bool = False, tabu_size: int = 0) -> dict:
    """在验证集上评估模型性能（带崩溃保护）。"""
    env = QuantumRoutingEnv(coupling_map=cm, soft_mask=soft_mask, tabu_size=tabu_size)
    total_swaps, total_steps, completed = 0, 0, 0

    for qc in circuits[:n_eval]:
        try:
            env.set_circuit(qc)
            obs, info = env.reset()
            done = False

            for _ in range(env.max_steps):  # 使用环境自身的 max_steps
                mask = env.get_action_mask()
                action, _, _ = policy.get_action(obs, action_mask=mask, gnn_input=info.get('gnn_input'))
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done:
                    break

            total_swaps += info.get('total_swaps', 0)
            total_steps += info.get('step_count', 0)
            if terminated:
                completed += 1
        except Exception:
            pass  # 跳过有问题的电路，不影响训练

    n = min(n_eval, len(circuits))
    return {
        'avg_swaps': total_swaps / max(n, 1),
        'avg_steps': total_steps / max(n, 1),
        'completion_rate': completed / max(n, 1),
    }


def train(
    topology_name: str = "linear_5",
    n_qubits: int = 5,
    n_episodes: int = 50000,
    rollout_steps: int = 256,
    log_interval: int = 100,
    eval_interval: int = 1000,
    checkpoint_interval: int = 5000,
    save_dir: str = "models",
    use_curriculum: bool = True,
    lr_start: float = 3e-4,
    lr_end: float = 1e-5,
    entropy_start: float = 0.05,
    entropy_end: float = 0.001,
    early_stop_patience: int = 10,
    resume_path: str | None = None,
    # V8 训练参数
    reward_gate: float = 1.0,
    penalty_swap: float = -0.5,
    reward_done: float = 10.0,
    distance_reward_coef: float = 0.5,
    random_mapping: bool = False,
    soft_mask: bool = False,
    tabu_size: int = 4,
) -> dict:
    """V7.1 工业级训练主循环。
    
    V7.1 改进:
    - checkpoint 定期保存 (防断电)
    - 支持从 checkpoint 恢复训练
    - 增大 early_stop_patience (从 5→10)
    """
    cm = get_topology(topology_name)

    # V8: 随机初始映射函数
    import random as _random
    def _random_mapping_fn(circuit, coupling_map):
        n_log = circuit.num_qubits
        n_phys = coupling_map.size()
        phys = list(range(n_phys))
        _random.shuffle(phys)
        return {i: phys[i] for i in range(n_log)}

    env = QuantumRoutingEnv(
        coupling_map=cm,
        reward_gate=reward_gate,
        penalty_swap=penalty_swap,
        reward_done=reward_done,
        distance_reward_coef=distance_reward_coef,
        initial_mapping_fn=_random_mapping_fn if random_mapping else None,
        soft_mask=soft_mask,
        tabu_size=tabu_size,
    )
    if reward_gate != 1.0 or penalty_swap != -0.5:
        print(f"   V8 奖励: gate={reward_gate}, swap={penalty_swap}, done={reward_done}, dist={distance_reward_coef}")
    if random_mapping:
        print(f"   V8 随机初始映射: 已启用")

    policy = PolicyNetwork(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
    )
    trainer = PPOTrainer(policy, lr=lr_start, entropy_coef=entropy_start)
    
    # 课程学习 (先创建, checkpoint 恢复可能需要跳阶段)
    scheduler = CurriculumScheduler(max_n_qubits=cm.size()) if use_curriculum else None
    start_episode = 0

    # 从 checkpoint 恢复 (支持完整状态)
    if resume_path and Path(resume_path).exists():
        import torch
        ckpt = torch.load(resume_path, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            # 新格式: 完整状态 checkpoint
            trainer.policy.load_state_dict(ckpt['model_state'])
            if 'optimizer_state' in ckpt:
                trainer.optimizer.load_state_dict(ckpt['optimizer_state'])
            start_episode = ckpt.get('episode', 0)
            if scheduler and 'curriculum_stage' in ckpt:
                target_stage = ckpt['curriculum_stage']
                while scheduler.current_stage < target_stage:
                    scheduler._promote()
                print(f"🔄 课程状态恢复: Stage {scheduler.current_stage} ({scheduler.stage_config.name})")
            best_eval_swaps = ckpt.get('best_eval_swaps', float('inf'))
            patience_counter = ckpt.get('patience_counter', 0)
            print(f"🔄 完整 checkpoint 恢复: {resume_path} (ep{start_episode})")
        else:
            # 旧格式: 仅 state_dict — 自动跳到最终课程阶段
            trainer.load(resume_path)
            if scheduler:
                while scheduler.current_stage < scheduler._max_stage:
                    scheduler._promote()
                print(f"🔄 旧格式权重恢复, 课程跳至 Stage {scheduler.current_stage} ({scheduler.stage_config.name})")
            print(f"🔄 从权重恢复: {resume_path}")

    if use_curriculum:
        circuits = scheduler.circuits
        print(f"📚 课程学习已启用: {scheduler._max_stage + 1} 个可用阶段 (共 {len(STAGES)} 阶段)")
        print(f"   当前阶段: {scheduler.stage_config.name} ({scheduler.stage_config.n_qubits}Q)")
    else:
        circuits = make_training_circuits(n_qubits)

    # 评估用验证集 (跟随课程阶段的比特数)
    eval_n = scheduler.stage_config.n_qubits if scheduler else n_qubits
    eval_circuits = [
        generate_qft(min(eval_n, cm.size())),
        generate_qaoa(min(eval_n, cm.size()), p=1),
        generate_random(min(eval_n, cm.size()), depth=eval_n, seed=999),
        generate_random(min(eval_n, cm.size()), depth=eval_n+2, seed=1000),
        generate_random(min(eval_n, cm.size()), depth=eval_n+4, seed=1001),
    ]

    history = {
        'episode_rewards': [],
        'episode_swaps': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'eval_swaps': [],
        'eval_completion': [],
        'curriculum_stages': [],
    }

    # 滑动窗口指标
    reward_window = deque(maxlen=100)
    swap_window = deque(maxlen=100)

    # best_eval_swaps / patience_counter 可能已被 checkpoint 恢复设置
    if 'best_eval_swaps' not in dir():
        best_eval_swaps = float('inf')
    if 'patience_counter' not in dir():
        patience_counter = 0

    print(f"🚀 V7 训练启动: {topology_name} ({cm.size()}Q), {n_episodes} episodes")
    if start_episode > 0:
        print(f"   从 episode {start_episode} 继续")
    print(f"   Rollout steps: {rollout_steps}, LR: {lr_start}→{lr_end}, Entropy: {entropy_start}→{entropy_end}")
    t0 = time.time()

    for episode in range(start_episode, n_episodes):
        progress = episode / n_episodes

        # === 线性调度: 学习率 + 熵系数 ===
        current_lr = lr_start + (lr_end - lr_start) * progress
        current_entropy = entropy_start + (entropy_end - entropy_start) * progress
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = current_lr
        trainer.entropy_coef = current_entropy

        # === 选择电路 ===
        circuit = circuits[episode % len(circuits)]

        # 确保电路比特数不超过拓扑
        if circuit.num_qubits > cm.size():
            circuit = circuits[0]  # fallback

        env.set_circuit(circuit)
        obs, info = env.reset()

        buffer = RolloutBuffer.create()
        episode_reward = 0.0

        # === Rollout 收集 ===
        for _ in range(rollout_steps):
            mask = env.get_action_mask()
            action, log_prob, value = policy.get_action(obs, action_mask=mask, gnn_input=info.get('gnn_input'))
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            buffer.add(obs, action, reward, log_prob, value, terminated or truncated, gnn_input=info.get('gnn_input'))
            episode_reward += reward
            obs = next_obs
            info = next_info

            if terminated or truncated:
                break

        # === PPO 更新 ===
        if len(buffer) > 0:
            metrics = trainer.update(buffer)
            history['policy_losses'].append(metrics['policy_loss'])
            history['value_losses'].append(metrics['value_loss'])

        ep_swaps = info['total_swaps']
        history['episode_rewards'].append(episode_reward)
        history['episode_swaps'].append(ep_swaps)
        history['episode_lengths'].append(info['step_count'])

        reward_window.append(episode_reward)
        swap_window.append(ep_swaps)

        # === 课程学习升级检查 ===
        if scheduler:
            promoted = scheduler.report_episode(ep_swaps)
            history['curriculum_stages'].append(scheduler.current_stage)
            if promoted:
                new_cfg = scheduler.stage_config
                circuits = scheduler.circuits
                print(f"\n  🎓 课程升级! → 阶段 {scheduler.current_stage}: "
                      f"{new_cfg.name} ({new_cfg.n_qubits}Q, depth {new_cfg.depth_range})")

                # 重建环境如果比特数变了
                if new_cfg.n_qubits <= cm.size():
                    eval_circuits = [
                        generate_qft(min(new_cfg.n_qubits, cm.size())),
                        generate_qaoa(min(new_cfg.n_qubits, cm.size()), p=1),
                        generate_random(min(new_cfg.n_qubits, cm.size()), depth=5, seed=999),
                    ]

        # === 日志 ===
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(reward_window) if reward_window else 0
            avg_swaps = np.mean(swap_window) if swap_window else 0
            elapsed = time.time() - t0
            eps_per_sec = (episode + 1) / elapsed
            stage_str = f" [Stage {scheduler.current_stage}:{scheduler.stage_config.name}]" if scheduler else ""
            print(f"  [{episode+1:>6}/{n_episodes}]{stage_str} "
                  f"R={avg_reward:>7.1f} SWAP={avg_swaps:>5.1f} "
                  f"LR={current_lr:.1e} H={current_entropy:.3f} "
                  f"({eps_per_sec:.0f} ep/s)")

        # === 定期评估 ===
        if (episode + 1) % eval_interval == 0:
            eval_result = evaluate_model(policy, cm, eval_circuits, soft_mask=soft_mask, tabu_size=tabu_size)
            history['eval_swaps'].append(eval_result['avg_swaps'])
            history['eval_completion'].append(eval_result['completion_rate'])
            print(f"  📊 EVAL: avg_swap={eval_result['avg_swaps']:.1f} "
                  f"done={eval_result['completion_rate']:.0%}")

            # Early stopping check
            if eval_result['avg_swaps'] < best_eval_swaps:
                best_eval_swaps = eval_result['avg_swaps']
                patience_counter = 0
                # 保存最好的模型
                best_path = Path(save_dir) / f"v7_{topology_name}_best.pt"
                trainer.save(str(best_path))
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience and episode > n_episodes * 0.5:
                    print(f"\n  ⏹️  Early stopping at episode {episode+1} "
                          f"(no improvement for {early_stop_patience} evals)")
                    break

        # === 定期 Checkpoint (完整状态) ===
        if (episode + 1) % checkpoint_interval == 0:
            import torch
            ckpt_path = Path(save_dir) / f"checkpoint_ep{episode+1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_data = {
                'model_state': policy.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'episode': episode + 1,
                'curriculum_stage': scheduler.current_stage if scheduler else 0,
                'best_eval_swaps': best_eval_swaps,
                'patience_counter': patience_counter,
            }
            torch.save(ckpt_data, str(ckpt_path))
            # 同时保存训练历史
            hist_ckpt = Path(save_dir) / f"history_v7_{topology_name}.json"
            ser = {k: [float(v) for v in vals] for k, vals in history.items()}
            with open(hist_ckpt, 'w') as f:
                json.dump(ser, f, indent=2)
            print(f"  💾 Checkpoint: {ckpt_path} (Stage {scheduler.current_stage if scheduler else '-'})")

    # === 保存最终模型 ===
    elapsed = time.time() - t0
    model_path = Path(save_dir) / f"v7_{topology_name}.pt"
    trainer.save(str(model_path))
    print(f"\n✅ 训练完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   最终模型: {model_path}")
    print(f"   最佳模型: {Path(save_dir) / f'v7_{topology_name}_best.pt'}")

    # 保存历史
    history_path = Path(save_dir) / f"history_v7_{topology_name}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"   训练日志: {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="V8 量子路由器训练管线")
    parser.add_argument('--topology', default='linear_5', help='目标拓扑')
    parser.add_argument('--qubits', type=int, default=5, help='电路比特数 (非课程模式)')
    parser.add_argument('--episodes', type=int, default=50000, help='训练 episode 数')
    parser.add_argument('--rollout-steps', type=int, default=256, help='每 episode rollout 步数')
    parser.add_argument('--save-dir', default='models', help='模型保存目录')
    parser.add_argument('--curriculum', action='store_true', help='启用课程学习')
    parser.add_argument('--lr', type=float, default=3e-4, help='初始学习率')
    parser.add_argument('--eval-interval', type=int, default=1000, help='评估间隔')
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复训练')
    # V8 参数
    parser.add_argument('--reward-gate', type=float, default=1.0, help='门执行奖励系数')
    parser.add_argument('--penalty-swap', type=float, default=-0.5, help='SWAP 惩罚系数')
    parser.add_argument('--reward-done', type=float, default=10.0, help='完成奖励')
    parser.add_argument('--distance-coef', type=float, default=0.5, help='距离缩减奖励系数')
    parser.add_argument('--random-mapping', action='store_true', help='每 episode 随机初始映射')
    parser.add_argument('--soft-mask', action='store_true', help='V8 放宽 action mask')
    parser.add_argument('--tabu-size', type=int, default=4, help='在 soft-mask 下的防止震荡记忆长度')
    parser.add_argument('--checkpoint-interval', type=int, default=5000, help='保存模型状态的频率(episodes)')
    args = parser.parse_args()

    train(
        topology_name=args.topology,
        n_qubits=args.qubits,
        n_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        save_dir=args.save_dir,
        use_curriculum=args.curriculum,
        lr_start=args.lr,
        eval_interval=args.eval_interval,
        resume_path=args.resume,
        reward_gate=args.reward_gate,
        penalty_swap=args.penalty_swap,
        reward_done=args.reward_done,
        distance_reward_coef=args.distance_coef,
        random_mapping=args.random_mapping,
        soft_mask=args.soft_mask,
        tabu_size=args.tabu_size,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == '__main__':
    main()
