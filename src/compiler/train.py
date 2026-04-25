"""
V13 工业级训练管线 (AsyncVectorEnv 全并行版)
==================
核心改进 (V13):
1. SABRE 相对奖励 — 信号极纯净，彻底消除 Reward Drowning
2. 默认随机初始映射 — 消除 Identity Mapping 偏见
3. Soft Mask — 允许"暂时变差"的 SWAP，打破局部最优
4. 调优课程阈值 — 匹配 IBM Tokyo 20Q 的真实 SABRE 基线
"""

from __future__ import annotations

import torch
import argparse
import json
import time
from pathlib import Path
from collections import deque
import random as _random

import numpy as np
import gymnasium as gym
from qiskit.transpiler import CouplingMap

from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork, PPOTrainer, RolloutBuffer
from src.compiler.curriculum import CurriculumScheduler, STAGES


def _random_mapping_fn(circuit, coupling_map):
    n_log = circuit.num_qubits
    n_phys = coupling_map.size()
    phys = list(range(n_phys))
    _random.shuffle(phys)
    return {i: phys[i] for i in range(n_log)}

class EnvFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self):
        from src.compiler.env import QuantumRoutingEnv
        return QuantumRoutingEnv(**self.kwargs)


def make_training_circuits(n_qubits: int, n_circuits: int = 20, seed: int = 42) -> list:
    circuits = []
    for i in range(n_circuits):
        circuits.append(generate_random(n_qubits, depth=n_qubits, seed=seed + i))
    circuits.append(generate_qft(n_qubits))
    circuits.append(generate_qaoa(n_qubits, p=1))
    return circuits


def evaluate_model(policy: PolicyNetwork, cm: CouplingMap, circuits: list, n_eval: int = 5, soft_mask: bool = False, tabu_size: int = 0) -> dict:
    from qiskit import transpile
    env = QuantumRoutingEnv(coupling_map=cm, soft_mask=soft_mask, tabu_size=tabu_size)
    total_swaps, total_steps, completed = 0, 0, 0
    sabre_swaps = 0

    for qc in circuits[:n_eval]:
        try:
            # 运行 Qiskit 官方 Sabre 作为基准
            try:
                transpiled_qc = transpile(qc, coupling_map=cm, optimization_level=3, routing_method='sabre', initial_layout=list(range(qc.num_qubits)))
                sabre_swaps += transpiled_qc.count_ops().get('swap', 0)
            except Exception as e:
                sabre_swaps += 0

            env.set_circuit(qc)
            obs, info = env.reset()
            done = False

            for _ in range(env.max_steps):
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
            pass

    n = min(n_eval, len(circuits))
    return {
        'avg_swaps': total_swaps / max(n, 1),
        'avg_sabre_swaps': sabre_swaps / max(n, 1),
        'avg_steps': total_steps / max(n, 1),
        'completion_rate': completed / max(n, 1),
    }


def train(
    topology_name: str = "linear_5",
    n_qubits: int = 5,
    n_episodes: int = 50000,
    rollout_steps: int = 8192,
    log_interval: int = 100,
    eval_interval: int = 1000,
    checkpoint_interval: int = 5000,
    save_dir: str = "models",
    use_curriculum: bool = True,
    lr_start: float = 3e-4,
    lr_end: float = 1e-5,
    entropy_start: float = 0.05,
    entropy_end: float = 0.001,
    early_stop_patience: int = 100,
    resume_path: str | None = None,
    reward_gate: float = 0.3,        # V13: 降低，消除 Reward Drowning
    penalty_swap: float = -0.5,
    reward_done: float = 0.0,        # V13: 由 SABRE 相对奖励替代
    distance_reward_coef: float = 0.3,  # V13: 降权 shaping
    random_mapping: bool = True,     # V13: 默认开启随机映射
    soft_mask: bool = True,          # V13: 默认开启 Soft Mask
    tabu_size: int = 4,
    num_envs: int = 20,              # V14: 可从 yaml 指定
    mini_batch_size: int = 4096,     # V14: 可从 yaml 指定
    early_stage_reward_floor: float = 5.0,   # V14 §V14-3: stage<=2 完成奖励下限
    early_stage_sabre_weight: float = 0.1,   # V14 §V14-3: stage<=2 SABRE 权重
    max_steps: int = 2000,           # V14+: env 每 episode 最大步数
    force_stage: int | None = None,  # V14.1: 纯权重 resume 时强制课程阶段
) -> dict:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    cm = get_topology(topology_name)
    # V14: 保证 save_dir 存在（yaml 指定路径可能还没建目录）
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"🔥 启动 AsyncVectorEnv ({num_envs} 进程并行推演矩阵)")

    env_factories = [EnvFactory(
        coupling_map=cm,
        reward_gate=reward_gate,
        penalty_swap=penalty_swap,
        reward_done=reward_done,
        distance_reward_coef=distance_reward_coef,
        initial_mapping_fn=_random_mapping_fn if random_mapping else None,
        soft_mask=soft_mask,
        tabu_size=tabu_size,
        early_stage_reward_floor=early_stage_reward_floor,
        early_stage_sabre_weight=early_stage_sabre_weight,
        max_steps=max_steps,
    ) for _ in range(num_envs)]
    
    envs = gym.vector.AsyncVectorEnv(env_factories)
    
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    policy = PolicyNetwork(obs_dim=obs_dim, n_actions=n_actions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    print(f"🖥️  训练设备: {device}")
    
    trainer = PPOTrainer(policy, lr=lr_start, entropy_coef=entropy_start)
    scheduler = CurriculumScheduler(max_n_qubits=cm.size()) if use_curriculum else None
    
    start_episode = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            trainer.policy.load_state_dict(ckpt['model_state'])
            if 'optimizer_state' in ckpt:
                trainer.optimizer.load_state_dict(ckpt['optimizer_state'])
            start_episode = ckpt.get('episode', 0)
            if scheduler and 'curriculum_stage' in ckpt:
                target_stage = ckpt['curriculum_stage']
                while scheduler.current_stage < target_stage:
                    scheduler._promote()
                print(f"🔄 课程状态恢复: Stage {scheduler.current_stage} ({scheduler.stage_config.name})")
                # V14.1 fix: resume 后必须把 stage 传给所有 env worker，否则 env 还以为 stage=0
                try:
                    envs.call("set_curriculum_stage", scheduler.current_stage)
                except Exception as e:
                    print(f"⚠️  set_curriculum_stage 传播失败: {e}")
            best_eval_swaps = ckpt.get('best_eval_swaps', float('inf'))
            patience_counter = ckpt.get('patience_counter', 0)
            print(f"🔄 完整 checkpoint 重接: {resume_path} (ep{start_episode})")
        else:
            trainer.load(resume_path)
            if scheduler:
                # V14.1: 用 force_stage 精确控制，否则推到 max_stage (旧行为)
                tgt = force_stage if force_stage is not None else scheduler._max_stage
                while scheduler.current_stage < tgt:
                    scheduler._promote()
                try:
                    envs.call("set_curriculum_stage", scheduler.current_stage)
                except Exception as e:
                    print(f"⚠️  set_curriculum_stage 传播失败: {e}")
            print(f"🔄 从旧权重恢复: {resume_path} → Stage {scheduler.current_stage if scheduler else '?'}")

    if use_curriculum:
        circuits = scheduler.circuits
        print(f"📚 当前阶段: {scheduler.stage_config.name} ({scheduler.stage_config.n_qubits}Q)")
    else:
        circuits = make_training_circuits(n_qubits)

    history = {
        'episode_rewards': [], 'episode_swaps': [], 'episode_lengths': [],
        'policy_losses': [], 'value_losses': [], 'eval_swaps': [], 'eval_sabre_swaps': [],
        'eval_completion': [], 'curriculum_stages': [],
    }

    if resume_path:
        hist_file = Path(save_dir) / f"history_v7_{topology_name}.json"
        if hist_file.exists():
            try:
                with open(hist_file) as f:
                    old_hist = json.load(f)
                for k in history:
                    if k in old_hist:
                        history[k] = old_hist[k]
            except Exception:
                pass

    reward_window = deque(maxlen=100)
    swap_window = deque(maxlen=100)

    if 'best_eval_swaps' not in dir():
        best_eval_swaps = float('inf')
    if 'patience_counter' not in dir():
        patience_counter = 0

    print(f"🚀 Vectorized V8启动: {n_episodes} episodes | Rollout: {rollout_steps}")
    t0 = time.time()
    last_log_time = t0

    global_episodes = start_episode
    episode_rewards = np.zeros(num_envs)
    
    circuit = circuits[global_episodes % len(circuits)]
    envs.call("set_circuit", circuit)
    obs, info = envs.reset()
    
    last_eval_episode = global_episodes
    last_log_episode = global_episodes
    last_ckpt_episode = global_episodes  # V14 fix: 准确触发 checkpoint 保存

    while global_episodes < n_episodes:
        progress = global_episodes / n_episodes
        current_lr = lr_start + (lr_end - lr_start) * progress
        current_entropy = entropy_start + (entropy_end - entropy_start) * progress
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = current_lr
        trainer.entropy_coef = current_entropy

        buffer = RolloutBuffer.create()
        steps_per_env = max(1, rollout_steps // num_envs)

        for _ in range(steps_per_env):
            masks_tuple = envs.call("get_action_mask")
            mask_batch = np.array(masks_tuple)
            gnn_batch = info.get('gnn_input') 
            
            actions, log_probs, values = policy.get_action_batch(obs, action_mask_batch=mask_batch, gnn_inputs_batch=gnn_batch)
            next_obs, rewards, terminateds, truncateds, next_infos = envs.step(actions)

            for i in range(num_envs):
                gnn_i = {'graph': gnn_batch['graph'][i], 'swap_edges': gnn_batch['swap_edges'][i]} if isinstance(gnn_batch, dict) else gnn_batch[i]
                buffer.add(obs[i], actions[i], rewards[i], log_probs[i], values[i], terminateds[i] or truncateds[i], gnn_input=gnn_i)
                episode_rewards[i] += rewards[i]

                if terminateds[i] or truncateds[i]:
                    # V13 fix: AsyncVectorEnv 的 final_info 提取兼容多种格式
                    ep_swaps = None
                    ep_len = None

                    # 尝试方式 1: gymnasium 标准 final_info
                    if isinstance(next_infos, dict) and 'final_info' in next_infos:
                        fi = next_infos['final_info']
                        if fi is not None and i < len(fi) and fi[i] is not None:
                            ep_swaps = fi[i].get('total_swaps', None)
                            ep_len = fi[i].get('step_count', None)

                    # 尝试方式 2: 直接从 next_infos 提取 (某些 gym 版本)
                    if ep_swaps is None and isinstance(next_infos, dict):
                        if 'total_swaps' in next_infos:
                            ts = next_infos['total_swaps']
                            if hasattr(ts, '__len__') and i < len(ts):
                                ep_swaps = int(ts[i])
                        if 'step_count' in next_infos:
                            sc = next_infos['step_count']
                            if hasattr(sc, '__len__') and i < len(sc):
                                ep_len = int(sc[i])

                    # 尝试方式 3: next_infos 是列表
                    if ep_swaps is None and isinstance(next_infos, (list, tuple)):
                        if i < len(next_infos) and isinstance(next_infos[i], dict):
                            ep_swaps = next_infos[i].get('total_swaps', None)
                            ep_len = next_infos[i].get('step_count', None)

                    if ep_swaps is not None:
                        history['episode_rewards'].append(float(episode_rewards[i]))
                        history['episode_swaps'].append(int(ep_swaps))
                        history['episode_lengths'].append(int(ep_len or 0))

                        reward_window.append(episode_rewards[i])
                        swap_window.append(ep_swaps)

                        if scheduler:
                            promoted = scheduler.report_episode(ep_swaps)
                            history['curriculum_stages'].append(scheduler.current_stage)
                            if promoted:
                                new_cfg = scheduler.stage_config
                                circuits = scheduler.circuits
                                print(f"\n  🎓 课程进阶突破! → 阶段 {scheduler.current_stage} ({new_cfg.name}, {new_cfg.n_qubits}Q)")
                                # V14 §V14-2/§V14-3: 传播 stage 给所有 env
                                try:
                                    envs.call("set_curriculum_stage", scheduler.current_stage)
                                except Exception:
                                    pass
                                circuit = circuits[0]
                                envs.call("set_circuit", circuit)
                                next_obs, next_infos = envs.reset()
                                episode_rewards = np.zeros(num_envs)
                                break

                    global_episodes += 1
                    episode_rewards[i] = 0.0

                    # V13: 每个 episode 结束后轮换电路
                    try:
                        circuit = circuits[global_episodes % len(circuits)]
                        envs.call("set_circuit", circuit)
                    except Exception:
                        pass
                    
            obs = next_obs
            info = next_infos
            if global_episodes >= n_episodes:
                break

        if len(buffer) > 0:
            metrics = trainer.update(buffer, mini_batch_size=mini_batch_size)
            history['policy_losses'].append(metrics['policy_loss'])
            history['value_losses'].append(metrics['value_loss'])

        if global_episodes - last_log_episode >= log_interval:
            avg_reward = np.mean(reward_window) if reward_window else 0
            avg_swaps = np.mean(swap_window) if swap_window else 0
            elapsed = time.time() - t0
            eps_per_sec = (global_episodes - start_episode) / (elapsed + 1e-6)
            stage_str = f" [Stage {scheduler.current_stage}:{scheduler.stage_config.name}]" if scheduler else ""
            print(f"  [{global_episodes:>6}/{n_episodes}]{stage_str} "
                  f"R={avg_reward:>7.1f} SWAP={avg_swaps:>5.1f} "
                  f"LR={current_lr:.1e} H={current_entropy:.3f} "
                  f"({eps_per_sec:.1f} eps/s 并行加速中)")
            last_log_episode = global_episodes

            hist_ckpt = Path(save_dir) / f"history_v7_{topology_name}.json"
            hist_ckpt.parent.mkdir(parents=True, exist_ok=True)
            ser = {k: [float(v) for v in vals] for k, vals in history.items()}
            with open(hist_ckpt, 'w') as f:
                json.dump(ser, f, indent=2)

        if global_episodes - last_eval_episode >= eval_interval:
            eval_n = scheduler.stage_config.n_qubits if scheduler else n_qubits
            eval_circuits = [
                generate_qft(min(eval_n, cm.size())),
                generate_qaoa(min(eval_n, cm.size()), p=1),
                generate_random(min(eval_n, cm.size()), depth=eval_n, seed=999),
            ]
            eval_result = evaluate_model(policy, cm, eval_circuits, soft_mask=soft_mask, tabu_size=tabu_size)
            history['eval_swaps'].append(eval_result['avg_swaps'])
            history['eval_sabre_swaps'].append(eval_result['avg_sabre_swaps'])
            history['eval_completion'].append(eval_result['completion_rate'])
            print(f"  📊 EVAL: avg_swap={eval_result['avg_swaps']:.1f} (vs SABRE={eval_result['avg_sabre_swaps']:.1f}) done={eval_result['completion_rate']:.0%}")
            
            if eval_result['avg_swaps'] < best_eval_swaps:
                best_eval_swaps = eval_result['avg_swaps']
                trainer.save(str(Path(save_dir) / f"v7_{topology_name}_best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
            last_eval_episode = global_episodes

        # V14 fix: 简化条件, 确保每 checkpoint_interval episodes 一定触发
        # 之前的 modulo 比较浮点除法总是 false, 永远不保存 checkpoint.
        if global_episodes - last_ckpt_episode >= checkpoint_interval:
            ckpt_path = Path(save_dir) / f"checkpoint_ep{global_episodes}.pt"
            torch.save({
                'model_state': policy.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'episode': global_episodes,
                'curriculum_stage': scheduler.current_stage if scheduler else 0,
                'best_eval_swaps': best_eval_swaps,
                'patience_counter': patience_counter,
            }, str(ckpt_path))
            print(f"  💾 Checkpoint: {ckpt_path}")
            last_ckpt_episode = global_episodes
            # 保留最近 3 个 checkpoint 防磁盘爆掉
            ckpts = sorted(Path(save_dir).glob("checkpoint_ep*.pt"),
                           key=lambda p: int(p.stem.split("ep")[1]))
            for old in ckpts[:-3]:
                old.unlink(missing_ok=True)

    envs.close()
    trainer.save(str(Path(save_dir) / f"v7_{topology_name}.pt"))
    print("\n✅ 训练完成（无损提速版）")
    return history


def main():
    parser = argparse.ArgumentParser(description="V14 量子路由器训练管线")
    # V14: yaml 配置为主路径
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config path (V14+ recommended). 若提供，命令行参数会覆盖 yaml')

    # 兼容 V13 及以前的命令行参数
    parser.add_argument('--topology', default='linear_5')
    parser.add_argument('--qubits', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--rollout-steps', type=int, default=32768)
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reward-gate', type=float, default=0.3)
    parser.add_argument('--penalty-swap', type=float, default=-0.5)
    parser.add_argument('--reward-done', type=float, default=0.0)
    parser.add_argument('--distance-coef', type=float, default=0.3)
    parser.add_argument('--random-mapping', action='store_true', default=True)
    parser.add_argument('--no-random-mapping', dest='random_mapping', action='store_false')
    parser.add_argument('--soft-mask', action='store_true', default=True)
    parser.add_argument('--no-soft-mask', dest='soft_mask', action='store_false')
    parser.add_argument('--tabu-size', type=int, default=4)
    parser.add_argument('--checkpoint-interval', type=int, default=2000)
    parser.add_argument('--force-stage', type=int, default=None,
                        help='V14.1: 从纯权重 resume 时强制 curriculum stage (绕过坏 checkpoint)')
    args = parser.parse_args()

    # V14: 若传了 --config，走 yaml 路径
    if args.config:
        from src.utils.config import load_config
        cfg = load_config(args.config)
        topology = cfg['topology']
        training = cfg['training']
        reward = cfg['reward']
        env_cfg = cfg['environment']
        curr = cfg.get('curriculum', {})
        paths = cfg.get('paths', {})

        save_dir = paths.get('save_dir', args.save_dir)
        # V14.1 fix: CLI --resume 优先于 yaml (yaml 里常 null 导致覆盖)
        resume = args.resume if args.resume else paths.get('resume', None)

        train(
            topology_name=topology['name'],
            n_qubits=topology['n_qubits'],
            n_episodes=training['episodes'],
            rollout_steps=training['rollout_steps'],
            log_interval=training.get('log_interval', 100),
            eval_interval=training.get('eval_interval', 500),
            checkpoint_interval=training.get('checkpoint_interval', 2000),
            save_dir=save_dir,
            use_curriculum=curr.get('enabled', True),
            lr_start=training.get('lr_start', 3e-4),
            lr_end=training.get('lr_end', 1e-5),
            entropy_start=training.get('entropy_start', 0.05),
            entropy_end=training.get('entropy_end', 0.001),
            resume_path=resume,
            reward_gate=reward.get('gate', 1.0),
            penalty_swap=reward.get('swap', -0.5),
            reward_done=reward.get('done', 5.0),
            distance_reward_coef=reward.get('distance_coef', 0.3),
            random_mapping=env_cfg.get('random_mapping', True),
            soft_mask=env_cfg.get('soft_mask', True),
            tabu_size=env_cfg.get('tabu_size', 4),
            num_envs=training.get('num_envs', 20),
            mini_batch_size=training.get('mini_batch_size', 4096),
            early_stage_reward_floor=reward.get('early_stage_reward_floor', 5.0),
            early_stage_sabre_weight=reward.get('early_stage_sabre_weight', 0.1),
            max_steps=env_cfg.get('max_steps', 2000),
            force_stage=args.force_stage,
        )
        return

    # 兼容 V13 旧路径
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
