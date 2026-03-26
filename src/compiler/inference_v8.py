"""
V8 推理优化器
=============
实现双向遍历 (Bidirectional) 和 Multi-Trial 推理策略，
不需要重训练即可大幅提升 AI 路由器性能。

用法:
    python -m src.compiler.inference_v8 models/v7_final_v2/v7_linear_5_best.pt
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from qiskit import QuantumCircuit

from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork
from src.compiler.mcts import RouterMCTS


@dataclass
class RouteResult:
    """路由结果。"""
    swaps: int
    completed: bool
    steps: int
    initial_mapping: dict


def run_single_route(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
    initial_mapping: dict | None = None,
    max_steps: int = 1000,
) -> RouteResult:
    """单次 AI 路由。"""
    env = QuantumRoutingEnv(coupling_map=coupling_map, max_steps=max_steps)
    env.set_circuit(circuit)
    obs, info = env.reset()
    
    # 如果提供了自定义初始映射，覆盖默认映射
    if initial_mapping is not None:
        env._mapping = dict(initial_mapping)
        # 在新映射下重新检查可执行门
        executed = env._dag.execute_executable(env._mapping, env.coupling_map)
        env._total_gates_executed += len(executed)
        obs = env._get_obs()
        info = env._get_info()
    
    done = False
    with torch.no_grad():
        while not done:
            mask = env.get_action_mask()
            action, _, _ = policy.get_action(
                obs, action_mask=mask, gnn_input=info.get("gnn_input")
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    return RouteResult(
        swaps=env._total_swaps,
        completed=env._dag.is_done(),
        steps=env._step_count,
        initial_mapping=env._mapping.copy(),
    )


def random_initial_mapping(n_logical: int, n_physical: int) -> dict:
    """生成随机初始映射。"""
    physical = list(range(n_physical))
    random.shuffle(physical)
    return {i: physical[i] for i in range(n_logical)}


def compile_bidirectional(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
    n_rounds: int = 3,
    initial_mapping: dict | None = None,
) -> RouteResult:
    """SABRE 式双向遍历推理：前向路由→取最终映射→反转电路→再路由→循环。"""
    n_logical = circuit.num_qubits
    
    # 反转电路（门的执行顺序反转）
    rev_circuit = circuit.reverse_ops()
    
    # 第一轮前向路由，确定一个映射
    if initial_mapping is None:
        initial_mapping = {i: i for i in range(n_logical)}
    
    mapping = dict(initial_mapping)
    best_result = None
    
    for round_idx in range(n_rounds):
        # 前向路由
        fwd_result = run_single_route(circuit, policy, coupling_map, initial_mapping=mapping)
        
        if fwd_result.completed and (best_result is None or fwd_result.swaps < best_result.swaps):
            best_result = fwd_result
        
        # 用前向路由的最终映射作为反向路由的初始映射
        final_mapping = fwd_result.initial_mapping  # 路由完成后的映射状态
        
        # 反向路由（用来探索不同的映射空间）
        rev_result = run_single_route(rev_circuit, policy, coupling_map, initial_mapping=final_mapping)
        
        # 用反向路由的最终映射作为下一轮前向路由的初始映射
        mapping = rev_result.initial_mapping
    
    # 最后一轮前向路由
    final_result = run_single_route(circuit, policy, coupling_map, initial_mapping=mapping)
    if final_result.completed and (best_result is None or final_result.swaps < best_result.swaps):
        best_result = final_result
    
    return best_result if best_result else final_result


def compile_best(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
    n_trials: int = 10,
    n_bidir_rounds: int = 2,
) -> RouteResult:
    """最强推理策略：Multi-Trial + 每个 trial 用双向遍历优化映射。"""
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()
    
    best_result = None
    
    for trial in range(n_trials):
        if trial == 0:
            mapping = {i: i for i in range(n_logical)}
        else:
            mapping = random_initial_mapping(n_logical, n_physical)
        
        result = compile_bidirectional(
            circuit, policy, coupling_map,
            n_rounds=n_bidir_rounds, initial_mapping=mapping,
        )
        
        if result.completed and (best_result is None or result.swaps < best_result.swaps):
            best_result = result
    
    return best_result if best_result else result


def compile_multi_trial(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
    n_trials: int = 10,
) -> RouteResult:
    """Multi-Trial 推理：以不同初始映射运行多次，取最优（无双向遍历）。"""
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()
    best_result = None
    for trial in range(n_trials):
        mapping = ({i: i for i in range(n_logical)} if trial == 0
                   else random_initial_mapping(n_logical, n_physical))
        result = run_single_route(circuit, policy, coupling_map, initial_mapping=mapping)
        if result.completed and (best_result is None or result.swaps < best_result.swaps):
            best_result = result
    return best_result if best_result else result


import copy

def compile_beam_search(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
    beam_width: int = 3,
    branch_factor: int = 3,
    initial_mapping: dict | None = None,
) -> RouteResult:
    """结合 Actor 概率的前瞻 Beam Search，解决 MCTS deepcopy 性能瓶颈"""
    env = QuantumRoutingEnv(coupling_map=coupling_map, max_steps=1000)
    env.set_circuit(circuit)
    obs, info = env.reset()

    if initial_mapping is not None:
        env._mapping = dict(initial_mapping)
        executed = env._dag.execute_executable(env._mapping, env.coupling_map)
        env._total_gates_executed += len(executed)
        obs = env._get_obs()
        info = env._get_info()

    try:
        from src.compiler.light_env import LightweightEnv
        env = LightweightEnv(env)
    except Exception as e:
        print("LightweightEnv error:", e)

    beam = [(0.0, env, obs, info)]
    best_done_env = None
    best_done_score = -float('inf')

    while beam:
        next_beam = []
        for cum_reward, current_env, current_obs, current_info in beam:
            with torch.no_grad():
                obs_t = torch.FloatTensor(current_obs).unsqueeze(0)
                gnn_input = current_info.get('gnn_input')
                if gnn_input is not None and 'graph' in gnn_input:
                    from torch_geometric.data import Batch
                    d_b = Batch.from_data_list([gnn_input['graph']])
                    edges = [gnn_input['swap_edges']]
                    dist, values = policy.forward(obs_t, d_b, edges)
                    logits = dist.logits
                else:
                    logits = torch.zeros((1, policy.n_actions))

            action_mask = current_env.get_action_mask()
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            logits = logits.masked_fill(mask_t == 0, -1e8)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) == 0:
                continue
                
            action_probs = [(a, probs[a]) for a in valid_actions]
            action_probs.sort(key=lambda x: x[1], reverse=True)
            top_actions = [a for a, p in action_probs[:branch_factor]]
            
            for a in top_actions:
                if hasattr(current_env, 'clone'):
                    new_env = current_env.clone()
                else:
                    new_env = copy.deepcopy(current_env)
                new_obs, reward, terminated, truncated, new_info = new_env.step(a)
                new_cum_reward = cum_reward + reward
                
                heuristic_score = new_cum_reward + np.log(probs[a] + 1e-8)
                
                if terminated or truncated:
                    if heuristic_score > best_done_score:
                        best_done_score = heuristic_score
                        best_done_env = new_env
                else:
                    next_beam.append((heuristic_score, new_env, new_obs, new_info))
        
        if len(next_beam) == 0:
            break
            
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    if best_done_env is None and len(beam) > 0:
        best_done_env = beam[0][1]
    elif best_done_env is None:
        best_done_env = env
        
    return RouteResult(
        swaps=best_done_env._total_swaps,
        completed=best_done_env._dag.is_done() if getattr(best_done_env, '_dag', None) else best_done_env.is_done(),
        steps=best_done_env._step_count,
        initial_mapping=best_done_env._mapping.copy(),
    )


def compile_beam_search_multi_trial(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
    n_trials: int = 10,
    beam_width: int = 3,
) -> RouteResult:
    """Beam Search + Multi-Trial"""
    n_logical = circuit.num_qubits
    n_physical = coupling_map.size()
    best_result = None
    
    for trial in range(n_trials):
        mapping = ({i: i for i in range(n_logical)} if trial == 0
                   else random_initial_mapping(n_logical, n_physical))
        result = compile_beam_search(
            circuit, policy, coupling_map,
            beam_width=beam_width, initial_mapping=mapping
        )
        if result.completed and (best_result is None or result.swaps < best_result.swaps):
            best_result = result
            
    return best_result if best_result else RouteResult(0, False, 0, {})


def compile_greedy(
    circuit: QuantumCircuit,
    policy: PolicyNetwork,
    coupling_map,
) -> RouteResult:
    """标准 Greedy 推理（baseline）。"""
    return run_single_route(circuit, policy, coupling_map)


# ─── CLI 入口 ─── #

def load_policy(model_path: str, topology_name: str = "linear_5") -> PolicyNetwork:
    """加载模型。"""
    cm = get_topology(topology_name)
    env = QuantumRoutingEnv(coupling_map=cm)
    policy = PolicyNetwork(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
    )
    policy.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    policy.eval()
    return policy


def run_ablation(model_path: str, topology_name: str = "linear_5"):
    """运行推理策略消融实验。"""
    from src.benchmarks.circuits import (
        generate_random, generate_qft, generate_qaoa, generate_grover,
    )
    
    cm = get_topology(topology_name)
    policy = load_policy(model_path, topology_name)
    
    circuits = [
        ("random_5_d2", generate_random(5, depth=2, seed=42)),
        ("random_5_d3", generate_random(5, depth=3, seed=42)),
        ("random_5_d4", generate_random(5, depth=4, seed=42)),
        ("random_5_d6", generate_random(5, depth=6, seed=42)),
        ("qft_5", generate_qft(5)),
        ("qaoa_5", generate_qaoa(5, p=1)),
        ("grover_5", generate_grover(5)),
    ]
    
    # SABRE 基线
    sabre_swaps = {
        "random_5_d2": 0, "random_5_d3": 2, "random_5_d4": 3,
        "random_5_d6": 4, "qft_5": 10, "qaoa_5": 2, "grover_5": 4,
    }
    
    strategies = [
        ("Greedy", lambda c: compile_greedy(c, policy, cm)),
        ("BiDir", lambda c: compile_bidirectional(c, policy, cm, n_rounds=3)),
        ("MT×10", lambda c: compile_multi_trial(c, policy, cm, n_trials=10)),
        ("Best×10", lambda c: compile_best(c, policy, cm, n_trials=10, n_bidir_rounds=2)),
        ("Beam×10", lambda c: compile_beam_search_multi_trial(c, policy, cm, n_trials=10, beam_width=3)),
        ("MT×100", lambda c: compile_multi_trial(c, policy, cm, n_trials=100)),
    ]
    
    # 表头
    header = f"{'Circuit':20s}| CX | SABRE |"
    for name, _ in strategies:
        header += f" {name:8s}|"
    print(header)
    print("-" * len(header))
    
    totals = {name: 0 for name, _ in strategies}
    total_sabre = 0
    
    for cname, qc in circuits:
        cx = dict(qc.count_ops()).get("cx", 0)
        sabre = sabre_swaps.get(cname, "?")
        row = f"{cname:20s}| {cx:3d}| {sabre:6}|"
        
        for sname, strategy in strategies:
            t0 = time.time()
            result = strategy(qc)
            dt = time.time() - t0
            row += f" {result.swaps:5d}({dt:.1f}s)|"
            totals[sname] += result.swaps
        
        total_sabre += sabre if isinstance(sabre, int) else 0
        print(row)
    
    # 总计
    print("-" * len(header))
    summary = f"{'TOTAL':20s}|  - | {total_sabre:6}|"
    for sname, _ in strategies:
        summary += f" {totals[sname]:8}|"
    print(summary)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python -m src.compiler.inference_v8 <model_path> [topology]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    topo = sys.argv[2] if len(sys.argv) > 2 else "linear_5"
    
    print(f"🧪 V8 推理策略消融实验")
    print(f"   模型: {model_path}")
    print(f"   拓扑: {topo}")
    print()
    
    run_ablation(model_path, topo)
