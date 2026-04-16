import torch
import json
from pathlib import Path
from src.benchmarks.circuits import generate_qft, generate_qaoa, generate_random, generate_grover
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork
from src.benchmarks.evaluate import evaluate_compiler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 参数配置
model_path = 'models/v12_tokyo20/checkpoint_ep50000.pt'
topo_name = 'ibm_tokyo'
results_dir = 'results/v12_tokyo20'
Path(results_dir).mkdir(parents=True, exist_ok=True)

print(f"📦 载入模型: {model_path} 拓扑: {topo_name}")
cm = get_topology(topo_name)
env_tmp = QuantumRoutingEnv(coupling_map=cm)
policy = PolicyNetwork(obs_dim=env_tmp.observation_space.shape[0], n_actions=env_tmp.action_space.n)
policy.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
policy.eval()

# 创建更大的测试集用于 20Q 晶片，但限制深度避免时间过长
circuits = [
    ('random_10_d3', generate_random(10, depth=3, seed=42)),
    ('random_15_d4', generate_random(15, depth=4, seed=42)),
    ('qft_10', generate_qft(10)),
    ('qft_15', generate_qft(15)),
    ('qaoa_10_p1', generate_qaoa(10, p=1)),
]

results = []
print("Circuit            | CX | AI_SWAPs | SABRE_SWAPs | Done")
print("-------------------|----|----------|-------------|-----")

sabre_cx = {}
ai_cx = {}

for name, qc in circuits:
    cx_count = dict(qc.count_ops()).get('cx', 0)
    
    # AI 评测
    env = QuantumRoutingEnv(coupling_map=cm, max_steps=2000)
    env.set_circuit(qc)
    obs, info = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            mask = env.get_action_mask()
            action, _, _ = policy.get_action(obs, action_mask=mask, gnn_input=info.get('gnn_input'))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
    completed = env._dag.is_done()
    ai_swaps = env._total_swaps
    
    # SABRE 评测
    try:
        r = evaluate_compiler(qc, cm, optimization_level=3)
        sabre_swaps = r.extra_swap
    except Exception as e:
        sabre_swaps = -1
        
    status = 'YES' if completed else 'NO'
    print(f"{name:19s}| {cx_count:3d}| {ai_swaps:9d}| {sabre_swaps:11d}| {status}")
    
    results.append({
        'circuit_name': name,
        'topology_name': topo_name,
        'compiler_name': 'ai_router',
        'compiled_cx': cx_count + ai_swaps * 3
    })
    results.append({
        'circuit_name': name,
        'topology_name': topo_name,
        'compiler_name': 'sabre',
        'compiled_cx': cx_count + sabre_swaps * 3 if sabre_swaps >= 0 else 0
    })

    sabre_cx[f"{name}@{topo_name}"] = cx_count + sabre_swaps * 3 if sabre_swaps >= 0 else 0
    ai_cx[f"{name}@{topo_name}"] = cx_count + ai_swaps * 3

json_path = f"{results_dir}/ai_vs_sabre_v12.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

# 画图
keys = sorted(sabre_cx.keys())
x = np.arange(len(keys))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, [sabre_cx.get(k, 0) for k in keys], width,
               label='SABRE', color='#42A5F5')
bars2 = ax.bar(x + width/2, [ai_cx.get(k, 0) for k in keys], width,
               label='AI Router (V12)', color='#EF5350')

ax.set_ylabel('Compiled CX Count')
ax.set_title(f'AI Router vs SABRE on {topo_name} (20Q)')
ax.set_xticks(x)
ax.set_xticklabels([k.split('@')[0] for k in keys], rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = Path(results_dir) / 'ai_vs_sabre_v12_comparison.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✅ 图表已保存: {fig_path}")

