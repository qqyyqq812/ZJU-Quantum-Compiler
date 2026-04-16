import torch
import json
from pathlib import Path
from src.benchmarks.circuits import generate_qft, generate_qaoa, generate_random, generate_grover
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork

model_path = 'models/v12_tokyo20/checkpoint_ep50000.pt'
topo_name = 'ibm_tokyo'
results_dir = 'results/v12_tokyo20'
Path(results_dir).mkdir(parents=True, exist_ok=True)

print("加载模型...")
cm = get_topology(topo_name)
env_tmp = QuantumRoutingEnv(coupling_map=cm)
policy = PolicyNetwork(obs_dim=env_tmp.observation_space.shape[0], n_actions=env_tmp.action_space.n)
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
if 'model_state' in state_dict:
    state_dict = state_dict['model_state']
policy.load_state_dict(state_dict)
policy.eval()

circuits = [
    ('random_5_d2', generate_random(5, depth=2, seed=42)),
    ('random_7_d3', generate_random(7, depth=3, seed=42)),
    ('qft_5', generate_qft(5)),
    ('qft_7', generate_qft(7)),
]

results = []
print("开始评估...")
for name, qc in circuits:
    print(f"评估: {name}")
    cx_count = dict(qc.count_ops()).get('cx', 0)
    env = QuantumRoutingEnv(coupling_map=cm, max_steps=500)
    env.set_circuit(qc)
    obs, info = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            mask = env.get_action_mask()
            action, _, _ = policy.get_action(obs, action_mask=mask, gnn_input=info.get('gnn_input'))
            # 注意: 如果你的环境只返回4个值,这里会直接触发 ValueError, 这是预设的. 但从先前的脚本来看是5个值
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
    completed = env._dag.is_done()
    results.append({
        'name': name,
        'cx': cx_count,
        'ai_swaps': env._total_swaps,
        'done': completed
    })
    print(f"-> 完成状态:{completed}, Swap: {env._total_swaps}")

with open(f"{results_dir}/ai.json", 'w') as f:
    json.dump(results, f)
print("✅ AI 评测完成")
