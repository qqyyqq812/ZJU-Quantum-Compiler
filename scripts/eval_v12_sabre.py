import json
from pathlib import Path
from src.benchmarks.circuits import generate_qft, generate_qaoa, generate_random, generate_grover
from src.benchmarks.topologies import get_topology
from src.benchmarks.evaluate import evaluate_compiler

topo_name = 'ibm_tokyo'
results_dir = 'results/v12_tokyo20'
Path(results_dir).mkdir(parents=True, exist_ok=True)

print("加载 SABRE 环境...")
cm = get_topology(topo_name)
circuits = [
    ('random_5_d2', generate_random(5, depth=2, seed=42)),
    ('random_7_d3', generate_random(7, depth=3, seed=42)),
    ('qft_5', generate_qft(5)),
    ('qft_7', generate_qft(7)),
]

results = []
print("开始 SABRE 评估...")
for name, qc in circuits:
    print(f"SABRE 评估: {name}")
    cx_count = dict(qc.count_ops()).get('cx', 0)
    r = evaluate_compiler(qc, cm, optimization_level=3)
    results.append({
        'name': name,
        'sabre_swaps': r.extra_swap,
        'cx': cx_count
    })
    print(f"-> SABRE Swap: {r.extra_swap}")

with open(f"{results_dir}/sabre.json", 'w') as f:
    json.dump(results, f)
print("✅ SABRE 评测完成")
