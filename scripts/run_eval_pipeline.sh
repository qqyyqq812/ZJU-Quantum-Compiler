#!/bin/bash
# V7.2 训练完成后的自动评测流水线
# 用法: bash run_eval_pipeline.sh [model_dir]

set -e

MODEL_DIR="${1:-models/v7_final_v2}"
TOPO="linear_5"
RESULTS_DIR="results/v7_final"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/projects/量子电路
source .venv/bin/activate

echo "================================================"
echo "🚀 V7.2 自动评测流水线"
echo "   模型目录: $MODEL_DIR"
echo "   拓扑: $TOPO"
echo "================================================"

# 1. 选择最佳模型
BEST_MODEL="$MODEL_DIR/v7_${TOPO}_best.pt"
if [ ! -f "$BEST_MODEL" ]; then
    BEST_MODEL="$MODEL_DIR/v7_${TOPO}.pt"
fi
echo "📦 使用模型: $BEST_MODEL"

# 2. AI-only 快速评测（不依赖 SABRE，避免 C++ 死锁）
echo ""
echo "📊 步骤 1: AI-only 快速评测..."
python -u -c "
import torch
from src.benchmarks.circuits import generate_qft, generate_qaoa, generate_random, generate_grover
from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork

cm = get_topology('$TOPO')
env_tmp = QuantumRoutingEnv(coupling_map=cm)
policy = PolicyNetwork(obs_dim=env_tmp.observation_space.shape[0], n_actions=env_tmp.action_space.n)
policy.load_state_dict(torch.load('$BEST_MODEL', map_location='cpu', weights_only=True))
policy.eval()

circuits = [
    ('random_5_d2', generate_random(5, depth=2, seed=42)),
    ('random_5_d3', generate_random(5, depth=3, seed=42)),
    ('random_5_d4', generate_random(5, depth=4, seed=42)),
    ('random_5_d5', generate_random(5, depth=5, seed=42)),
    ('random_5_d6', generate_random(5, depth=6, seed=42)),
    ('qft_5', generate_qft(5)),
    ('qaoa_5_p1', generate_qaoa(5, p=1)),
    ('qaoa_5_p2', generate_qaoa(5, p=2)),
    ('grover_5', generate_grover(5)),
]

print('Circuit            | CX | AI_SWAPs | Done | Steps')
print('-------------------|----|----------|------|------')
results = []
for name, qc in circuits:
    cx_count = dict(qc.count_ops()).get('cx', 0)
    env = QuantumRoutingEnv(coupling_map=cm, max_steps=1000)
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
    status = 'YES' if completed else 'NO'
    print(f'{name:20s}| {cx_count:3d}| {env._total_swaps:9d}| {status:5s}| {env._step_count}')
    results.append({'name': name, 'cx': cx_count, 'ai_swaps': env._total_swaps, 'done': completed, 'steps': env._step_count})

import json
from pathlib import Path
Path('$RESULTS_DIR').mkdir(parents=True, exist_ok=True)
with open('$RESULTS_DIR/ai_only_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n💾 结果已保存: $RESULTS_DIR/ai_only_eval.json')
"

# 3. SABRE 独立评测（单独进程，避免与 AI 模型冲突）
echo ""
echo "📊 步骤 2: SABRE 基线评测..."
timeout 120 python -u -c "
from src.benchmarks.circuits import generate_qft, generate_qaoa, generate_random, generate_grover
from src.benchmarks.topologies import get_topology
from src.benchmarks.evaluate import evaluate_compiler

cm = get_topology('$TOPO')
circuits = [
    ('random_5_d2', generate_random(5, depth=2, seed=42)),
    ('random_5_d3', generate_random(5, depth=3, seed=42)),
    ('random_5_d4', generate_random(5, depth=4, seed=42)),
    ('random_5_d5', generate_random(5, depth=5, seed=42)),
    ('random_5_d6', generate_random(5, depth=6, seed=42)),
    ('qft_5', generate_qft(5)),
    ('qaoa_5_p1', generate_qaoa(5, p=1)),
    ('qaoa_5_p2', generate_qaoa(5, p=2)),
    ('grover_5', generate_grover(5)),
]

import json
from pathlib import Path
results = []
print('Circuit            | CX | SABRE_SWAPs')
print('-------------------|----|------------')
for name, qc in circuits:
    cx_count = dict(qc.count_ops()).get('cx', 0)
    r = evaluate_compiler(qc, cm, optimization_level=3)
    print(f'{name:20s}| {cx_count:3d}| {r.extra_swap:12d}')
    results.append({'name': name, 'cx': cx_count, 'sabre_swaps': r.extra_swap})

Path('$RESULTS_DIR').mkdir(parents=True, exist_ok=True)
with open('$RESULTS_DIR/sabre_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n💾 结果已保存: $RESULTS_DIR/sabre_baseline.json')
" || echo "⚠️ SABRE 评测超时或失败，跳过"

# 4. 训练曲线可视化
echo ""
echo "📊 步骤 3: 训练曲线可视化..."
HISTORY="$MODEL_DIR/history_v7_${TOPO}.json"
if [ -f "$HISTORY" ]; then
    python -m src.benchmarks.plot_training "$HISTORY" "$RESULTS_DIR"
else
    echo "⚠️ 训练历史文件不存在: $HISTORY"
fi

# 5. 生成对比报告
echo ""
echo "📊 步骤 4: 生成对比报告..."
python -u -c "
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
ai_file = results_dir / 'ai_only_eval.json'
sabre_file = results_dir / 'sabre_baseline.json'

if not ai_file.exists():
    print('❌ AI 评测结果不存在')
    exit(1)

with open(ai_file) as f:
    ai_results = json.load(f)

sabre_results = {}
if sabre_file.exists():
    with open(sabre_file) as f:
        for r in json.load(f):
            sabre_results[r['name']] = r['sabre_swaps']

report = '# V7.2 量子电路 AI 路由器评测报告\n\n'
report += f'- **模型**: \`$BEST_MODEL\`\n'
report += f'- **拓扑**: \`$TOPO\` (5 qubits)\n'
report += f'- **基线**: Qiskit SABRE (O3)\n\n'
report += '## 基准测试结果\n\n'
report += '| 电路 | CX 数 | AI SWAPs | SABRE SWAPs | 比率 | 完成 |\n'
report += '|------|-------|----------|-------------|------|------|\n'

total_ai, total_sabre, count = 0, 0, 0
for r in ai_results:
    name = r['name']
    ai_sw = r['ai_swaps']
    sabre_sw = sabre_results.get(name, '-')
    done = '✅' if r['done'] else '❌'
    
    if isinstance(sabre_sw, int) and sabre_sw > 0 and ai_sw > 0:
        ratio = f'{ai_sw / sabre_sw:.2f}x'
        total_ai += ai_sw
        total_sabre += sabre_sw
        count += 1
    else:
        ratio = '-'
    
    sabre_str = str(sabre_sw) if isinstance(sabre_sw, int) else '-'
    report += f'| {name} | {r[\"cx\"]} | {ai_sw} | {sabre_str} | {ratio} | {done} |\n'

if count > 0:
    overall = total_ai / total_sabre
    report += f'| **总计** | - | **{total_ai}** | **{total_sabre}** | **{overall:.2f}x** | - |\n'

report += '\n> 比率 < 1.0x 表示 AI 优于 SABRE, > 1.0x 表示 SABRE 更优\n'

report_path = results_dir / 'evaluation_report.md'
with open(report_path, 'w') as f:
    f.write(report)
print(f'📄 评测报告已保存: {report_path}')
print(report)
"

echo ""
echo "================================================"
echo "✅ 评测流水线完成!"
echo "   结果目录: $RESULTS_DIR"
echo "================================================"
