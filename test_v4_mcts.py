import sys
import time
sys.path.insert(0, '/home/qq/projects/量子电路')

from src.benchmarks.topologies import get_topology
from src.compiler.pass_manager import AIRouter
from src.benchmarks.circuits import get_benchmark_suite
from src.benchmarks.evaluate import evaluate_compiler

TOPOLOGY = 'linear_5'
MODEL_PATH = f'models/router_v3_{TOPOLOGY}.pt'
cm = get_topology(TOPOLOGY)

print("="*60)
print(f"V4 架构 (MCTS + Crosstalk Penalty) 综合测试")
print(f"Topology: {TOPOLOGY} | Base Model: {MODEL_PATH}")
print("="*60)

# V3 贪心路由器
router_v3 = AIRouter(cm, model_path=MODEL_PATH, use_gnn=False, use_mcts=False)
# V4 MCTS 路由器 (30次仿真前瞻搜索)
router_v4 = AIRouter(cm, model_path=MODEL_PATH, use_gnn=False, use_mcts=True)

suite = get_benchmark_suite([5])

print(f"{'Circuit':<15s} | {'SABRE':<8s} | {'V3 Greedy':<10s} | {'V4 MCTS':<10s}")
print("-" * 60)

total_sabre = 0
total_v3 = 0
total_v4 = 0

for bench in suite:
    if cm.size() < bench['n_qubits']:
        continue
        
    sabre = evaluate_compiler(bench['circuit'], cm, circuit_name=bench['name'],
                              topology_name=TOPOLOGY, compiler_name='sabre_O1')
    
    t0 = time.time()
    v3_res = router_v3.route_count_only(bench['circuit'])
    v3_time = time.time() - t0
    
    t0 = time.time()
    v4_res = router_v4.route_count_only(bench['circuit'])
    v4_time = time.time() - t0
    
    print(f"{bench['name']:<15s} | {sabre.extra_swap:<8d} | {v3_res['ai_swaps']:<3d} ({v3_time:.1f}s) | {v4_res['ai_swaps']:<3d} ({v4_time:.1f}s)")
    
    total_sabre += sabre.extra_swap
    total_v3 += v3_res['ai_swaps']
    total_v4 += v4_res['ai_swaps']

print("-" * 60)
ratio_v3 = total_v3 / max(total_sabre, 1)
ratio_v4 = total_v4 / max(total_sabre, 1)
print(f"SABRE Total: {total_sabre}")
print(f"V3 Greedy Total: {total_v3} ({ratio_v3:.2f}x)")
print(f"V4 MCTS Total: {total_v4} ({ratio_v4:.2f}x)")
print("="*60)
