import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa, generate_grover
from src.benchmarks.topologies import get_topology
from src.compiler.inference_v8 import load_policy, compile_multi_trial, compile_beam_search_multi_trial
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def run_sota_ablation():
    """
    SOTA 路由算法大比武
    实验目标: 对比解决“软掩码循环震荡”死锁问题的两种前沿方案性能
    变量组
    - 基线: SABRE 的 SWAP 耗费
    - 方案 0: 无掩码 MTx10 (即失败爆表的软掩码)
    - 方案 1: 禁忌记忆法 Tabu-RL (MTx10 + Tabu=3)
    - 方案 2: 波束树搜索 BeamSearch (BMx10 + BW=3)
    """
    topo_name = "ibm_tokyo"
    model_path = "models/v9_tokyo20/v7_ibm_tokyo_best.pt"
    
    cm = get_topology(topo_name)
    policy, obs_dim = load_policy(model_path, topo_name)
    
    circuits = [
        ("qft_5", generate_qft(5)),
        ("qaoa_5", generate_qaoa(5, p=1)),
        ("grover_5", generate_grover(5)),
    ]
    
    def get_sabre(qc):
        pm = generate_preset_pass_manager(optimization_level=3, coupling_map=cm, routing_method="sabre")
        return dict(pm.run(qc).count_ops()).get('swap', 0)
    
    print("=" * 105)
    print("🏆 SOTA 路由算法防御对抗赛 (Topology: IBM Tokyo 20Q)")
    print("=" * 105)
    header = f"{'Circuit':12s} | {'SABRE':7s} | {'无掩码裸跑(软)':16s} | {'Tabu-RL (方案1)':17s} | {'Beam (方案2)':15s}"
    print(header)
    print("-" * 105)
    
    # 指标累加
    metrics = {
        'sabre': {'swaps': 0, 'time': 0},
        'naked': {'swaps': 0, 'time': 0},
        'tabu': {'swaps': 0, 'time': 0},
        'beam': {'swaps': 0, 'time': 0},
    }
    
    for cname, qc in circuits:
        t0 = time.time()
        sabre = get_sabre(qc)
        metrics['sabre']['swaps'] += sabre
        metrics['sabre']['time'] += (time.time() - t0)
        
        # Naked (Soft Mask) -> 预期爆表
        t0 = time.time()
        res_naked = compile_multi_trial(qc, policy, cm, n_trials=10, soft_mask=True, tabu_size=0)
        t_naked = time.time() - t0
        metrics['naked']['swaps'] += res_naked.swaps
        metrics['naked']['time'] += t_naked
        
        # Tabu-RL -> MTx10 + Tabu=3
        t0 = time.time()
        res_tabu = compile_multi_trial(qc, policy, cm, n_trials=10, soft_mask=True, tabu_size=3)
        t_tabu = time.time() - t0
        metrics['tabu']['swaps'] += res_tabu.swaps
        metrics['tabu']['time'] += t_tabu
        
        # BeamSearch -> BSx10, BeamWidth=3
        t0 = time.time()
        try:
            res_beam = compile_beam_search_multi_trial(qc, policy, cm, n_trials=10, beam_width=3)
            s_beam = res_beam.swaps
        except Exception as e:
            s_beam = -1
        t_beam = time.time() - t0
        metrics['beam']['swaps'] += s_beam
        metrics['beam']['time'] += t_beam
        
        row = (f"{cname:12s} | {sabre:7d} | "
               f"{res_naked.swaps:7d} ({t_naked:4.1f}s) | "
               f"{res_tabu.swaps:7d} ({t_tabu:4.1f}s) | "
               f"{s_beam:7d} ({t_beam:4.1f}s)")
        print(row)
        
    print("-" * 105)
    
    row_tot = (f"{'TOTAL':12s} | {metrics['sabre']['swaps']:7d} | "
           f"{metrics['naked']['swaps']:7d} ({metrics['naked']['time']:4.1f}s) | "
           f"{metrics['tabu']['swaps']:7d} ({metrics['tabu']['time']:4.1f}s) | "
           f"{metrics['beam']['swaps']:7d} ({metrics['beam']['time']:4.1f}s)")
    print(row_tot)
    print("=" * 105)

    print("\n[请更新至 benchmark_tracking.md]")
    tot_sabre = metrics['sabre']['swaps']
    tot_tabu = metrics['tabu']['swaps']
    ratio_tabu = tot_sabre / tot_tabu if tot_tabu > 0 else 0
    t_tabu_str = f"{metrics['tabu']['time']:.1f}s"
    
    tot_beam = metrics['beam']['swaps']
    ratio_beam = tot_sabre / tot_beam if tot_beam > 0 else 0
    t_beam_str = f"{metrics['beam']['time']:.1f}s"

    md = f"""| `$(date '+%m/%d')` | SOTA 对决: Tabu vs Beam | IBM Tokyo | QFT/QAOA/Grover | {tot_sabre} | Tabu:**{tot_tabu}** ({t_tabu_str}), Beam:**{tot_beam}** ({t_beam_str}) | Tabu:**{ratio_tabu:.2f}x**, Beam:**{ratio_beam:.2f}x** | 对决结束 |"""
    print(md)

if __name__ == "__main__":
    run_sota_ablation()
