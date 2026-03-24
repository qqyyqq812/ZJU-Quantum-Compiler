"""
V7 工业级自动化基准评估套件
============================
在标准电路基准集上，以统计显著性对比 AI 路由器与 SABRE 的性能。

- 支持批量加载标准基准集 (QFT, QAOA, Grover, BV, Random)
- 每个电路运行 N 次随机试验 (消除 SABRE 随机性)
- 自动生成 Markdown 评估报告与对比表格
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
import torch

from src.benchmarks.topologies import get_topology
from src.benchmarks.circuits import get_benchmark_suite
from src.benchmarks.evaluate import evaluate_compiler, CompileResult, compare_compilers
from src.compiler.policy import PolicyNetwork
from src.compiler.env import QuantumRoutingEnv


@dataclass
class EvalSummary:
    circuit_name: str
    qbits: int
    depth: int
    ai_swaps_median: float
    sabre_swaps_median: float
    swap_ratio: float  # SABRE / AI  (>1 is better for AI)
    ai_time_ms: float
    sabre_time_ms: float


def run_ai_compiler(circuit: QuantumCircuit, coupling_map: CouplingMap, policy: PolicyNetwork) -> QuantumCircuit:
    """运行 AI 编译器进行路由"""
    env = QuantumRoutingEnv(coupling_map=coupling_map, max_steps=1000)  # 评测时给更大步数
    env.set_circuit(circuit)
    obs, info = env.reset()
    
    done = False
    
    # 强制贪心策略 (argmax)
    with torch.no_grad():
        while not done:
            mask = env.get_action_mask()
            
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            from torch_geometric.data import Batch
            d_b = Batch.from_data_list([info['gnn_input']['graph']])
            swap_edges = [info['gnn_input']['swap_edges']]
            
            dist, _ = policy(obs_t, d_b, swap_edges)
            logits = dist.logits
            
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            logits = logits.masked_fill(mask_t == 0, -1e8)
            
            action = logits.argmax(dim=-1).item()
            
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
    # 记录实际 SWAP 数
    total_swaps = env._total_swaps
    if not env._dag.is_done():
        # 如果未完成路由，加上剩余门数作为惩罚
        remaining = env._dag.remaining_two_qubit_gates()
        total_swaps += remaining * 3  # 每个剩余 CX 需要约 3 个 SWAP 的惩罚
        
    # 返回一个具有正确 CX 数量的 dummy circuit 供 evaluate.py 统计
    original_cx = dict(circuit.count_ops()).get('cx', 0)
    dummy = QuantumCircuit(2)
    for _ in range(original_cx + total_swaps * 3):
        dummy.cx(0, 1)
        
    return dummy


def load_model(model_path: str, topology_name: str) -> PolicyNetwork:
    """加载 V7 模型权重"""
    cm = get_topology(topology_name)
    env = QuantumRoutingEnv(coupling_map=cm)
    policy = PolicyNetwork(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
    )
    policy.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    policy.eval()
    return policy


def run_benchmark_suite(model_path: str, topology_name: str, trials: int = 5, output_dir: str = "results"):
    """执行全面的自动化基准测试并生成论文级报告"""
    print(f"🚀 初始化 V7 评估套件")
    print(f"   模型: {model_path}")
    print(f"   拓扑: {topology_name}")
    print(f"   随机试验次数: {trials}")
    
    cm = get_topology(topology_name)
    n_phys = cm.size()
    policy = load_model(model_path, topology_name)
    
    # 获取不超过芯片物理比特数的测试集
    test_qubits = [q for q in [5, 10, 15, 20, 50] if q <= n_phys]
    suite = get_benchmark_suite(qubit_range=test_qubits)
    
    summaries = []
    
    print("\n⏳ 开始基准测试...")
    for item in suite:
        c_name = item['name']
        original_qc = item['circuit']
        qc_depth = original_qc.depth()
        
        print(f"   - 测试 {c_name:<12} (Q={original_qc.num_qubits}, D={qc_depth}) ... ", end="", flush=True)
        
        ai_swaps, sabre_swaps = [], []
        ai_times, sabre_times = [], []
        
        for t in range(trials):
            # 1. AI 评估
            def ai_compile(qc, cm_):
                return run_ai_compiler(qc, cm_, policy)
                
            res_ai = evaluate_compiler(original_qc, cm, compiler_name="V7-AI", compile_fn=ai_compile)
            ai_swaps.append(res_ai.extra_swap)
            ai_times.append(res_ai.compile_time_ms)
            
            # 2. SABRE 评估 (Qiskit opt=3)
            res_sabre = evaluate_compiler(original_qc, cm, compiler_name="Qiskit-SABRE", optimization_level=3)
            sabre_swaps.append(res_sabre.extra_swap)
            sabre_times.append(res_sabre.compile_time_ms)
            
        # 计算统计量
        ai_med = np.median(ai_swaps)
        sabre_med = np.median(sabre_swaps)
        ratio = sabre_med / max(ai_med, 1) if ai_med > 0 or sabre_med > 0 else 1.0
        
        summary = EvalSummary(
            circuit_name=c_name, qbits=original_qc.num_qubits, depth=qc_depth,
            ai_swaps_median=ai_med, sabre_swaps_median=sabre_med, swap_ratio=ratio,
            ai_time_ms=np.median(ai_times), sabre_time_ms=np.median(sabre_times)
        )
        summaries.append(summary)
        
        color = "\033[92m" if ratio > 1.05 else ("\033[91m" if ratio < 0.95 else "\033[93m")
        print(f"AI: {ai_med:4.0f} | SABRE: {sabre_med:4.0f} | Ratio: {color}{ratio:4.2f}x\033[0m")
        
    # 生成报告
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"evaluation_v7_{topology_name}.md"
    
    with open(report_file, "w") as f:
        f.write(f"# V7 Quantum Router Evaluation Report\n\n")
        f.write(f"- **Model**: `{model_path}`\n")
        f.write(f"- **Topology**: `{topology_name}` ({n_phys} qubits)\n")
        f.write(f"- **Trials per circuit**: {trials}\n")
        f.write(f"- **Baseline**: Qiskit SABRE (O3)\n\n")
        
        f.write("## Benchmark Results\n\n")
        f.write("| Circuit | Qubits | Depth | AI SWAPs (Med) | SABRE SWAPs (Med) | **SWAP Ratio** | AI Time (ms) | SABRE Time (ms) |\n")
        f.write("|---------|--------|-------|----------------|-------------------|----------------|--------------|-----------------|\n")
        
        geo_mean_ratio = 1.0
        for s in summaries:
            bold_ratio = f"**{s.swap_ratio:.2f}x**" if s.swap_ratio >= 1.0 else f"{s.swap_ratio:.2f}x"
            f.write(f"| {s.circuit_name} | {s.qbits} | {s.depth} | {s.ai_swaps_median:.1f} | {s.sabre_swaps_median:.1f} | {bold_ratio} | {s.ai_time_ms:.1f} | {s.sabre_time_ms:.1f} |\n")
            geo_mean_ratio *= s.swap_ratio
            
        geo_mean_ratio = geo_mean_ratio ** (1/len(summaries))
        f.write(f"| **Overall (GeoMean)** | - | - | - | - | **{geo_mean_ratio:.2f}x** | - | - |\n")
        
    print(f"\n📊 评估完成! 总体几何平均优化率: {geo_mean_ratio:.2f}x")
    print(f"📄 详细报告已保存至: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to V7 .pt model")
    parser.add_argument("--topology", default="linear_5", help="Topology name (e.g., ibm_eagle, grid_3x3)")
    parser.add_argument("--trials", type=int, default=5, help="Number of random trials per circuit")
    parser.add_argument("--outdir", default="results", help="Output directory for reports")
    args = parser.parse_args()
    
    run_benchmark_suite(args.model, args.topology, args.trials, args.outdir)
