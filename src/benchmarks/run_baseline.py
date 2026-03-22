"""
SABRE 基线运行器
================
在全套基准电路×拓扑组合上运行 Qiskit SABRE，建立性能基线。

输出: results/baseline_sabre.json
"""

from __future__ import annotations

from pathlib import Path

from src.benchmarks.circuits import get_benchmark_suite
from src.benchmarks.topologies import get_standard_topologies, get_topology_info
from src.benchmarks.evaluate import (
    evaluate_compiler, compare_compilers, save_results, CompileResult
)


def run_sabre_baseline(
    qubit_range: list[int] | None = None,
    topology_names: list[str] | None = None,
    optimization_levels: list[int] | None = None,
) -> list[CompileResult]:
    """运行 SABRE 基线。

    Args:
        qubit_range: 比特数列表，默认 [5, 10, 15]
        topology_names: 拓扑名称列表，默认使用标准集
        optimization_levels: Qiskit 优化级别列表，默认 [1]
    Returns:
        CompileResult 列表
    """
    if qubit_range is None:
        qubit_range = [5, 10, 15]
    if optimization_levels is None:
        optimization_levels = [1]

    suite = get_benchmark_suite(qubit_range)

    if topology_names is None:
        topologies = get_standard_topologies()
    else:
        from src.benchmarks.topologies import get_topology
        topologies = {name: get_topology(name) for name in topology_names}

    results: list[CompileResult] = []

    for bench in suite:
        circuit = bench['circuit']
        n_q = bench['n_qubits']

        for topo_name, cm in topologies.items():
            # 跳过比特数不够的拓扑
            if cm.size() < n_q:
                continue

            for opt_level in optimization_levels:
                try:
                    result = evaluate_compiler(
                        circuit=circuit,
                        coupling_map=cm,
                        circuit_name=bench['name'],
                        topology_name=topo_name,
                        compiler_name=f"sabre_O{opt_level}",
                        optimization_level=opt_level,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"⚠️ 跳过 {bench['name']} @ {topo_name} O{opt_level}: {e}")

    return results


def generate_report(results: list[CompileResult]) -> str:
    """生成 Markdown 格式的基线报告。"""
    report = ["# SABRE 基线性能报告\n"]
    report.append(compare_compilers(results))
    report.append(f"\n\n**总计**: {len(results)} 次编译")

    # 汇总统计
    if results:
        avg_cx_overhead = sum(r.cx_overhead for r in results) / len(results)
        avg_depth_overhead = sum(r.depth_overhead for r in results) / len(results)
        avg_time = sum(r.compile_time_ms for r in results) / len(results)
        report.append(f"\n**平均 CX 开销比**: {avg_cx_overhead:.2%}")
        report.append(f"**平均深度开销比**: {avg_depth_overhead:.2%}")
        report.append(f"**平均编译时间**: {avg_time:.1f} ms")

    return "\n".join(report)


def main():
    """运行完整基线并保存结果。"""
    print("🚀 运行 SABRE 基线...")
    results = run_sabre_baseline(
        qubit_range=[5, 10, 15],
        optimization_levels=[1],
    )

    # 保存结果
    output_path = Path(__file__).parent.parent.parent / "results" / "baseline_sabre.json"
    save_results(results, str(output_path))
    print(f"✅ 结果已保存: {output_path}")

    # 生成报告
    report = generate_report(results)
    print(report)

    return results


if __name__ == '__main__':
    main()
