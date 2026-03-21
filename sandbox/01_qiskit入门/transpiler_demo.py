"""
M1.4 Qiskit Transpiler 演示：观察 SABRE 的实际效果
===================================================
学习目标：
1. 理解 Transpiler 的作用
2. 观察 SABRE 如何插入 SWAP 门
3. 比较编译前后电路的差异
"""

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import json


def create_test_circuit(n_qubits: int = 5) -> QuantumCircuit:
    """创建一个需要路由的测试电路：非相邻比特间的 CNOT"""
    qc = QuantumCircuit(n_qubits)
    # 这些 CNOT 门连接的比特在线性拓扑上不一定相邻
    qc.h(0)
    qc.cx(0, 4)   # q0 和 q4 在线性拓扑上距离=4，需要 SWAP
    qc.cx(1, 3)   # q1 和 q3 距离=2，需要 SWAP
    qc.cx(2, 4)   # q2 和 q4 距离=2，需要 SWAP
    qc.cx(0, 3)   # q0 和 q3 距离=3，需要 SWAP
    qc.h(range(n_qubits))
    return qc


def compile_and_compare(qc: QuantumCircuit, coupling_map: CouplingMap,
                        opt_level: int = 1) -> dict:
    """编译电路并比较编译前后的差异"""
    pm = generate_preset_pass_manager(
        optimization_level=opt_level,
        coupling_map=coupling_map,
        basis_gates=['cx', 'id', 'rz', 'sx', 'x']
    )
    compiled = pm.run(qc)

    # 统计门数
    original_ops = dict(qc.count_ops())
    compiled_ops = dict(compiled.count_ops())

    return {
        'original': {
            'depth': qc.depth(),
            'ops': original_ops,
            'cx_count': original_ops.get('cx', 0),
        },
        'compiled': {
            'depth': compiled.depth(),
            'ops': compiled_ops,
            'cx_count': compiled_ops.get('cx', 0),
        },
        'overhead': {
            'extra_cx': compiled_ops.get('cx', 0) - original_ops.get('cx', 0),
            'depth_increase': compiled.depth() - qc.depth(),
        },
        'compiled_circuit': compiled,
    }


def main():
    qc = create_test_circuit(5)
    print("=== 原始逻辑电路 ===")
    print(qc.draw(output='text'))
    print()

    # 在不同拓扑上编译
    topologies = {
        '线性 (5q)': CouplingMap.from_line(5),
        '环形 (5q)': CouplingMap.from_ring(5),
        '全连接 (5q)': CouplingMap.from_full(5),
    }

    print("=" * 70)
    print(f"{'拓扑':<15} {'原始CX':>8} {'编译后CX':>10} {'额外CX':>8} {'原始深度':>8} {'编译深度':>8}")
    print("-" * 70)

    for name, cm in topologies.items():
        result = compile_and_compare(qc, cm)
        r = result
        print(f"{name:<15} "
              f"{r['original']['cx_count']:>8} "
              f"{r['compiled']['cx_count']:>10} "
              f"{r['overhead']['extra_cx']:>8} "
              f"{r['original']['depth']:>8} "
              f"{r['compiled']['depth']:>8}")

    # 展示线性拓扑下编译后的具体电路
    print("\n=== 线性拓扑编译后电路 (观察 SWAP 插入) ===")
    result_linear = compile_and_compare(qc, topologies['线性 (5q)'])
    print(result_linear['compiled_circuit'].draw(output='text', fold=120))

    print("\n=== 全连接拓扑编译后电路 (无需 SWAP) ===")
    result_full = compile_and_compare(qc, topologies['全连接 (5q)'])
    print(result_full['compiled_circuit'].draw(output='text', fold=120))

    print("\n✅ Transpiler 演示完成！")
    print("关键洞察：拓扑约束越强（线性）→ 需要越多 SWAP → 电路越深 → 噪声越大")


if __name__ == '__main__':
    main()
