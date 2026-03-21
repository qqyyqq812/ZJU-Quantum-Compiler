"""
M1.1 Qiskit 入门：Bell 态电路创建与模拟
==========================================
学习目标：
1. 创建量子电路
2. 添加量子门（H + CNOT）
3. 模拟运行并查看结果

Bell 态是最简单的量子纠缠态：|Φ+⟩ = (|00⟩ + |11⟩) / √2
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import json


def create_bell_state() -> QuantumCircuit:
    """创建 Bell 态电路：H门 + CNOT门"""
    qc = QuantumCircuit(2, 2)  # 2 量子比特, 2 经典比特
    qc.h(0)       # Hadamard 门：|0⟩ → (|0⟩+|1⟩)/√2
    qc.cx(0, 1)   # CNOT 门：控制位0, 目标位1
    qc.measure([0, 1], [0, 1])  # 测量
    return qc


def simulate_circuit(qc: QuantumCircuit, shots: int = 1024) -> dict:
    """用 Aer 模拟器运行电路"""
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    return counts


def main():
    # 1. 创建 Bell 态电路
    qc = create_bell_state()
    print("=== Bell 态电路 ===")
    print(qc.draw(output='text'))
    print()

    # 2. 模拟运行
    counts = simulate_circuit(qc, shots=4096)
    print("=== 测量结果 (4096 shots) ===")
    print(json.dumps(counts, indent=2))
    print()

    # 3. 验证：应该只有 |00⟩ 和 |11⟩，概率各约 50%
    total = sum(counts.values())
    for state, count in sorted(counts.items()):
        pct = count / total * 100
        print(f"  |{state}⟩: {count}/{total} = {pct:.1f}%")

    # 4. 断言检查
    assert set(counts.keys()) <= {'00', '11'}, "Bell 态应只包含 |00⟩ 和 |11⟩"
    for state in counts:
        assert 40 < counts[state] / total * 100 < 60, f"{state} 概率应接近 50%"

    print("\n✅ Bell 态验证通过！量子纠缠工作正常。")
    return counts


if __name__ == '__main__':
    main()
