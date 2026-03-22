"""
基准量子电路集
==============
生成标准量子算法电路，供编译器基准测试使用。

支持的电路类型:
- QFT (Quantum Fourier Transform): 密集 CNOT，层次化结构
- Grover: 重复迭代结构
- QAOA: 参数化变分电路
- Random: 随机电路（最坏情况压力测试）
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full


def generate_qft(n_qubits: int) -> QuantumCircuit:
    """生成 QFT (量子傅里叶变换) 电路。

    QFT 是量子计算中最基础的子程序，包含大量受控旋转门和 CNOT。
    特点：门数 O(n²)，CNOT 连接跨越多个比特 → SWAP 需求大。

    Args:
        n_qubits: 量子比特数 (建议 5-20)
    Returns:
        QFT 量子电路 (已分解为基础门)
    """
    qc = synth_qft_full(n_qubits)
    qc.name = f"qft_{n_qubits}"
    return qc


def generate_grover(n_qubits: int, num_iterations: int = 1) -> QuantumCircuit:
    """生成 Grover 搜索算法电路。

    Grover 算法通过 Oracle + Diffusion 的迭代结构实现二次加速搜索。
    特点：重复结构，适合测试路由器对迭代电路的优化能力。

    Args:
        n_qubits: 量子比特数 (建议 5-15, 不含辅助比特)
        num_iterations: Grover 迭代次数
    Returns:
        Grover 风格量子电路
    """
    qc = QuantumCircuit(n_qubits, name=f"grover_{n_qubits}")

    # 初始叠加
    qc.h(range(n_qubits))

    for _ in range(num_iterations):
        # Oracle: 标记目标态 (简化版: 多控 Z 门)
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        # 多控 CNOT 链
        for i in range(n_qubits - 1):
            qc.cx(i, n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))

        # Diffusion: 2|s⟩⟨s| - I
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        for i in range(n_qubits - 1):
            qc.cx(i, n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    return qc


def generate_qaoa(n_qubits: int, p: int = 1) -> QuantumCircuit:
    """生成 QAOA (量子近似优化算法) 电路。

    QAOA 是变分量子算法的代表，用于组合优化。
    特点：参数化门 + 图相关的 ZZ 交互 → 拓扑敏感。

    Args:
        n_qubits: 量子比特数
        p: QAOA 层数 (depth parameter)
    Returns:
        QAOA 量子电路
    """
    qc = QuantumCircuit(n_qubits, name=f"qaoa_{n_qubits}_p{p}")

    # 初始叠加
    qc.h(range(n_qubits))

    rng = np.random.RandomState(42)
    for layer in range(p):
        gamma = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(0, np.pi)

        # Problem unitary: ZZ 交互（环形图）
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)

        # Mixer unitary: RX 旋转
        qc.rx(beta, range(n_qubits))

    return qc


def generate_random(n_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """生成随机量子电路。

    无结构的随机电路作为最坏情况压力测试。
    特点：CNOT 连接随机，无规律可利用。

    Args:
        n_qubits: 量子比特数
        depth: 电路层数 (每层包含若干随机门)
        seed: 随机种子 (保证可复现)
    Returns:
        随机量子电路
    """
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits, name=f"random_{n_qubits}_d{depth}")

    single_gates = ['h', 'x', 'rz', 'sx']

    for _ in range(depth):
        # 随机单比特门
        for q in range(n_qubits):
            gate = rng.choice(single_gates)
            if gate == 'rz':
                qc.rz(rng.uniform(0, 2 * np.pi), q)
            else:
                getattr(qc, gate)(q)

        # 随机 CNOT (约 n/2 个)
        available = list(range(n_qubits))
        rng.shuffle(available)
        for i in range(0, len(available) - 1, 2):
            qc.cx(available[i], available[i + 1])

    return qc


def get_benchmark_suite(qubit_range: list[int] | None = None) -> list[dict]:
    """生成完整的基准电路套件。

    Args:
        qubit_range: 要测试的比特数列表，默认 [5, 10, 15, 20]
    Returns:
        [{"name": "qft_5", "circuit": QuantumCircuit, "n_qubits": 5, "type": "qft"}, ...]
    """
    if qubit_range is None:
        qubit_range = [5, 10, 15, 20]

    suite = []
    for n in qubit_range:
        suite.append({
            "name": f"qft_{n}", "circuit": generate_qft(n),
            "n_qubits": n, "type": "qft"
        })
        suite.append({
            "name": f"grover_{n}", "circuit": generate_grover(n),
            "n_qubits": n, "type": "grover"
        })
        suite.append({
            "name": f"qaoa_{n}", "circuit": generate_qaoa(n),
            "n_qubits": n, "type": "qaoa"
        })
        suite.append({
            "name": f"random_{n}", "circuit": generate_random(n, depth=n),
            "n_qubits": n, "type": "random"
        })

    return suite
