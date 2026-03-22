# ZJU-Quantum-Compiler 🧬

> **量子电路 AI 编译与路由优化** — 用 GNN + 强化学习 (PPO) 替代传统启发式路由算法

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-62%20passed-green)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

量子计算中，逻辑电路必须映射到物理芯片上才能执行，但芯片的量子比特连接是受限的（Coupling Map）。**路由问题**就是在满足拓扑约束的前提下，用最少的 SWAP 操作完成映射——这是一个 NP-Hard 问题。

本项目探索用 **图注意力网络 (GAT) + 近端策略优化 (PPO)** 替代 Qiskit 默认的 SABRE 启发式算法，将路由决策建模为马尔可夫决策过程 (MDP)。

### 技术特色

- 🧠 **GAT 双图编码器**: 同时编码电路 DAG + 芯片 Coupling Map
- 🎮 **Gymnasium RL 环境**: 标准接口，State=映射+前沿, Action=SWAP
- 📊 **完整评估管线**: 4 种基准电路 × 7 种拓扑 × 多级优化
- 📈 **消融实验**: 超参数敏感性分析
- 🔌 **Qiskit 插件**: `compile_with_ai()` 一行调用

## 快速开始

```bash
# 环境配置
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 运行测试
PYTHONPATH=. pytest tests/ -v

# 训练 AI 路由器 (grid 拓扑, 5 qubit, 500 episodes)
export TORCH_DYNAMO_DISABLE=1
PYTHONPATH=. python -m src.compiler.train --topology grid_3x3 --qubits 5 --episodes 500

# 用 AI 路由器编译电路
PYTHONPATH=. python -c "
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from src.compiler.pass_manager import compile_with_ai

qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0, 4)  # 远程 CNOT，需要路由
cm = CouplingMap.from_line(5)
compiled = compile_with_ai(qc, cm, model_path='models/router_linear_5_5q.pt')
print(compiled)
"
```

## 项目结构

```
src/
├── benchmarks/
│   ├── circuits.py        # QFT, Grover, QAOA, Random 基准电路
│   ├── topologies.py      # IBM Eagle, Google Sycamore 等拓扑
│   ├── evaluate.py        # CompileResult 评估框架
│   └── run_baseline.py    # SABRE 基线运行器
├── compiler/
│   ├── dag.py             # 核心 DAG 操作 (前沿/执行/SWAP/GNN特征)
│   ├── gnn_encoder.py     # GAT 双图编码器
│   ├── env.py             # Gymnasium RL 环境
│   ├── policy.py          # PPO Actor-Critic + 训练器
│   ├── train.py           # 训练管线 + CLI
│   └── pass_manager.py    # Qiskit AIRouter 集成
├── evaluation/            # 评估管线
└── visualization/         # 结果可视化
tests/                     # 62 项自动化测试
docs/                      # 技术文档 + AI 协同日志
results/                   # 基线数据 + 对比结果 + 图表
models/                    # 训练好的路由器模型
```

## 实验结果

### SABRE 基线 (60 次编译)
- 平均 CX 开销: **123.5%**
- 平均深度开销: **328%**

### 消融实验 (grid_3x3, 5 qubit)

| 配置 | Avg Reward ↑ | Avg SWAPs ↓ |
|------|------------|-------------|
| baseline (100ep) | 29.1 | 33.5 |
| higher_penalty | 20.9 | 36.7 |
| lower_lr | 26.9 | 40.9 |
| **more_training (200ep)** | **30.3** | **26.9** |

### AI vs SABRE

当前 AI 路由器（仅 300 episodes 训练）尚未超越 SABRE。这是预期结果：
- SABRE 是经过多年优化的生产算法
- PPO 需要 10,000+ episodes 才能在此类组合优化问题上收敛
- **关键贡献**: 完整的 GNN+RL 框架已搭建，为后续大规模训练和改进提供了基础

## 技术栈

| 类别 | 工具 | 用途 |
|------|------|------|
| 量子 | Qiskit 2.3 | 电路创建 + Transpiler |
| 图学习 | PyTorch Geometric 2.7 | GAT 实现 |
| RL | Gymnasium + PPO | 路由策略学习 |
| 训练 | PyTorch 2.10 | 深度学习引擎 |
| 测试 | pytest | 62 项自动化测试 |

## AI 协同开发说明

本项目采用 AI 辅助开发模式，详见 [AI 协同日志](docs/03_AI协同日志.md)。

## License

MIT
