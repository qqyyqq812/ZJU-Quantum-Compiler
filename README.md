# ZJU-Quantum-Compiler 🧬

> 复杂拓扑结构下的量子电路人工智能编译与动态路由优化

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## 项目简介

物理量子芯片上的量子比特存在严格的连通性限制——并非所有比特之间都能直接执行双比特门。为满足这一硬件约束，编译器必须通过插入 SWAP 门来移动量子态，而如何最小化这些额外操作（SWAP/CNOT 门数、电路深度）是一个 **NP-hard** 优化问题。

本项目利用**图神经网络 (GNN)** 与**强化学习 (RL)** 技术，开发一个智能化的量子电路编译器插件，学习目标物理拓扑的约束特征，动态决策最优的比特映射与 SWAP 门插入策略。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行基准测试
python -m src.benchmarks.evaluate

# 运行测试
pytest tests/ -v
```

## 项目结构

```
.
├── docs/                    # 文档
│   ├── technical/           # 技术文档（长效知识）
│   └── 03_AI协同日志.md     # AI 交互记录
├── src/                     # 核心源码
│   ├── compiler/            # 编译器核心算法
│   ├── benchmarks/          # 基准测试
│   └── visualization/       # 可视化
├── sandbox/                 # 探索与试错区
├── tests/                   # 自动化测试
└── workspace_rules.md       # 工作规约
```

## 技术栈

- **量子框架**: Qiskit
- **深度学习**: PyTorch + PyTorch Geometric (GNN)
- **强化学习**: Stable-Baselines3 / 自定义 PPO
- **可视化**: Matplotlib + NetworkX

## 评估指标

| 指标 | 说明 |
|------|------|
| 电路深度 (Depth) | 编译后电路的最大门层数 |
| CNOT 门数 | 编译后的双比特门总数 |
| 额外 SWAP 数 | 插入的 SWAP 门数量 |
| 编译时间 | 编译器运行耗时 |

## 对标基准

- Qiskit SABRE 算法（当前业界默认编译器）
- Qiskit Stochastic SWAP
- tket 编译器

## 许可证

MIT License

## 致谢

浙江大学 量子信息课程大作业 · 课题四
