# ZJU-Quantum-Compiler 🧬

> **量子电路 AI 编译与路由优化** — 用 GNN + 强化学习 (PPO) 替代 SABRE 启发式路由算法

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-62%20passed-green)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

量子计算中，逻辑电路必须映射到物理芯片上执行，但芯片的量子比特连接是受限的（Coupling Map）。**路由问题**是在拓扑约束下用最少 SWAP 操作完成映射——NP-Hard 问题。

本项目用 **GraphSAGE + PPO** 将路由决策建模为 MDP，训练端到端 AI 路由器，对标工业标准 SABRE 算法。

### 核心技术特色

- 🧬 **V9 拓扑无关编码**: 291 维固定观测向量，5Q/20Q 共用同一模型
- 🧠 **GNN Edge-Scoring**: GraphSAGE 编码芯片图 + Attention 为每条 SWAP 边独立打分
- 📈 **5 阶课程学习**: 3Q warm-up → 5Q → 10Q → 20Q master 自动升阶
- ⚡ **MTx100 推断引擎**: 100 次随机初始映射贪心采样，零拷贝环境优化
- 🏭 **IBM Tokyo 20Q**: 真实芯片拓扑 (43 条双向边, 直径 4)

## 已安装的扩展模块
- ✅ **[/supervised-dev]** — 用户强制监督、量化防爆与可视化验收工作流（已配置 `docs/qa_supervision` 与打分追溯板）

## 快速开始

```bash
# 1. 环境配置
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 运行测试 (62 项)
PYTHONPATH=. pytest tests/ -v

# 3. 用 AI 路由器编译电路 (使用已训练的 V7.2 模型)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DYNAMO_DISABLE=1
PYTHONPATH=. python -c "
from src.benchmarks.circuits import generate_qft
from src.benchmarks.topologies import get_topology
from src.compiler.inference_v8 import load_policy, compile_multi_trial

# 加载 V7.2 最佳模型
policy = load_policy('models/v7_final_v2/v7_linear_5_best.pt', 'linear_5')
cm = get_topology('linear_5')

# 编译 5-qubit QFT 电路
qc = generate_qft(5)
result = compile_multi_trial(qc, policy, cm, n_trials=100)
print(f'QFT-5 路由结果: {result.swaps} SWAPs, 完成={result.completed}')
"

# 4. 运行推断策略消融实验
PYTHONPATH=. python -m src.compiler.inference_v8 models/v7_final_v2/v7_linear_5_best.pt linear_5

# 5. 快速训练体验 (5Q, 500 episodes, ~30秒)
PYTHONPATH=. python -m src.compiler.train --topology linear_5 --qubits 5 --episodes 500 --curriculum --save-dir models/quick_test

# 6. 启动 V9 20Q 完整训练
bash scripts/run_train_v9_20Q.sh          # 全新训练
bash scripts/run_train_v9_20Q.sh resume   # 从 checkpoint 恢复
```

## 实验结果

### V7.2 (5Q linear, 50000 episodes)

| 指标 | 数值 |
|------|------|
| 评估 avg_swap | **9.3** |
| 完成率 | **100%** |
| SABRE 基线 | ~10 |
| 结论 | **与 SABRE 持平** |

### V8 (5Q linear, 随机初始映射)

| 指标 | 数值 |
|------|------|
| 评估 avg_swap | **13.0** (随机映射条件) |
| 完成率 | **100%** |
| Early Stop | @ep17000 (26.9 分钟) |
| 改进 | 随机映射增强泛化性，奖励函数调优 |

### V9 (20Q IBM Tokyo, 进行中)

| 指标 | 数值 |
|------|------|
| 课程升阶 | 3Q(ep300) → 5Q(ep600) → 10Q(ep900) → 20Q(ep3000) |
| 当前 SWAP | ~87-107 (训练中) |
| 评估 done | 33% (step 预算已从 500→2000 修复) |
| 状态 | 训练中断 @ep7400，待恢复 |

## 项目结构

```
src/
├── compiler/
│   ├── env.py             # V9 拓扑无关 RL 环境 (291维边驱动编码)
│   ├── policy.py          # GNN Actor-Critic (GraphSAGE + Edge-Scoring)
│   ├── train.py           # PPO 训练管线 + 课程学习 + checkpoint
│   ├── curriculum.py      # 5 阶课程学习调度器
│   ├── inference_v8.py    # V8 推断引擎 (MTx100/BiDir/BeamSearch)
│   ├── light_env.py       # 纯 Numpy 零拷贝环境
│   ├── dag.py             # 核心 DAG (前沿/执行/SWAP)
│   ├── gnn_encoder.py     # 3 层 GraphSAGE 编码器
│   ├── gnn_extractor.py   # 物理图特征提取
│   └── pass_manager.py    # Qiskit AIRouter 集成
├── benchmarks/
│   ├── circuits.py        # QFT, Grover, QAOA, Random 基准电路
│   ├── topologies.py      # 拓扑库 (含 IBM Tokyo 20Q)
│   ├── evaluate.py        # CompileResult 评估框架
│   └── run_baseline.py    # SABRE 基线运行器
└── server/
    └── app.py             # FastAPI 后端 (MTx100 推断)
models/
├── v7_final_v2/           # V7.2 best (5Q, avg_swap=9.3)
├── v8_rewarded/           # V8 best (5Q 随机映射)
└── v9_tokyo20/            # V9 best (20Q IBM Tokyo)
tests/                     # 62 项自动化测试
docs/                      # 项目报告 + 文献综述 + 技术文档
results/                   # 评估结果 + 训练曲线
scripts/                   # 训练启动脚本
```

## 技术栈

| 类别 | 工具 | 用途 |
|------|------|------|
| 量子 | Qiskit 2.3 | 电路创建 + Transpiler |
| 图学习 | PyTorch Geometric 2.7 | GraphSAGE 实现 |
| RL | Gymnasium + PPO | 路由策略学习 |
| 训练 | PyTorch 2.10 | 深度学习引擎 |
| 后端 | FastAPI | 推断 API 服务 |
| 测试 | pytest | 62 项自动化测试 |

## 详细文档

- [项目报告](docs/项目报告.md) — 完整技术方案 + 训练结果 + 11 篇参考文献
- [文献综述](docs/technical/文献综述.md) — 核心论文分析
- [开发日志](docs/03_AI协同日志.md) — 项目开发过程记录

## License

MIT
