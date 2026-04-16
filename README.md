# ZJU 量子电路 AI 编译器

> 课题四：复杂拓扑结构下的量子电路人工智能编译与动态路由优化

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 本地运行（Colab）：
   - 打开 `notebooks/03_train_v10_ppo_hardmask.ipynb`
   - 在 Google Colab 中运行

3. 服务器评估：
   ```bash
   bash run_teacher_eval.sh
   ```

## 项目结构

```
.
├── notebooks/          # Jupyter 训练脚本（Colab 环境）
│   ├── 01_train_v9_ppo_baseline.ipynb
│   ├── 02_train_v9_ppo_fallback.ipynb
│   ├── 03_train_v10_ppo_hardmask.ipynb
│   └── 04_train_v11_dqn.ipynb
├── configs/            # 训练超参数配置
│   ├── v9_baseline.yaml
│   ├── v9_fallback.yaml
│   └── v13_gpu.yaml
├── src/                # 核心源码
│   ├── compiler/       # RL 训练、环境、策略网络
│   ├── benchmarks/     # 基准测试与评估
│   ├── visualization/  # 结果可视化
│   └── evaluation/     # 性能对比
├── docs/               # 技术文档和论文参考
│   └── technical/      # 架构设计、SABRE 精读等
├── models/             # 训练产出模型
├── tests/              # 单元与集成测试
└── run_train_v13.sh    # GPU 服务器一键训练
```

## 评分标准对标

| 维度 | 占比 | 支撑材料 |
|------|------|---------|
| 项目周期管理 | 20% | Commit history（git log） |
| 工程规范 | 25% | README + CLAUDE.md + docs/technical/ |
| 算法设计 | 30% | src/compiler/ + docs/technical/SABRE精读.md |
| 社区展示 | 15% | GitHub Stars / README |
| AI 协同 | 10% | AI-Collaboration.md |

## 算法版本历程

| 版本 | 算法 | 特点 |
|------|------|------|
| V9 | PPO | 基线，20Q IBM Tokyo，硬掩码 |
| V10 | PPO | Hard Mask 强化，消除软约束漏洞 |
| V11 | DQN | 对比实验，验证 PPO 优越性 |
| V13 | PPO + GNN | SABRE 相对奖励 + 9 维 GNN 编码，GPU 大批量 |

## GitHub

[https://github.com/qqyyqq812/ZJU-Quantum-Compiler](https://github.com/qqyyqq812/ZJU-Quantum-Compiler)
