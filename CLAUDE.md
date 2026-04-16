# 量子电路 AI 编译器 (ZJU Quantum Compiler)

**GitHub**: https://github.com/qqyyqq812/ZJU-Quantum-Compiler

**Project Core**: 复杂拓扑结构下的量子电路人工智能编译与动态路由优化，基于 PPO/DQN 强化学习。

## 快速开始

```bash
cd /home/qq/projects/量子电路

# 安装依赖
pip install -r requirements.txt

# 查看 Notebooks
ls notebooks/

# 运行基准评估
bash run_teacher_eval.sh

# GPU 服务器训练
bash run_train_v13.sh
```

## 导航索引

- **技术文档**: `docs/technical/`（架构、SABRE 精读、深度技术反思）
- **任务面板**: `docs/task.md`
- **AI 协同日志**: `AI-Collaboration.md`
- **训练配置**: `configs/*.yaml`
- **Code Tour**: `.tours/` 目录
- **项目约束**: `.claude/rules/`（如有）

## 目录结构说明

| 目录/文件 | 说明 |
|-----------|------|
| `src/compiler/` | RL 训练主循环、环境、策略网络 |
| `src/benchmarks/` | SABRE 基准测试与评估 |
| `notebooks/` | Colab 训练脚本（01-04 有序编号） |
| `configs/` | 超参数配置文件（yaml） |
| `models/` | 训练产出模型和 checkpoint |
| `docs/technical/` | 技术文档 |
| `tests/` | 单元与集成测试 |

## 文件更新策略

| 文件 | 谁维护 | 何时更新 |
|------|--------|--------|
| CLAUDE.md | CC | 目录结构改变时 |
| docs/technical/ | CC | 架构/核心组件有显著改变时 |
| AI-Collaboration.md | CC | 每次 AI 协同操作后记录 |
| .tours/ | AG | 用户要求讲解时生成 |
| docs/task.md | User | 主动刷新任务清单 |

## 代码原则

- 超参数通过 `configs/*.yaml` 管理，禁止硬编码训练参数
- 笔记本文件名格式：`0N_description.ipynb`，提交前清除输出
- 路径使用 `os.path.abspath(__file__)` 相对定位，禁止 `/root/` 绝对路径
