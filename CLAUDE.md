# 量子电路编译器 (Quantum Circuit Compiler)

基于 PPO + 纯 PyTorch GNN 的量子电路 AI 编译与 SABRE 动态路由优化。

**GitHub**: https://github.com/qqyyqq812/ZJU-Quantum-Compiler

## 关键文件导航

| 文件/目录 | 用途 |
|----------|------|
| `src/compiler/` | 核心 RL 编译器代码（env.py, policy.py, train.py, gnn_*.py, pass_manager.py） |
| `src/benchmarks/` | 电路生成、拓扑定义、评测脚本 |
| `docs/technical/decisions.md` | **版本决策与踩坑**（V9→V14 全部决策）⭐ |
| `docs/technical/01_物理基础.md` | 量子电路基础知识 |
| `docs/technical/03_SABRE精读.md` | SABRE 算法细节 |
| `docs/technical/colab_workflow_and_pitfalls.md` | Colab 踩坑 |
| `notebooks/` | 4 个有序 Colab 训练笔记本（01-04） |
| `configs/` | 所有训练超参（yaml） |
| `models/v<N>_<topology>/` | 训练产出（checkpoint + history + eval_report） |
| `.claude/rules/` | **本项目定制 harness 规则**（覆盖 global rules）⭐ |

## 本地 harness 规则（必读）

本项目的 Claude Code 行为规则在 `.claude/rules/` 下，**覆盖全局规则**：

| 规则文件 | 范围 |
|---------|------|
| `doc-governance.md` | 禁止生成 handoff/V[n]改进/SOTA 等文档 |
| `experiment-log.md` | 每次训练必产出 history + log + eval_report |
| `code-and-config.md` | yaml 优先、零硬编码、禁止 torch_geometric |
| `workflow-agents.md` | 何时并行 subagent、何时触发 ECC skill |
| `deployment.md` | Colab/GPU 部署协议（Zero-Touch Remote） |
| `git-rules.md` | commit 频率、message 格式、禁止事项 |

任何代码改动前，先读对应规则。

## 当前状态 (2026-04-24)

- **最新版本**：V14（重构中）— 见 `docs/technical/decisions.md`
- **前一版本**：V13 在 GPU 上训练发散（Stage 1 卡死），根因见 decisions.md
- **V14 四大改动**：
  1. SABRE 基线缓存（吞吐 1.0→15 eps/s）
  2. 阶段化 Mask（早期 Hard，后期 Soft）
  3. 奖励分层（早期固定完成奖励，中后期 SABRE 相对）
  4. 修复 `pass_manager.py` 假集成
- **GPU**：RTX 5090 32GB (AutoDL) — 目前离线

## 快速开始

```bash
# 本地 smoke test（CPU, 1000 ep）
python -m src.compiler.train --config configs/v14_local_smoke.yaml

# GPU 训练（AutoDL）— 参见 .claude/rules/deployment.md
bash run_train_v14.sh

# 评测（对比 SABRE）
python -m src.benchmarks.evaluate --model models/v14_tokyo20/v7_ibm_tokyo_best.pt
```

## 开发硬规则（摘要，详细见 .claude/rules/）

1. **超参数**：**必须**在 `configs/*.yaml`，代码里零硬编码
2. **笔记本**：提交前清除所有单元输出（`nbconvert --ClearOutputPreprocessor`）
3. **远端代码**：**禁止**在 Colab/GPU 上直接修改 — 只允许 `git pull` + 运行
4. **文档**：**禁止**生成 `handoff_*.md` / `V[n]改进*.md` / `EXECUTION_PLAN_*.md`
5. **决策记录**：算法改动**必须**同步更新 `docs/technical/decisions.md`
6. **Commit**：每周至少 1 次，格式 `<type>: <desc> [<version>]`

## 评分对标

| 维度 | 占比 | 本项目应对 |
|------|------|---------|
| 项目周期管理 | 20% | Git 周活跃 + 版本化训练（see git-rules.md） |
| 工程规范 | 25% | `.claude/rules/` + yaml 配置 + 清晰目录 |
| 算法设计 | 30% | `src/compiler/` + `decisions.md`（V9→V14 演进） |
| 社区展示 | 15% | GitHub README + 演示 notebook |
| AI 协同 | 10% | `AI-Collaboration.md`（V13→V14 AI 协同日志） |

## GitHub

https://github.com/qqyyqq812/ZJU-Quantum-Compiler
