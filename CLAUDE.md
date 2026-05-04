# 量子电路编译器 (Quantum Circuit Compiler)

基于 PPO + 纯 PyTorch GNN 的量子电路 AI 编译与 SABRE 动态路由优化。

**GitHub**: https://github.com/qqyyqq812/ZJU-Quantum-Compiler

## claude-mem 启动纪律

- 本项目**单独打开为 VSCode/Antigravity 工作区根**（File → Open Folder → `projects/量子电路/`），不要从 `/home/qq/` 根启动会话。
- claude-mem 按 `basename(cwd)` 自动注入本项目专属历史；从根启动会污染 `qq` 桶。
- 自检：会话第一条工具调用前确认 `process.cwd()` 是 `/home/qq/projects/量子电路` 而非 `/home/qq/`。
- 跨桶查（罕见）：`mcp-search query="..." project=null`，不用切窗口。

## 关键文件导航

| 文件/目录 | 用途 |
|----------|------|
| `src/compiler/` | 核心 RL 编译器代码（env.py, policy.py, train.py, gnn_*.py, pass_manager.py） |
| `src/benchmarks/` | 电路生成、拓扑定义、评测脚本 |
| `docs/technical/decisions.md` | **版本决策与踩坑**（V9→V15 全部决策）⭐ |
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

## 当前状态 (2026-05-04)

- **最新版本**：V15.2 代码存在，但训练未达标，当前暂停继续训练。
- **P1 评测**：`models/v14_tokyo20/eval_report_mqt.md` 已补
  `checkpoint_ep25333.pt` 的 AI 列；12 条主集合电路 AI 只完成 4 条，
  完成且可比的 AI/SABRE 平均为 2.500。
- **关键结论**：ep25333 不能支持“5Q backbone 稳定可用”的 warmstart 假设。
- **V14 四大改动**（已实装）：
  1. ✅ SABRE 基线缓存（`src/compiler/sabre_cache.py`）— 吞吐预期 1.0→15 eps/s
  2. ✅ 阶段化 Mask（`env.py::get_action_mask` 读 `_curriculum_stage`）
  3. ✅ 奖励分层（`env.py::step` terminal 根据 stage 切换）
  4. ✅ pass_manager 真集成（`pass_manager.py::_build_routed_circuit` 直接发 SwapGate）
- **V15 问题**：self-play 基本串行，`num_workers`/`num_inference_workers`
  配置未真正接入；后续必须先修 batch inference/并行 self-play，而不是继续调 yaml。
- **云端**：只读健康检查已完成；远端 `eb55a96`，GPU idle，无 V14/V15 训练进程。
- **Golden Pool**：`/home/qq/docs/GitHub_Golden_Pool/量子电路/` 是历史参考池，
  不直接覆盖当前仓库。

## V14 运行流程

```bash
# 1. 本地 smoke（CPU，1000 ep，3-5 分钟）
python -m src.compiler.train --config configs/v14_local_smoke.yaml

# 2. P1 评测（当前主线）
.venv/bin/python scripts/eval_mqt_bench.py \
    --ai-model models/v14_tokyo20/checkpoint_ep25333.pt \
    --n-qubits 5,10,20 \
    --benchmarks qft,qaoa,ghz,vqe \
    --output models/v14_tokyo20/eval_report_mqt.md

# 3. 不要直接重启 V15；先读 decisions.md 顶部当前状态
```

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
