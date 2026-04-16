# 量子电路 (Quantum Circuit)

**Project Core**: 量子电路 AI 编译器与优化框架，包含多版本模型训练、推理优化与性能评估。

## 导航索引
- **技术手册**: `docs/technical/architecture.md`（如存在）
- **任务面板**: `docs/task_board.md`
- **改动日志**: `docs/technical/cc-updates.md`（CC 自动维护）
- **Code Tour**: `.tours/` 目录（AG 讲解用）
- **项目约束**: `.claude/rules/`（如有）

## 代码原则与底层边界

- 本地项目私有法则位于 `.claude/rules/`，执行修改时严格遵循。
- 执行一切文件操作与开发任务时，通过 ECC 标准技能/验证工作流触发。

## 文件更新策略

| 文件 | 谁维护 | 何时更新 |
|------|--------|--------|
| CLAUDE.md | CC | 仅当目录结构改变 |
| docs/technical/ | CC | 架构/核心组件有显著改变时 |
| docs/technical/cc-updates.md | Hook 自动 | 每次改动自动记录 |
| .tours/ | AG | 用户要求讲解时生成 |
| docs/task_board.md | User | 主动刷新任务清单 |

## 快速开始

```bash
cd /home/qq/projects/量子电路

# 查看最近改动
cat docs/technical/cc-updates.md

# 查看可用的 notebooks
ls colab_train_*.ipynb

# 查看项目任务
cat docs/task_board.md
```
