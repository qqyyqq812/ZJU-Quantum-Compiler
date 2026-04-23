# 工作流与子代理规则 (量子电路 RL 项目)

> **偏离全局规则**：全局 `agents.md` 提到"并行 subagent 用于独立任务"，本项目明确规定了 5 种必须并行、3 种必须串行的场景。

## 必须并行（单条 user 消息中启动多个 Agent）

### 场景 1: 多模型对比评测

训练出多个 V<N> 权重后，要评测它们：

```
单条消息中同时启动 N 个 subagent:
  - Agent 1: 用 v12_tokyo20 跑 MQT-Bench
  - Agent 2: 用 v13_tokyo20 跑 MQT-Bench
  - Agent 3: 用 SABRE 基线跑 MQT-Bench
结果汇总到一张 markdown 表格。
```

### 场景 2: 多拓扑对比训练

不同物理拓扑（linear_5 / grid_3x3 / tokyo_20）的训练相互独立，可并行启动 Colab notebook 或 GPU 任务。

### 场景 3: 代码审查

每次 `src/compiler/` 改完后，并行启动：
- `code-reviewer`（通用质量）
- `python-reviewer`（Python idiom）
- `performance-optimizer`（GPU 利用率）

### 场景 4: 文档检索

查量子计算论文、Qiskit API、PyTorch Geometric 迁移方案时，并行：
- WebSearch（最新论文）
- docs-lookup（Qiskit 官方）
- Explore（本地 `docs/technical/`）

### 场景 5: 多 hyperparameter 实验

grid search 阶段一次性启动 4 个 subagent 跑不同 lr/entropy 组合。

## 必须串行（禁止并行）

### 场景 1: 训练 → 评测 → 决策更新

```
训练完成 → 读 history → 写 eval_report → 更新 decisions.md
```

链式依赖，不能并行。

### 场景 2: 代码修改 → pytest → push

必须等 pytest 绿了再 push，不能并行。

### 场景 3: GPU 任务

RTX 5090 只有一张卡，同一时间只能跑 1 个训练。要排队。

## ECC Skill 主动触发规则

当以下情境出现，**主动**调用 ECC skill（不等用户要求）：

| 情境 | 触发的 skill |
|-----|-------------|
| 修改 `src/` 下 Python 代码 | `python-expert-best-practices-code-review` |
| 要写训练循环 / 评测脚本 | `tdd-workflow`（先写 smoke test） |
| 设计新算法（V14 重构等） | `brainstorming` 先理清思路再动手 |
| 调试训练不收敛 | `problem-solving`（逆向思维、简化） |
| Colab 部署新方案 | `writing-plans` 先写 plan |
| 写论文/项目报告 | `academic-writing-cs` |
| 写 Markdown 文档 | `docs-writer` |
| 多次失败后的决策 | `council` 四声协商 |

## 子代理沟通模板

启动子代理时，**必须**传递项目上下文：

```
"[子代理任务]

项目上下文：
- 本项目：量子电路 RL 路由器（/home/qq/projects/量子电路）
- 当前版本：V14（见 docs/technical/decisions.md）
- 关键约束：输出不能写 handoff_*.md / V[n]改进*.md（见 .claude/rules/doc-governance.md）
- 遵守：.claude/rules/code-and-config.md（yaml 优先）
"
```

## 禁止事项

1. **禁止**让子代理做"读全项目再回答"— 指定 3-5 个具体文件
2. **禁止**串联 3 层以上子代理（Agent → subAgent → subsubAgent）— 超过 2 层改为用 TodoWrite 管理
3. **禁止**并行任务间共享可变状态（比如同时写 decisions.md）
