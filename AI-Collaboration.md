# AI 协同文档

本项目使用 Claude Code 进行 AI 辅助开发，记录 AI 协同的主要任务与方法。

## 协同任务清单

### 项目整改（2026-04-16）

执行完整的项目规范化整改：

1. **清理旧架构残留** - 删除 `.agent_memory/` 知识图谱目录和 `v12_backup.tar.gz`（~42MB）
2. **Jupyter 笔记本规范化** - 清除全部 4 个笔记本的嵌入输出，按 `0N_description.ipynb` 格式重命名并移入 `notebooks/`
3. **硬编码路径修复** - 将 `monitor_dashboard.py` 中的 `/root/projects/量子电路/` 替换为 `os.path.abspath(__file__)` 相对路径；`cleanup_and_setup.sh` 中 `/root/quantum` 参数化为 `$PROJECT_ROOT`
4. **SSH 敏感信息脱敏** - 移除 `run_train_v13.sh` 中的 AutoDL 服务器 IP 和端口
5. **配置管理规范化** - 创建 `configs/` 目录，将 V9/V10/V13 超参数提取为独立 yaml 文件
6. **文档重写** - 重写 `README.md`（含评分对标表）、更新 `CLAUDE.md`、新建本文件

## Prompt 工程实践

### 训练参数管理

通过 `configs/*.yaml` 参数化所有超参数，避免硬编码：

```bash
# 使用配置文件启动训练
python -m src.compiler.train --config configs/v13_gpu.yaml
```

### 实验版本追踪

| 配置文件 | 对应 Notebook | 关键参数 |
|----------|--------------|---------|
| `configs/v9_baseline.yaml` | `01_train_v9_ppo_baseline.ipynb` | rollout_steps=256 |
| `configs/v9_fallback.yaml` | `02_train_v9_ppo_fallback.ipynb` | rollout_steps=2048, soft_mask=true |
| `configs/v13_gpu.yaml` | `run_train_v13.sh` | rollout_steps=32768, GPU |

### 代码审查

使用 Claude Code 进行以下代码审查：
- 验证 RL 训练循环的正确性（环境 reset、rollout 收集、GAE 计算）
- 检查 GNN 编码器的图批处理逻辑
- 审查奖励函数的 SABRE 相对设计合理性

## AI 协同效果评估

| 任务 | 效率提升 | 说明 |
|------|---------|------|
| 项目结构整改 | 大幅提升 | 批量文件重命名、路径修复、配置提取 |
| 文档生成 | 显著 | README、CLAUDE.md 从零生成 |
| 代码审查 | 显著 | 静态分析识别硬编码和安全问题 |
| 调试 | 一般 | RL 训练 bug 仍需人工介入 |

---

## V13 → V14 协同记录（2026-04-23 ~ 04-24）

### 关键决策与 AI 角色

V13 在 GPU 上训练 41k episodes 后发散（Stage 1 SWAP 从 475 涨到 852）。AI 协同诊断流程：

1. **根因定位** — 提交 history.json 给 Claude，让它分析 SWAP 时序与 stage 切换的关联，三个根因被命中：
   - SABRE baseline 在每次 reset 重新计算（吞吐 1.0 eps/s vs 预期 20 eps/s）
   - Soft Mask 在 stage 0/1 简单电路上引入了大量看似合法但无益的 SWAP 候选
   - 奖励过度依赖 SABRE 相对值，policy 早期得到 -100+ 信号后放弃探索

2. **方案设计** — V14 四大改动（V14-1 SABRE 缓存、V14-2 阶段化 Mask、V14-3 奖励分层、V14-4 pass_manager 真集成），由 Claude 起草 → 人工评审 → 落地。每条改动入 `docs/technical/decisions.md` 并单独 commit。

3. **实装与验证** — 9/9 V14 smoke test 通过（包括 SABRE 缓存命中率、阶段切换 mask 一致性、pass_manager 输出可重现性）。

### V14.1 三级修复（4 月 25 日 GPU 场景）

GPU 训练后发现 V14 在 Stage 3 入口遇到 reward 悬崖（stage<=2 给 +5 floor，stage 3 直接切到 -481）。再次 AI 协同诊断 → 三个补丁：
- truncation 必须惩罚（否则 agent 学"刷 SWAP 到超时"退化）
- Stage 3 桥接 reward = floor + 0.3·SABRE（避免悬崖）
- resume 后必须 propagate stage 到所有 worker

详见 `decisions.md §V14.1`。

---

## V14 → V15 协同记录（2026-04-25）

### 触发事件

V14.2 在 5090 GPU 上跑 ~2200 episodes 后，训练 SWAP 从 589 上涨到 722，eval avg_swap 在 45-62 间震荡，**reward 趋势恶化** —— 这不是收敛慢，是真的卡住了。

### AI 主导的 SOTA 调研

启动 3 路并行调研 subagent（参考 `.claude/rules/workflow-agents.md` 场景 4）：
- WebSearch + WebFetch 拉 Qiskit / IBM Quantum 2024-2026 路由 ML 进展
- arXiv + Google Scholar 拉 RL routing 论文
- GitHub 拉公开实现

**关键发现**：
- 我们当前 V14 的算法路线 (PPO + GNN + SABRE 相对奖励) 最接近 Zhou 2024 (arXiv:2407.00736)，**未开源、未复现**
- 已验证 20Q SOTA 全部是 **MCTS + 神经网络**：QRoute (AAAI 2022, 公开)、AlphaRouter (Amazon 2024, 公开)
- LightSABRE (IBM, Sep 2024) 把启发式基线抬高了 18.9%，让 2024 之前很多"打败 SABRE"的 RL 论文成绩失效

### 决策：V14 → V15 算法切换

**保留 70% V14 工程**（env / GNN / SABRE 缓存 / curriculum / pass_manager）+ **替换学习算法**（PPO → AlphaZero 风格 MCTS+GNN）。

V15 不是从头来过，是站在 V14 的肩膀上：
- V14 ep25333 权重 → V15 网络 backbone warmstart
- V14 LightweightEnv O(1) clone → MCTS 必备的快速仿真
- V14 9D GraphSAGE → V15 PolicyValueNet 共享 backbone

详见 `decisions.md §V15`。

### V15 实装协同

并发 subagent 任务（每个限定 3-5 文件读取范围、限定 200-500 字输出）：
- A1: pip 包化（pyproject.toml + src/cli.py + 验收）
- A2: 训练曲线图（4 张 PNG，从 history.json 直出）
- A4: MQT-Bench 评测管线
- 主线程：V15 代码骨架（5 模块 + 测试 + yaml 配置）

V15 设计是 AI 与人工协同的产物：
- AI 调研出 SOTA 路径 → 人工确认"接受技术路线切换"
- AI 写代码骨架（network/tree/selfplay/replay/train）→ 人工评审 + smoke test
- 每个模块 ≤ 250 行（参考 `.claude/rules/code-and-config.md` 高内聚原则）

### AI 协同方法论沉淀

| 协同模式 | 适用场景 | 反例 |
|---------|---------|------|
| **诊断驱动** | 训练异常、reward 不收敛 | 直接让 AI 写新算法（缺乏诊断） |
| **SOTA 调研** | 路线选择前 | 闭门造车 |
| **并行 subagent** | 独立工程任务（pip / 图 / 评测） | 串行依赖任务硬塞并行 |
| **决策记录** | 算法改动入 `decisions.md` | 写 handoff_*.md（项目硬规则禁止） |

### 协同效果评估（V14→V15）

| 任务 | 效率提升 | 说明 |
|------|---------|------|
| SOTA 调研 | 大幅提升 | 3 个 subagent 并行 ~15 分钟拿出有引用的报告，对比人工调研 1-2 天 |
| 算法路线选择 | 大幅提升 | AI 把"PPO vs MCTS"的证据链摆出来，人工只需做 yes/no 决策 |
| 代码骨架 | 大幅提升 | V15 的 5 模块 ~1200 行从设计到实装 ~2 小时（含审查） |
| Reward 设计 | 显著 | tanh value head + clip [-1,1] 是 AI 提出的，避免重蹈 V14 -500 量级覆辙 |
| 训练调参 | 一般 | 仍需 GPU 实际跑数据 |
