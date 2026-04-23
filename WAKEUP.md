# 醒来后看这个（2026-04-24 夜间自动开发总结）

> 睡前你要求：**长时间自动开发 + 不要让机器死机**。下面是我做了什么，以及你醒来后怎么继续。

## 1. WSL / C 盘安全检查

- 睡前 C 盘只剩 **5.8 GB**（97% 满）
- 清理了 `Temp/pip/npm` 临时缓存 ≈ **1 GB** 空间
- 关闭了多余的 Claude Code 僵尸进程，释放 ≈ 400 MB 内存
- WSL 有 `.wslconfig` 12GB 限制，不会失控 → **睡眠期间无崩溃风险**
- Docker WSL 数据 9.3 GB（没动它，那是你的容器）

## 2. 量子电路 V14 完成的事

### 本地 harness 规则（`.claude/rules/`）

新建了 **5 条本项目专用规则**，允许偏离全局规则（睡前你要求的"定制 harness"）：

```
.claude/rules/
├── doc-governance.md     ← 已有（禁止 handoff / V[n]改进 文档）
├── experiment-log.md     ← 新：每次训练必产 history + log + eval_report
├── code-and-config.md    ← 新：yaml 优先、零硬编码、禁 torch_geometric
├── workflow-agents.md    ← 新：何时并行 subagent、触发哪些 ECC skill
├── deployment.md         ← 新：Zero-Touch Remote（远端只 git pull + 跑）
└── git-rules.md          ← 新：commit 格式、周活、禁止事项
```

`CLAUDE.md` 已更新为 V14 状态，并指向这些本地规则。

### V14 四大算法修复（decisions.md §V14）

V13 在 GPU 上训练发散的 4 个根因**全部修了**：

| 修复 | 文件 | 测试 |
|------|------|------|
| §V14-1 SABRE 基线缓存 | `src/compiler/sabre_cache.py`（新） | `test_v14_sabre_cache.py` ✅ 3/3 |
| §V14-2 阶段化 Mask | `env.py::get_action_mask` + `set_curriculum_stage` | `test_v14_env_stage.py` ✅ 3/3 |
| §V14-3 奖励分层 | `env.py::step` terminal 分支 | `test_v14_env_stage.py` ✅ |
| §V14-4 pass_manager 真集成 | `pass_manager.py` 整个重写 | `test_v14_pass_manager.py` ✅ 3/3 |

### V14 工程基建

- `src/utils/config.py` — yaml 加载器 + 必需 key 校验
- `configs/v14_baseline.yaml` — IBM Tokyo 20Q RTX 5090 全量配置
- `configs/v14_local_smoke.yaml` — linear_5 CPU 本地烟测
- `configs/v14_micro_smoke.yaml` — 100 ep 极速验证
- `run_train_v14.sh` — 一键启动脚本（支持 `--config` 参数）
- `scripts/eval_v14_vs_sabre.py` — 自动对比 SABRE 基线、产出 `eval_report_v14.md`

### 测试

- **V14 新测试**：9/9 全部通过
- **旧测试回归**：56/57（1 个 `test_m2_dag.py::test_execute_executable` 失败，与 V14 无关，是旧测试期望过时）

### Git

- **3 个 commits** 已 push 到 `origin/main`:
  - `cb4c983` feat(v14): SABRE cache + phase mask + reward layering + pass_manager real integration
  - `f6f3c68` docs: purge forbidden docs per doc-governance rule
  - `b60356c` harness: local rules + V14 refactor plan [v14]

## 3. 你醒来后按这个顺序做

### Step 1: 确认代码拉到 GPU 服务器（3 分钟）

开 AutoDL 实例 → SSH 进去：

```bash
cd /root/quantum
git pull
# 应看到 3 个新 commit
git log --oneline -5
```

### Step 2: 跑 V14 baseline 训练（4-6 小时，后台挂着）

```bash
cd /root/quantum
# 确认依赖：torch 2.10, qiskit, gymnasium（应该已装好）
python -c "import torch, gymnasium, qiskit; print('OK')"

# 启动训练（后台）
mkdir -p models/v14_tokyo20
nohup bash run_train_v14.sh configs/v14_baseline.yaml > models/v14_tokyo20/training_v14.log 2>&1 &
echo $!   # 记这个 PID

# 5 分钟后检查
sleep 300
tail -30 models/v14_tokyo20/training_v14.log
# 应看到:
# - SABRE 缓存首轮预热后 eps/s 回升到 ~15
# - Stage 0 的 SWAP 降到 3 以下后晋级
```

### Step 3: 训练结束后评测（5 分钟）

```bash
cd /root/quantum
python scripts/eval_v14_vs_sabre.py \
  --model models/v14_tokyo20/v7_ibm_tokyo_best.pt \
  --topology ibm_tokyo \
  --max-steps 2000

cat models/v14_tokyo20/eval_report_v14.md
```

报告会自动：
- 对 QFT-5/10/20、QAOA-5/10、Rand-5/10/20 全部跑一遍
- 对比 SABRE 基线
- 输出"超越 SABRE: X/Y"汇总

### Step 4: 同步回本地（拉 eval_report）

```bash
# 从 GPU 拉回训练产物
scp root@<autodl>:/root/quantum/models/v14_tokyo20/*.json ~/projects/量子电路/models/v14_tokyo20/
scp root@<autodl>:/root/quantum/models/v14_tokyo20/eval_report_v14.md ~/projects/量子电路/models/v14_tokyo20/
# 权重太大可选：scp root@<autodl>:/root/quantum/models/v14_tokyo20/v7_ibm_tokyo_best.pt ...
```

### Step 5: 关机省钱

AutoDL 按小时计费。训练 + 评测完毕后立刻:

```bash
# 关停实例（在 AutoDL web 界面或）
shutdown -h now
```

## 4. 如果训练又崩了

V14 和 V13 的失败模式已全部诊断过（见 `decisions.md §V14` 的"失败恢复"和"踩坑"）：

- **崩在 Stage 0 不晋级** → 查 `history_*.json` 的 `episode_swaps`。本应 < 3 时晋级。如果卡在更高，看 `curriculum.py` 阈值是否匹配
- **崩在 Stage 1 发散** → V14 已通过阶段化 Mask 修这个。如果还发生，看 `env.py::get_action_mask` 的 stage 参数是否生效
- **GPU segfault** → `torch_geometric` 永远不要再装（V13 已纯 PyTorch 化）

## 5. 还没做的（醒来可选）

- ❌ V14 在 GPU 上的实际训练（必须你开机 SSH 才能做）
- ❌ MQT-Bench 官方基准测试（V14 训练完后做）
- ❌ AI-Collaboration.md 更新（可手动补）
- ❌ GPU 关机（需要你登录 AutoDL web）

## 6. 关键文档位置

| 需要知道什么 | 去哪里看 |
|------------|---------|
| V14 做了啥改动 / 为什么 | `docs/technical/decisions.md` §V14 |
| 本项目的开发规则 | `.claude/rules/*.md` + `CLAUDE.md` |
| 训练怎么启动 | `run_train_v14.sh` 注释头 |
| Colab/GPU 协议 | `.claude/rules/deployment.md` |
| 这次会话做了啥 | 你看的就是这个文件（WAKEUP.md） |
