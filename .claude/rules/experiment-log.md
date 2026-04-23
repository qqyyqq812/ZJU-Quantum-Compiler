# 实验日志治理规则 (量子电路 RL 项目)

> **偏离全局规则**：全局 `common/git-workflow.md` 要求"文档按需生成"，本项目强制每次训练产出固定 3 文件。

## 单次训练必产出的 3 个文件

每次 `python -m src.compiler.train` 执行完毕，**必须**产出以下文件到 `models/<version>/` 下：

| 文件 | 格式 | 作用 | 保留期 |
|-----|------|------|-------|
| `history_<topology>.json` | JSON | 每 episode 的 swap/reward/stage 时间序列 | 永久（进 Git） |
| `training_<version>.log` | 文本 | stdout 完整日志（stage 晋级/eval/错误） | 7 天（gitignore） |
| `eval_report_<version>.md` | Markdown | 训练收敛后的 SABRE 对比表 | 永久（进 Git） |

## eval_report 必含字段

```markdown
# V13_tokyo20 评测报告
- 训练 episodes: 45,000
- 最终课程阶段: 4 (master, 20Q)
- 训练耗时: 5h 23min
- 模型: models/v13_tokyo20/v7_ibm_tokyo_best.pt

## SABRE 对比
| 电路 | AI SWAP | SABRE SWAP | 相对 | 完成率 |
|-----|---------|-----------|------|-------|
| QFT-20 | 28 | 32 | **-12.5%** | 100% |
...

## 关键学习曲线
![training curve](training_curve.png)
```

## 禁止事项

1. **禁止**把 training log 贴进 notebook — 提交前清空输出
2. **禁止**在 `docs/` 下写单次训练报告 — 只能写在 `models/<version>/`
3. **禁止**修改历史 `history_*.json` — 训练失败的记录也要保留，只能 append 新 version
4. **禁止**用 `print(训练过程...)` 代替 logger — 使用 `logging` 模块，writing 到 `training_*.log`

## 版本命名约定

```
models/v<N>_<topology>/
  ├─ history_<topology>.json
  ├─ training_v<N>.log
  ├─ eval_report_v<N>.md
  ├─ <model_name>_best.pt
  └─ checkpoint_ep<episode>.pt  (每 2000 ep 一次)
```

- `v<N>` 单调递增，失败的版本号**不回收**（保留作反例）
- `<topology>` = 物理拓扑名（`tokyo20`, `linear5`, `grid3x3`）
- 清理旧 checkpoint：只保留 `best.pt` 和最后 2 个 checkpoint

## 与 decisions.md 的配合

- 每开一个新版本号（如 V13→V14），**必须**在 `docs/technical/decisions.md` 追加一段说明"为什么要有这个版本、改了什么"
- `eval_report_v<N>.md` 末尾**必须**有一节"与 V<N-1> 的差异"并引用 decisions.md 的章节号
