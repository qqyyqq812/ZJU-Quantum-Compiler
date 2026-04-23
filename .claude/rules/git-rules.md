# Git 规则 (量子电路 RL 项目)

> **偏离全局规则**：全局 `git-workflow.md` 允许 "conventional commits"，本项目采用扩展格式以便与 `decisions.md` 联动。

## Commit 频率

**评分 20% 按 commit 历史打分**，必须保持"活"的 Git 状态：

- **最低频率**：每周至少 1 个有意义 commit
- **推荐频率**：每个训练阶段 1 commit（约 2-3 天 1 个）
- **禁止**：节前 24 小时内疯狂 commit（会被判定为突击作业）

## Commit Message 格式

```
<type>: <简洁描述> [<version>]

<可选的 body>

<可选的 decision ref>
```

### Types

| type | 使用场景 |
|------|---------|
| `feat` | 新算法、新功能（GNN 升级 5→9 维） |
| `fix` | 修 bug（如 V13 `final_info` 提取失败） |
| `refactor` | 代码重构（如纯 PyTorch 替代 PyG） |
| `docs` | 文档改动（decisions.md 新章节） |
| `chore` | 琐碎（gitignore、scripts） |
| `exp` | 实验性改动（不保证保留） |
| `train` | 训练相关（yaml 改参数、脚本更新） |

### 示例

```
feat(v14): cache SABRE baseline per episode

Reduces reset time 50%, allows training 20 eps/s on 5090.

Decision: docs/technical/decisions.md §V14-1
```

```
fix(pass_manager): actually emit AI SWAPs into output QuantumCircuit

Previously _build_routed_circuit() called preset SABRE PM silently,
discarding RL decisions. Now uses SwapGate.inject() directly.

Fixes critical debt listed in the warning screenshot.
```

## 分支策略

**单分支 `main`** — 学生项目不需要 feature branch。

唯一例外：**重大重构**（如 V13→V14）用短期 branch `v14-exp`，合并后删除。

## 禁止事项

1. **禁止**提交 `models/*.pt` 大文件 — 已在 `.gitignore`
2. **禁止**提交 training log（`training_*.log`）
3. **禁止**提交 `.ipynb` 的输出（用 `nbconvert --ClearOutputPreprocessor` 清空）
4. **禁止**提交 `/tmp/*`, `*.pyc`, `__pycache__/`
5. **禁止** `git push --force` 到 `main`

## Diff 检查清单

push 前必跑：

```bash
git diff --cached | grep -iE 'password|secret|token|(api[_-]?key)' && echo "LEAK!" || echo "clean"
git diff --cached --stat
```

## 历史版本号保留

- `v9_tokyo20/`, `v12_tokyo20/` 的 history 和 best.pt **必须保留**
- 即使失败的版本也留下（作为反例 + 对比基线）
- `v<N>_tokyo20/checkpoint_ep*.pt` 除最后 2 个外可删除（gitignore 已排除）
