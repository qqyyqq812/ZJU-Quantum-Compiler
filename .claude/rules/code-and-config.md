# 代码与配置治理规则 (量子电路 RL 项目)

> **偏离全局规则**：全局 `python/coding-style.md` 要求"type hints 全覆盖"，本项目对 `sandbox/` 下探索代码放宽（不要求 type hints）。

## 超参数：零硬编码

### 硬性规则

```python
# ❌ 错误 — 硬编码在代码里
learning_rate = 3e-4
rollout_steps = 32768
entropy_coef = 0.01

# ✅ 正确 — 从 yaml 读取
from src.utils.config import load_config
cfg = load_config("configs/v14_gpu.yaml")
learning_rate = cfg["learning_rate"]
```

**例外**：算法常量（如 PPO 的 `clip_epsilon=0.2`、GAE 的 `lambda=0.95`）可写死在代码里，但必须加注释说明出处。

### yaml 文件结构约定

```yaml
# configs/v14_gpu.yaml
version: v14
algorithm: PPO
environment:
  topology: ibm_tokyo
  n_qubits: 20
  max_steps: 2000      # V14 从 500 放宽
training:
  episodes: 100000
  rollout_steps: 32768
  learning_rate: 3e-4
  entropy_coef: 0.01
reward:                # V14 新增：奖励可调
  gate_reward: 0.3
  penalty_swap: -0.5
  sabre_relative_weight: 1.0
  sabre_cache: true    # V14 核心优化：缓存 SABRE 基线
curriculum:
  enabled: true
  promotion_thresholds: [3.0, 15.0, 20.0, 45.0, 100.0]
hardware:
  device: cuda
  num_workers: 20
```

## 代码组织

### 三层目录

```
src/
├── compiler/        ← 生产代码（env, policy, train, gnn_*）
│   └── 必须有 type hints，必须有 docstring
├── benchmarks/      ← 评测代码（电路生成、拓扑、基线对比）
│   └── 可免 type hints，必须可跑
└── utils/           ← 共享工具（config loader, logger）
    └── 必须有 type hints
sandbox/             ← 探索代码（Qiskit 实验、数据分析）
    └── 豁免 type hints、docstring、测试
```

### 禁止事项

1. **禁止**在 `src/compiler/` 下 `import torch_geometric` — 已由纯 PyTorch 替代（见 decisions.md §6）
2. **禁止**在训练代码里写 `print()` — 用 `logger.info(...)`
3. **禁止**在 `src/` 下 `try: except: pass` — 必须 log 异常
4. **禁止**引入新的顶层目录（如 `utils/`, `tools/`）— 必须走 `src/utils/`

## 模型文件管理

### 命名约定

```
models/v<N>_<topology>/
  ├─ v7_<topology>_best.pt      ← 历史命名延续（不改）
  ├─ checkpoint_ep<N>.pt        ← 断点
  └─ ...

注意：现有 models/ 下有大量历史遗产（v7_linear_5.pt 等），不要动。
新训练只写 v<N>_<topology>/。
```

### 权重兼容性

每个 checkpoint `.pt` 文件**必须**同目录存一个 `arch.json` 说明网络结构：

```json
{
  "version": "v14",
  "gnn": {"in_channels": 9, "hidden": 256, "out": 256, "implementation": "pure_torch"},
  "actor_hidden": 512,
  "obs_dim": 291,
  "note": "V13→V14: SABRE 基线缓存, max_steps 2000"
}
```

这样外部评测脚本能知道"这个 .pt 是哪套代码训练的"，避免 torch_geometric vs pure_torch 不兼容的重演。

## 测试策略

本项目**不追求 80% 覆盖率**（全局 `testing.md` 要求），优先级：

1. ✅ **必测**：`env.py` reset/step 的 invariant（SWAP 数单调增、门数单调减）
2. ✅ **必测**：`pass_manager.py` 的 `route_count_only()` 与 `route()` 输出一致性
3. ⚠️ **可选**：GNN 层的前向 smoke test
4. ❌ **不要求**：训练循环端到端测试（用一小步训练替代）
