# 技术决策记录 (量子电路)

## 当前版本状态 (V14)

**最后更新**：2026-04-24
**GPU 环境**：RTX 5090 32GB (AutoDL)
**当前阶段**：V14 代码实现中 — 修复 V13 发散根因 + pass_manager 假集成

---

## V14 重构（2026-04-24）

### 背景：V13 发散事故复盘

V13 于 2026-04-10 在 GPU 上跑了 41k episodes，暴露 **4 个根因**：

| 根因 | 现象 | 位置 |
|-----|------|------|
| **SABRE baseline 每 reset 重算** | 吞吐从 2.8 降到 1.0 eps/s | `env.py::_compute_sabre_baseline()` |
| **Soft Mask + 随机映射组合爆炸** | Stage 1 (5Q) SWAP 从 475 涨到 852 | `env.py::get_action_mask()` |
| **奖励 shaping 削太狠** | `reward_gate=0.3` 让模型失去执行门动力 | `env.py` |
| **pass_manager 假集成** | AI SWAP 决策被丢弃，输出本质是 SABRE 重编译 | `pass_manager.py::_build_routed_circuit()` |

### V14 四大改动

#### §V14-1: SABRE 基线缓存

**决策**：用 `(circuit_id → sabre_swap_count)` 的 LRU 缓存，一个电路只跑一次 SABRE。

**实现**：
- `src/compiler/sabre_cache.py` 新模块
- `env.py::_compute_sabre_baseline()` 改为查询缓存
- 缓存键：电路的 qubits 数 + 门序列 hash

**预期收益**：吞吐从 1.0 → 15+ eps/s（20 个 worker 共享缓存）

---

#### §V14-2: 阶段化 Mask

**决策**：Mask 策略随课程阶段演化：
- Stage 0-1（3Q/5Q 学基础）：**Hard Mask**，只允许降距离的 SWAP
- Stage 2-3（5Q/10Q 过渡）：**Soft Mask delta≤1**
- Stage 4（20Q 真实挑战）：**Soft Mask delta≤2** + Tabu

**理由**：V13 的 Soft Mask 在 5Q 上让 action 空间爆炸。先用 Hard Mask 把 policy 教会"基本动作正确"，再逐渐放宽。

**实现**：`env.py::get_action_mask()` 增加 `curriculum_stage` 参数。

---

#### §V14-3: 奖励分层

**决策**：奖励函数随 stage 切换：
- Stage 0-2（学会完成电路）：`reward_done=5.0` + `reward_gate=1.0`（恢复 V12 配置）
- Stage 3-4（学会打败 SABRE）：`reward_done=0.0` + `reward_gate=0.3` + SABRE 相对终端

**理由**：V13 一上来就用 SABRE 相对奖励，但模型连"把电路路由完成"都没学会（done=67% 而非 100%），相对奖励信号太稀疏。先教"完成"，再教"超越"。

**实现**：`env.py` 的奖励计算读取 `curriculum_stage`，`train.py` 每次晋级时传入。

---

#### §V14-4: `pass_manager.py` 真集成

**决策**：`_build_routed_circuit()` 必须把 AI 的 SWAP 决策**真正写入**输出的 `QuantumCircuit`，不能再调 SABRE 重编译。

**实现**：
1. 遍历原电路的门序列
2. 应用 AI mapping 转换逻辑 qubit → 物理 qubit
3. 在每次 AI 决策的位置插入 `SwapGate`
4. 保证输出电路的功能与输入完全等价（可用 `Operator(原电路) == Operator(输出)` 验证）

**理由**：这是**硬伤级工程债**。当前的集成是"假"的，外部用户调用 `AIRouter.route()` 不能复现我们的 SWAP 数。论文交付时会被 reviewer 一眼看穿。

**参考**：Qiskit 的 `SabreSwap` TranspilerPass 源码。

---

## 核心算法决策

### 1. 奖励函数：SABRE 相对终端奖励
**决策**：从原始奖励方案改为 SABRE 相对终端奖励 `r = sabre_swaps - ai_swaps`

**理由**：
- 直接对标 IBM SABRE 启发式算法，有明确的竞争基线
- 相对奖励避免了绝对奖励的稀疏性问题
- 更容易衡量 AI 编译器的真实性能优势

**关联文件**：`src/compiler/env.py`, `train.py`

---

### 2. 动作约束：Soft Mask (允许 delta≤1)
**决策**：将硬约束 (Hard Mask) 改为软约束，允许偏离 SABRE 启发式 ≤1 步的动作

**理由**：
- Hard Mask 过度限制了探索空间，导致 policy 无法学到非贪心策略
- Soft Mask 允许偏差但给予惩罚，平衡了探索与约束
- 与真实硬件路由的灵活性更加贴近

**实现**：`src/compiler/curriculum.py` (action masking logic)

---

### 3. GNN 特征工程：从 5D 扩展到 9D
**决策**：增加 GNN 输入特征维数 5 → 9，新增：映射距离、DAG 深度、前沿目标距离

**特征列表**：
- 原有 5D：qubit 状态、swap 历史、连接性...
- 新增 4D：qubit 当前到目标的映射距离、当前 DAG 的深度、前沿层中目标节点的距离、局部队列深度

**理由**：
- 帮助 GNN 更全面地理解 DAG 拓扑和路由状态
- 提高 policy 对全局最优的感知
- 减少 RL 训练所需的样本数量

**关联文件**：`src/compiler/gnn_extractor.py`, `dag.py`

---

### 4. 初始映射：消除 Identity 偏见
**决策**：弃用默认恒等映射，采用随机初始化映射

**理由**：
- Identity mapping 导致 policy 对起点位置产生偏见
- 随机初始化强制 policy 学习真正的路由策略，而非"近邻贪心"
- 更符合实际硬件部署中的多样化初始条件

**实现**：`src/compiler/env.py` 的初始化逻辑

---

### 5. 课程学习：大幅放宽阈值
**决策**：提高课程学习的电路复杂度阈值，匹配 IBM Tokyo (20Q) 的真实 SABRE 基线

**调整**：
- 前期课程：8-12Q 简单电路 → 目标与 SABRE 差距 ≤10%
- 中期课程：14-18Q 中等电路 → 目标与 SABRE 差距 ≤5%
- 后期课程：20Q 真实电路 → 追平或超越 SABRE 基线

**理由**：
- 之前的低阈值导致 policy 在简单电路上 overfit，泛化能力弱
- 匹配 IBM Tokyo 20Q 是论文的实验对标点

**关联文件**：`src/compiler/curriculum.py`

---

## 踩坑记录

### 1. Action Mask 困境（V8-V12 历史）
**问题**：Hard Mask 导致 policy 无法探索最优解。

**症状**：
- 在简单电路上能与 SABRE 持平，但面对复杂电路时大幅落后
- 样本效率极低（需要 10M+ 步）
- Policy 收敛到局部最优但无法突破

**解决**：V13 改为 Soft Mask，允许偏离但带惩罚项。结果在相同步数下性能翻倍。

---

### 2. GNN 特征不足（V11 调查）
**问题**：5D 特征无法捕捉 DAG 的全局结构。

**调查结果**：Policy 倾向于短视的贪心决策，而非考虑整体 DAG 拓扑。

**解决**：V13 扩展到 9D，直接编码 DAG 深度和前沿信息。验证后样本效率提升 40%。

---

### 3. Identity Bias（V10 发现）
**问题**：恒等映射导致 policy 过度依赖起始配置。

**症状**：迁移到不同初始化后性能急剧下降。

**解决**：V13 改为随机初始化，强制 policy 学习真正的路由逻辑。

---

## V14 重构计划 (2026-04-24 启动)

V13 在 GPU 上跑了 41k episodes 后暴露出**训练发散**问题：Stage 1 (elementary 5Q) 的 SWAP 均值从 475 一路涨到 852，而不是下降。复盘定位根因：

### 根因 1：SABRE 基线重复计算拖慢 20 倍
V13 在每个 `env.reset()` 都调用 `qiskit.transpile(..., routing_method='sabre')`。20 个 AsyncVectorEnv worker × 每个 episode reset × 每次 transpile 几十毫秒，吞吐从预期的 20 eps/s 掉到 **1.0 eps/s**。

### 根因 2：Soft Mask 在训练早期放大了噪声
允许 delta<=1 的 SWAP 在 5Q 简单电路上产生大量"看似合法"但实际无益的动作，policy 分不清信号。

### 根因 3：奖励过度依赖 SABRE 相对值，早期学习困难
当 AI 还没学会基本路由时，SABRE 相对奖励可能极负（-100+），entropy 衰减后 policy 放弃探索。

---

## V14 核心改动

### V14-1: SABRE 基线缓存（P0）
- 预先计算每条训练电路的 SABRE baseline，存入 `data/sabre_baseline_cache.json`
- 训练时 `env.reset()` 直接查表，不调 Qiskit
- **预期效果**：吞吐恢复到 15-20 eps/s

**实现位置**：`src/compiler/env.py` 的 `reset()` 方法

### V14-2: 阶段化 Mask 策略（P0）
- Stage 0-1 (warm-up + elementary)：**Hard Mask**（只允许严格缩短距离的 SWAP）
- Stage 2-3 (standard + challenge)：**Soft Mask delta<=1**
- Stage 4 (master)：**Soft Mask delta<=2** + Tabu

**理由**：早期需要强约束收敛，后期需要松约束探索。

**实现位置**：`src/compiler/env.py` 的 `get_action_mask()`，读取 curriculum stage 自动切换

### V14-3: 奖励分层平衡（P0）
新奖励公式：

```
r_step = 0.5 * (gates_executed) - 0.5 * (swap_this_step) + 0.2 * (distance_reduction)
r_terminal = clip(sabre_swaps - ai_swaps, -20, +20) if use_sabre else +10
```

- 训练早期（前 10k ep）：`use_sabre=False`（用固定 +10 学会完成）
- 中期开始启用 SABRE 相对奖励
- clip(-20, +20) 防止早期发散时的毁灭性梯度

**实现位置**：`src/compiler/env.py` + `train.py` 读 curriculum stage 决定

### V14-4: 修复 pass_manager 假集成（P0）
**问题**：`_build_routed_circuit()` 内部用 Qiskit preset PM 重编译，AI SWAP 决策被丢弃。

**方案**：改用 `qiskit.circuit.library.SwapGate` 按 AI 决策逐步插入，生成**真正**由 AI 产出的电路。

```python
from qiskit.circuit.library import SwapGate
# 按 swap_list 和 mapping 顺序构建
for swap in swap_list:
    qc.append(SwapGate(), [swap.qubit_a, swap.qubit_b])
```

**实现位置**：`src/compiler/pass_manager.py`

### V14-5: max_steps 按电路深度自适应（P1）
V13 固定 `max_steps=500`，20Q 深电路不够。改为：

```python
max_steps = max(500, 10 * n_two_qubit_gates)
```

### V14-6: MQT Bench 评测管线（P1）
在 `src/benchmarks/` 下新增 `mqt_bench.py`，拉取 MQT Bench 官方 `.qasm` 文件批量评测。

---

## V14.1 踩坑 & 修复（2026-04-25）

### 症状
V14 训练从 ep5000 进入 Stage 3 (10Q) 后 SWAP 立即从 Stage 2 的 ~66 飙到 500+，eval 结果 AI SWAP≈47 vs SABRE SWAP≈19（差 2.5×）。21000 episodes 训练完全无收敛。

### 根因分析（3 个）

**(a) Truncation 无惩罚**：`env.step` 仅在 `terminated` 分支给终端奖励；`truncated`（跑满 max_steps=600）直接返 0。agent 很快找到退化策略："不断 SWAP 直到超时"比"尝试完成电路但可能做出 50 次不利 SWAP"更安全。

**(b) Stage 2→3 奖励悬崖**：stage≤2 时奖励 = `max(5, done) + 0.1·(sabre_swaps - ai_swaps)`；stage≥3 突然切到 `1.0·(sabre_swaps - ai_swaps)`。10Q 电路 SABRE=19，AI=500 意味着奖励从 +5 级瞬间变 -481，策略梯度爆炸。

**(c) Resume 未传播 stage 到 env**：`envs.call("set_curriculum_stage", ...)` 只在 `promoted` 分支调用。断点续训时 scheduler 内部推到 Stage 3，但 100 个 AsyncVectorEnv worker 的 env 实例 `_curriculum_stage` 还是 0，于是奖励分层和 mask 阶段性全错。

### 修复（commit 061c680, 4856f98）

1. **truncation 惩罚**：`env.step` 的 `elif truncated` 分支给 `-remaining_gates - 0.5·(ai_swaps - sabre)`，强迫 agent 必须完成电路
2. **Stage 3 桥接**：reward_floor=5 + 0.3·SABRE 相对（而非 1.0×），避开悬崖
3. **Resume 传播**：`resume_path` 加载后立即 `envs.call("set_curriculum_stage", target_stage)`
4. **max_steps 600→300**：短回合 + 惩罚 ⇒ 不完成电路代价极高
5. **CLI --resume 优先 yaml**：yaml 的 `resume: null` 之前覆盖了命令行参数

### 验证
- 59/59 V14 smoke tests 通过（1 个历史无关 deselect）
- 本地单 env truncation 回合 reward=-139（原本 0）
- GPU 续训 ep25333 → Stage 3 状态正确恢复（待观察 SWAP 下降）

---

## V14 验收标准

| 指标 | V13 | V14 目标 |
|-----|-----|---------|
| 训练吞吐 | 1.0 eps/s | ≥ 15 eps/s |
| Stage 1 收敛 | ❌ 发散 | ✅ SWAP 单调下降 |
| Stage 4 完成率 | ? | ≥ 80% |
| IBM Tokyo 20Q vs SABRE | ? | 相对差 ≤ 5% 或优于 |
| pass_manager 真实性 | ❌ 假集成 | ✅ AI SWAP 真实输出 |
| MQT Bench 覆盖 | ❌ 只有手工 | ✅ 10 条官方电路 |

---

## 后续目标

- [ ] V14-1 ~ V14-4 实现并本地 smoke test（无 GPU）
- [ ] GPU 上训练 V14 100k ep + 生成 eval_report_v14.md
- [ ] MQT Bench 基线评测
- [ ] 论文图表生成（训练曲线 + SWAP 对比）
- [ ] AI-Collaboration.md 扩充（覆盖 V13→V14 的协同决策）

---

## 参考
- 笔记本：`notebooks/03_train_v10_ppo_hardmask.ipynb`
- 训练代码：`src/compiler/train.py`
- 配置文件：`configs/v14_gpu.yaml` (待建)
- 训练脚本：`run_train_v14.sh` (待建)
- 规则：`.claude/rules/`（本地 harness）

---

## V15 重构 (2026-04-25)：从 PPO 切到 AlphaZero (MCTS+GNN)

### 触发事件

V14.2 在 5090 GPU 上跑了 ~2200 episodes（Stage 3 challenge，10Q），训练 SWAP 从 589 上涨到 722，eval avg_swap 在 45-62 之间震荡，**reward 趋势恶化**（−404 → −573）。详见远端日志 `models/v14_tokyo20/training_v14.log`。

诊断：**这不是收敛慢的问题，是 reward landscape 在 Stage 3 入门处形成了平坦盆地**。完成与超时都是大负数 reward，PPO 梯度信号是噪声，无法逃出局部解。继续训练 100k episodes 大概率不会突破。

### SOTA 现状（2026-04 调研）

| 方法 | 算法 | 20Q 拓扑结果 | 是否开源 |
|-----|-----|------------|--------|
| LightSABRE (IBM 2024)    | 启发式（Rust）         | -18.9% SWAP vs 原 SABRE | ✅ Qiskit ≥1.2 默认 |
| AIRouting (IBM 2024)     | RL transformer        | 100+Q heavy-hex -24% 双比特门 | ✅ qiskit-ibm-transpiler |
| **AlphaRouter (Amazon 2024)** | **MCTS + transformer** | **20Q Tokyo -10~20% SWAP** | 🟡 HYUIDSL/AlphaRouter |
| **QRoute (AAAI 2022)**   | **MCTS + GNN**        | **超过 SABRE/TKET on Tokyo 20Q** | ✅ 公开 |
| Zhou 2024 (arXiv 2407.00736) | PPO + GNN            | 声称 -5~15% vs SABRE   | ❌ 闭源 |

V14 的 PPO+GNN+SABRE 相对奖励路线最近似 Zhou 2024（**未开源、未复现**）。已验证的 20Q SOTA 路线全部是 **MCTS + 神经网络**。

### V15 设计：保留 70% V14 工程，更换学习算法

```
保留 (无修改):
  src/compiler/env.py               — 状态/动作/reward 不动
  src/compiler/light_env.py         — O(1) clone, MCTS 必备
  src/compiler/gnn_encoder.py       — 9D 纯 PyTorch GraphSAGE
  src/compiler/gnn_extractor.py     — 物理图特征提取
  src/compiler/sabre_cache.py       — V14-1 吞吐优化
  src/compiler/pass_manager.py      — V14-4 真集成
  src/compiler/curriculum.py        — 课程学习调度

新增 (src/compiler/v15/):
  network.py    — PolicyValueNet (共享 GNN backbone + policy/value 双头)
  tree.py       — PUCT 选择 + Dirichlet 根噪声 + lazy expansion
  replay.py     — (state, π_MCTS, z) replay buffer
  selfplay.py   — 单个 episode self-play + outcome 反向填充
  train.py      — 外循环 (生成 → 训练 → 评测 → 检查点)

新增脚本 + 配置:
  scripts/train_v15.py     — 入口
  configs/v15_baseline.yaml — 全部 hparam
  tests/test_v15_smoke.py  — CPU 烟雾测试
```

### V15 关键设计决策

**V15-1: 沿用 LightweightEnv 而非自建 env**
理由：MCTS 需要 ~100 次仿真 × 800 步 = 80k env 拷贝。`LightweightEnv.clone()` 是 O(1)（只复制 dict + numpy bool 数组），比 deepcopy QuantumRoutingEnv 快 1000+ 倍。这是项目已有积累，直接复用。

**V15-2: 共享 GNN backbone，重新初始化双头**
V14 ep25333 的 GraphSAGE 编码器在 5Q 任务上已收敛到合理特征空间。V15 网络的 backbone 直接 load V14 权重做 warmstart；policy/value 双头从零训（监督信号不同：PPO advantage vs MCTS visit/outcome）。

**V15-3: Reward scheme = relative_sabre, clip ∈ [-1, 1]**
完成时 `z = clip((sabre_swaps - ai_swaps) / sabre_swaps, -1, 1)`，超时 `z = -1`。tanh value head 天然匹配 [-1, 1]，避免 V14 那种 `−500` reward 量级导致的梯度爆炸。

**V15-4: Dirichlet root noise α=0.3**
AlphaZero 围棋用 α=0.03（动作空间 ~362）。路由 38 动作，按 αN ≈ 10 经验法则取 α=0.3。混合系数 ε=0.25（标准）。

**V15-5: 课程学习继承 V14 STAGES 阈值**
不动 `curriculum.py`。但 V15 的 episode 比 V14 PPO 步慢（每步 100 次 MCTS 仿真），把 `min_episodes_per_stage` 从 100 调到 50，`promotion_patience` 从 3000 调到 500。

### V15 验收标准（替代 V14 验收）

| 指标 | V14 目标 | V14 实测 | V15 目标 |
|-----|---------|---------|---------|
| 训练稳定性 | ✅ SWAP 单调下降 | ❌ Stage 3 恶化 | ✅ relative_sabre z 单调上升 |
| Stage 0-2 (3-5Q) 完成率 | — | ≥95% (ep25333) | ≥98% (warmstart 起点) |
| Stage 3 (10Q) 完成率 | ≥80% | ❌ 67-100% 震荡 | ≥85% |
| Stage 4 (20Q Tokyo) AI/SABRE | ≤1.05 | 未达 | ≤1.10 (匹配 QRoute SOTA) |
| 训练时间 | 4-6h | 已 8h+ 未收敛 | 5-7 天（self-play 比 PPO 慢但更稳） |

### 与 V14 之前版本的关系

V15 不废弃任何之前版本：
- V9-V12 的 history 文件留存（评分要求"保留失败版本作反例"）
- V13 的 9D GNN 改造 + 随机初始映射 → V15 直接用
- V14.0~14.2 的所有改动（SABRE 缓存、阶段化 mask、reward 分层、pass_manager 真集成、truncation 惩罚、Stage 3 桥接、resume 传播、max_steps=800）→ V15 全部保留在 env.py 中
- V14.2 ep25333 权重 → V15 网络 warmstart 起点

### 后续里程碑（V15 时间线）

- [x] V15 代码骨架（network/tree/replay/selfplay/train + smoke tests + yaml）— 2026-04-25
- [ ] CPU 本地 smoke：`pytest tests/test_v15_smoke.py -v` 全绿
- [ ] CPU 本地 1 iteration 跑通（games_per_iter=2, n_simulations=4）
- [ ] GPU push V15：cd /root/quantum && git pull && nohup bash -c "python scripts/train_v15.py --config configs/v15_baseline.yaml" &
- [ ] GPU 训练 5-7 天（共享 5090 与近似乘法器项目，量子占 ~5GB 显存可并行）
- [ ] eval_report_v15.md：MQT-Bench 上 v14 ep25333 / v15 / SABRE / LightSABRE 四方对比
- [ ] AI-Collaboration.md V13→V15 章节扩写

### 参考代码

- AlphaRouter（参考 MCTS 实现）：https://github.com/HYUIDSL/AlphaRouter
- QRoute 论文：Sinha et al. AAAI 2022
- AlphaZero 原文：Silver et al. Science 2018
- 本项目 V14 SOTA 调研：sub-agent 报告（已并入本节，不另存外部 md）
