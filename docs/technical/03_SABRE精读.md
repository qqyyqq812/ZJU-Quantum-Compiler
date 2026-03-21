# SABRE 算法精读笔记

> **论文**: Li et al., *"Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices"*, ASPLOS 2019
> **核心贡献**: 提出 SWAP-based BidiREctional heuristic search (SABRE)，成为 Qiskit 默认路由算法

---

## 1. 问题定义

**输入**:
- 逻辑量子电路（DAG 表示）
- 物理芯片的 Coupling Map

**输出**:
- 满足 Coupling Map 约束的物理电路（插入必要的 SWAP 门）
- 逻辑比特到物理比特的映射

**优化目标**: 最小化额外插入的 SWAP 门数量 → 降低电路深度和噪声

---

## 2. 核心算法流程

```
输入: 逻辑 DAG D, Coupling Map G, 初始映射 π

1. 提取前沿层 F = DAG 中无依赖的门集合
2. WHILE F 非空:
   a. 对 F 中每个门 g = CNOT(q1, q2):
      - 如果 π(q1) 和 π(q2) 在 G 中相邻 → 执行 g，从 D 中移除
   b. 如果有门被执行 → 更新 F → 回到 2
   c. 如果没有门可执行（所有前沿门的比特都不相邻）:
      - 对 G 中每条边 e 计算启发式代价 H(SWAP(e))
      - 选择 H 值最小的 SWAP 执行
      - 更新映射 π
3. 返回修改后的电路和最终映射
```

---

## 3. 启发式函数 H

SABRE 的核心创新在于启发式函数的设计：

```
H(SWAP) = Σ_{g ∈ F} (1/|F|) × distance(π'(q1_g), π'(q2_g))
```

其中：
- `F` = 前沿层（当前可执行的门集合）
- `π'` = 执行 SWAP 后的新映射
- `distance()` = Coupling Map 上的最短路径长度
- 直觉：**选择让前沿门们整体更接近可执行的 SWAP**

### 扩展启发式（look-ahead）

SABRE 还考虑了下一层的门（extended set E）：

```
H_extended = H_front + W × Σ_{g ∈ E} (1/|E|) × distance(π'(q1_g), π'(q2_g))
```

- `W` = 权重因子（论文中 W = 0.5）
- `E` = 前沿层后继门的集合
- 这使得算法不完全贪心，有一定的前瞻能力

---

## 4. 双向搜索（Bidirectional）

SABRE 的另一个创新是**双向搜索**：

1. **正向遍历**: 从 DAG 头到尾执行路由
2. **反向遍历**: 将 DAG 反转，从尾到头执行路由
3. 取**门数更少**的结果

反向遍历的直觉：正向搜索的初始映射对前面的门友好但对后面的门不友好；反向搜索相当于让初始映射对"最后执行的门"友好。

---

## 5. SABRE 的局限性（为什么可以被超越）

| 局限 | 解释 | 改进方向 |
|------|------|----------|
| **贪心策略** | 只看当前前沿和下一层，无全局视野 | RL 可以学全局策略 |
| **固定启发式** | H 函数形式固定，不自适应不同电路 | GNN 可以学更好的特征 |
| **初始映射随机** | 好的初始映射能显著减少 SWAP | GNN 可以学最优初始映射 |
| **无学习能力** | 不能从历史编译中学习经验 | RL 训练后泛化 |

---

## 6. Qiskit 实现

Qiskit 中 SABRE 通过两个 Pass 实现：

- `SabreLayout`: 负责初始映射（多次随机+SABRE路由，选最优）
- `SabreSwap`: 负责路由（上述核心算法）

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit 默认使用 SABRE 进行 layout 和 routing
pm = generate_preset_pass_manager(
    optimization_level=1,        # 0-3, 越高优化越激进
    coupling_map=coupling_map,
    basis_gates=['cx', 'id', 'rz', 'sx', 'x']
)
compiled = pm.run(circuit)
```

---

## 7. 关键数据（论文结果）

| 基准电路 | Qubits | 原始门数 | SABRE CNOT数 | vs FPP 提升 |
|----------|--------|---------|-------------|------------|
| QFT-16   | 16     | 120     | 245         | -27%       |
| Grover-9 | 9      | 114     | 198         | -31%       |
| Random-20| 20     | 200     | 412         | -24%       |

SABRE 比此前最优方法（FPP）平均减少 **~28% 的额外 CNOT**。

---

## 8. 对我们项目的启示

1. **初始映射 + 路由 是两个可分离的子问题** → 我们可以用 GNN 分别优化
2. **启发式函数是 SABRE 的瓶颈** → 用 learned heuristic 替代是最小侵入的改进
3. **DAG 前沿层操作是核心数据结构** → `src/compiler/dag.py` 的设计要高效
4. **评估标准：CNOT 数 + 电路深度** → M2 的评估管线要实现这两个指标
