# src/ — 核心源码区

> **职责**：存放项目的所有生产代码。严禁存放调试脚本或临时文件。

## 目录结构

```
src/
├── compiler/          # 量子电路编译器核心
│   ├── __init__.py
│   ├── mapping.py     # 初始映射算法（逻辑比特→物理比特）
│   ├── routing.py     # 路由算法（SWAP门插入策略）
│   ├── dag.py         # DAG 电路表示与操作
│   └── pass_manager.py # Qiskit TranspilerPass 集成接口
├── benchmarks/        # 基准测试
│   ├── __init__.py
│   ├── circuits.py    # 标准测试电路生成
│   └── evaluate.py    # 性能评估指标
└── visualization/     # 可视化
    ├── __init__.py
    └── plot.py        # 电路/拓扑/结果可视化
```

## 开发原则

- 每个模块独立可测试
- 所有公共函数必须有 docstring
- 使用 type hints
