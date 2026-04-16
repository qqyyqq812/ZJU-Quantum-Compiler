# tests/ — 自动化测试区

> **职责**：所有自动化测试代码。使用 pytest 框架。

## 组织方式

```
tests/
├── test_dag.py          # DAG 操作测试
├── test_mapping.py      # 映射算法测试
├── test_routing.py      # 路由算法测试
├── test_benchmarks.py   # 基准测试正确性
└── conftest.py          # pytest fixtures（共享测试数据）
```

## 运行方式

```bash
# 在项目根目录执行
pytest tests/ -v
```

## 测试原则

- 每个 `src/` 模块必须有对应的测试文件
- 测试函数命名：`test_功能描述`
- 使用 fixture 管理测试数据（标准量子电路和 coupling map）
