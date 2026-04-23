"""YAML 配置加载器 (V14).

遵循 `.claude/rules/code-and-config.md`:
- 所有 RL 超参必须从 yaml 读入
- 顶层必需 key: version, training, reward, environment
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_KEYS = ("version", "training", "reward", "environment")


def load_config(path: str | Path) -> dict[str, Any]:
    """加载并校验 yaml 配置。

    Args:
        path: 配置文件路径 (相对 repo 根或绝对路径)
    Returns:
        嵌套 dict

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 缺少必需顶层 key
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config {p} missing required top-level keys: {missing}")

    return cfg


def flatten_for_argparse(cfg: dict[str, Any]) -> dict[str, Any]:
    """将嵌套 yaml config 平铺为 train() 可用的单层 kwargs。

    约定: 嵌套路径 training.lr → flat lr
    冲突时（如多个子节下同名 key），后写入覆盖前写入。
    """
    flat: dict[str, Any] = {}
    for section_key, section_value in cfg.items():
        if section_key in {"version", "algorithm", "hardware", "curriculum", "paths"}:
            # 这些块保持嵌套，由 train.py 专门处理
            flat[section_key] = section_value
            continue
        if isinstance(section_value, dict):
            flat.update(section_value)
        else:
            flat[section_key] = section_value
    return flat
