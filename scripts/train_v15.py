"""V15 training entry point.

Usage:
    python scripts/train_v15.py --config configs/v15_baseline.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.benchmarks.topologies import get_topology
from src.compiler.v15.train import run_training


def _setup_logging(save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_v15.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    # 压制 qiskit transpiler 的高频 pass-level 日志（每条电路触发数十条）
    for noisy in ("qiskit.passmanager", "qiskit.compiler", "qiskit.transpiler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v15_baseline.yaml",
        help="Path to V15 yaml config",
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    save_dir = Path(cfg["paths"]["save_dir"])
    _setup_logging(save_dir)

    log = logging.getLogger("train_v15")
    log.info("=" * 60)
    log.info("V15 AlphaZero-MCTS+GNN training")
    log.info("config: %s", args.config)
    log.info("=" * 60)

    topology_name = cfg["topology"]["name"]
    coupling_map = get_topology(topology_name)

    history = run_training(cfg, coupling_map)
    log.info("Training complete. Final stage: %d", history["stage"][-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
