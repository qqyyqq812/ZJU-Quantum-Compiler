"""
统一训练入口 — 通过 configs/*.yaml 加载超参数

用法:
    python train.py --config configs/v9_baseline.yaml
    python train.py --config configs/v13_gpu.yaml
"""
import argparse
import yaml
import sys
import os

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="量子电路 AI 编译器训练入口")
    parser.add_argument(
        "--config",
        default="configs/v9_baseline.yaml",
        help="训练配置文件路径（默认: configs/v9_baseline.yaml）"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    print(f"[INFO] 加载配置: {args.config}")
    print(f"[INFO] 算法: {config.get('algorithm')}, 环境: {config.get('environment')}")
    print(f"[INFO] episodes={config.get('episodes')}, rollout_steps={config.get('rollout_steps')}")

    # 调用 src.compiler.train 主逻辑
    from src.compiler.train import main as train_main
    train_main(config)

if __name__ == "__main__":
    main()
