"""
训练曲线可视化工具
==================
从训练日志 JSON 文件中绘制 reward/SWAP/课程升级曲线。

用法:
    python -m src.benchmarks.plot_training models/v7_final_v2/history_v7_linear_5.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np


def smooth(data: list[float], window: int = 100) -> np.ndarray:
    """滑动窗口平滑。"""
    arr = np.array(data)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def plot_training_curves(history_path: str, output_dir: str = "results"):
    """绘制完整训练曲线图。"""
    with open(history_path) as f:
        history = json.load(f)
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('V7.2 Quantum Router Training Curves', fontsize=16, fontweight='bold')
    
    # 1. 奖励曲线
    ax = axes[0, 0]
    rewards = history.get('episode_rewards', [])
    if rewards:
        ax.plot(smooth(rewards, 200), color='#2196F3', linewidth=0.8)
        ax.set_title('Episode Reward (smoothed)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    # 2. SWAP 数曲线
    ax = axes[0, 1]
    swaps = history.get('episode_swaps', [])
    if swaps:
        ax.plot(smooth(swaps, 200), color='#FF5722', linewidth=0.8)
        ax.set_title('Episode SWAP Count (smoothed)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('SWAPs')
        ax.grid(True, alpha=0.3)
    
    # 3. 评估 SWAP (每 eval_interval 个)
    ax = axes[1, 0]
    eval_swaps = history.get('eval_swaps', [])
    if eval_swaps:
        x_eval = [(i + 1) * 1000 for i in range(len(eval_swaps))]
        ax.plot(x_eval, eval_swaps, 'o-', color='#4CAF50', markersize=4, linewidth=1.5)
        ax.set_title('Evaluation avg_swap')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg SWAPs')
        ax.grid(True, alpha=0.3)
    
    # 4. 课程阶段
    ax = axes[1, 1]
    stages = history.get('curriculum_stages', [])
    if stages:
        ax.plot(stages, color='#9C27B0', linewidth=0.5)
        ax.set_title('Curriculum Stage')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Stage')
        ax.set_yticks(range(max(int(max(stages)) + 1, 3)))
        ax.grid(True, alpha=0.3)
    
    # 5. 损失曲线 (如有)
    # policy_losses/value_losses 放到子图的 twin axis 或跳过
    
    plt.tight_layout()
    save_path = out / 'training_curves.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 训练曲线已保存: {save_path}")
    return str(save_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python -m src.benchmarks.plot_training <history.json> [output_dir]")
        sys.exit(1)
    
    hist_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    plot_training_curves(hist_path, out_dir)
