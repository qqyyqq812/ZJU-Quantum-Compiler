"""
M4 可视化：训练曲线 + 性能对比图
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history_dir: str = "models", output_dir: str = "results"):
    """绘制训练曲线。"""
    Path(output_dir).mkdir(exist_ok=True)
    history_files = list(Path(history_dir).glob("history_*.json"))

    if not history_files:
        print("⚠️ 没有找到训练历史文件")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hf in history_files:
        with open(hf) as f:
            history = json.load(f)

        name = hf.stem.replace('history_', '')
        episodes = range(len(history['episode_rewards']))

        # 平滑曲线
        window = min(20, len(history['episode_rewards']) // 5)
        if window > 0:
            rewards_smooth = np.convolve(
                history['episode_rewards'],
                np.ones(window) / window, mode='valid'
            )
            swaps_smooth = np.convolve(
                history['episode_swaps'],
                np.ones(window) / window, mode='valid'
            )
        else:
            rewards_smooth = history['episode_rewards']
            swaps_smooth = history['episode_swaps']

        axes[0].plot(rewards_smooth, label=name, alpha=0.8)
        axes[1].plot(swaps_smooth, label=name, alpha=0.8)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('SWAP Count')
    axes[1].set_title('SWAP Count per Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = Path(output_dir) / 'training_curves.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {fig_path}")
    plt.close()


def plot_comparison(results_path: str = "results/ai_vs_sabre.json",
                    output_dir: str = "results"):
    """绘制 AI vs SABRE 对比图。"""
    Path(output_dir).mkdir(exist_ok=True)

    try:
        with open(results_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 未找到 {results_path}")
        return

    # 按电路和拓扑分组
    sabre_cx = {}
    ai_cx = {}
    for r in data:
        key = f"{r['circuit_name']}@{r['topology_name']}"
        if 'sabre' in r['compiler_name']:
            sabre_cx[key] = r['compiled_cx']
        else:
            ai_cx[key] = r['compiled_cx']

    if not sabre_cx:
        print("⚠️ 对比数据不足")
        return

    # 柱状图
    keys = sorted(sabre_cx.keys())
    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, [sabre_cx.get(k, 0) for k in keys], width,
                   label='SABRE', color='#42A5F5')
    bars2 = ax.bar(x + width/2, [ai_cx.get(k, 0) for k in keys], width,
                   label='AI Router', color='#EF5350')

    ax.set_ylabel('Compiled CX Count')
    ax.set_title('AI Router vs SABRE: CNOT Count Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([k.split('@')[0] for k in keys], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = Path(output_dir) / 'ai_vs_sabre_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存: {fig_path}")
    plt.close()


def plot_ablation(ablation_path: str = "results/ablation.json",
                  output_dir: str = "results"):
    """绘制消融实验结果。"""
    Path(output_dir).mkdir(exist_ok=True)

    try:
        with open(ablation_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 未找到 {ablation_path}")
        return

    names = list(data.keys())
    rewards = [data[n]['avg_reward'] for n in names]
    swaps = [data[n]['avg_swaps'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh(names, rewards, color='#66BB6A')
    ax1.set_xlabel('Average Reward (last 50 episodes)')
    ax1.set_title('Ablation: Reward')

    ax2.barh(names, swaps, color='#FFA726')
    ax2.set_xlabel('Average SWAP Count (last 50 episodes)')
    ax2.set_title('Ablation: SWAP Count')

    plt.tight_layout()
    fig_path = Path(output_dir) / 'ablation_results.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✅ 消融实验图已保存: {fig_path}")
    plt.close()


def main():
    plot_training_curves()
    plot_comparison()
    plot_ablation()


if __name__ == '__main__':
    main()
