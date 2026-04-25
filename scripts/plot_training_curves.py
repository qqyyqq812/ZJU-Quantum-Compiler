#!/usr/bin/env python3
"""Generate paper-grade training curve PNGs for the V14.2 / V9 comparison.

Outputs (under docs/figures/):
  1. v14_training_curve.png   : V14.2 SWAP curve + curriculum bg + eval AI vs SABRE
  2. v14_loss_curve.png       : V14.2 PPO policy / value losses (per update step)
  3. v9_v14_compare.png       : V9 vs V14.2 smoothed SWAPs (log-y) overlay
  4. v14_completion_rate.png  : V14.2 eval completion rate with stage boundaries

Run:
    cd /home/qq/projects/量子电路 && python scripts/plot_training_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path & style constants (no hard-coded paths in functions)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

V9_HISTORY = PROJECT_ROOT / "models" / "v9_tokyo20" / "history_v7_ibm_tokyo.json"
V14_HISTORY = PROJECT_ROOT / "models" / "v14_tokyo20" / "history_v7_ibm_tokyo.json"

OUT_V14_TRAINING = FIG_DIR / "v14_training_curve.png"
OUT_V14_LOSS = FIG_DIR / "v14_loss_curve.png"
OUT_V9_V14_COMPARE = FIG_DIR / "v9_v14_compare.png"
OUT_V14_COMPLETION = FIG_DIR / "v14_completion_rate.png"

DPI = 150

# Curriculum stages 0..4 -> distinct light shading colors
STAGE_COLORS = {
    0: "#FFF7CC",  # light yellow  - warmup
    1: "#D9F2D0",  # light green   - small
    2: "#CFE6FA",  # light blue    - medium
    3: "#E8D5F2",  # light purple  - large
    4: "#F8D2D2",  # light red     - master
}
STAGE_NAMES = {
    0: "Stage 0 (warmup)",
    1: "Stage 1 (small)",
    2: "Stage 2 (medium)",
    3: "Stage 3 (large)",
    4: "Stage 4 (master)",
}

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_history(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def smooth(values: Iterable[float], window: int) -> np.ndarray:
    """Centered moving-average smoothing using numpy.convolve.

    Pads the ends with edge values so the output keeps the input length and
    does not roll off near the boundaries.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), arr.size))
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def stage_segments(stages: list[int]):
    """Yield (start_ep, end_ep, stage_id) runs from a per-episode stage list."""
    if not stages:
        return
    arr = np.asarray(stages, dtype=int)
    start = 0
    cur = int(arr[0])
    for i in range(1, len(arr)):
        s = int(arr[i])
        if s != cur:
            yield start, i, cur
            start = i
            cur = s
    yield start, len(arr), cur


def shade_curriculum(ax, stages: list[int]) -> set[int]:
    """Shade axvspan blocks per curriculum stage. Returns the set of stage ids."""
    seen: set[int] = set()
    for s, e, st in stage_segments(stages):
        ax.axvspan(s, e, color=STAGE_COLORS.get(st, "#EEEEEE"), alpha=0.45, zorder=0)
        seen.add(st)
    return seen


def stage_legend_handles(stage_ids: Iterable[int]):
    return [
        plt.Rectangle(
            (0, 0), 1, 1,
            color=STAGE_COLORS.get(s, "#EEEEEE"),
            alpha=0.7,
            label=STAGE_NAMES.get(s, f"Stage {s}"),
        )
        for s in sorted(stage_ids)
    ]


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: V14 training curve (SWAPs + curriculum bg) + eval AI vs SABRE
# ---------------------------------------------------------------------------


def plot_v14_training(history: dict) -> Path:
    swaps = history.get("episode_swaps", [])
    stages = history.get("curriculum_stages", [])
    eval_ai = history.get("eval_swaps", [])
    eval_sabre = history.get("eval_sabre_swaps", [])

    if not swaps:
        raise ValueError("V14 history.episode_swaps is empty; cannot plot")

    n = len(swaps)
    eps = np.arange(n)
    win = 200

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    fig.suptitle(
        "V14.2 Training Curves — IBM Tokyo 20Q",
        fontsize=15, fontweight="bold", y=0.995,
    )

    # --- Top: episode_swaps raw + smoothed + curriculum stage bg
    stage_ids = shade_curriculum(ax_top, stages) if stages else set()
    ax_top.plot(eps, swaps, color="#888888", alpha=0.25, linewidth=0.6,
                label="raw SWAPs / episode")
    ax_top.plot(eps, smooth(swaps, win), color="#A23B25", linewidth=2.0,
                label=f"moving avg (window={win})")

    # Cap y at 99th percentile so outliers don't crush the plot
    y_cap = float(np.percentile(swaps, 99))
    if y_cap > 0:
        ax_top.set_ylim(-0.5, y_cap * 1.15)
    ax_top.set_ylabel("SWAP count / Episode")
    ax_top.set_title("Episode SWAP Count with Curriculum Stages")

    handles, labels = ax_top.get_legend_handles_labels()
    handles += stage_legend_handles(stage_ids)
    ax_top.legend(handles=handles, loc="upper right", fontsize=9, ncol=2)

    # --- Bottom: eval_swaps vs eval_sabre_swaps as two lines
    n_eval = min(len(eval_ai), len(eval_sabre))
    if n_eval > 0:
        # Spread eval points across the same x-axis so the two subplots align
        eval_x = np.linspace(0, n - 1, n_eval) if n_eval > 1 else np.array([n - 1])
        ai = np.asarray(eval_ai[:n_eval], dtype=float)
        sabre = np.asarray(eval_sabre[:n_eval], dtype=float)
        ax_bot.plot(eval_x, ai, "-o", color="#1F4E79", linewidth=1.8,
                    markersize=5, label="AI compiler (V14.2)")
        ax_bot.plot(eval_x, sabre, "-s", color="#D98C3F", linewidth=1.8,
                    markersize=5, label="SABRE baseline")
        ax_bot.legend(loc="upper right", fontsize=10)
    else:
        ax_bot.text(0.5, 0.5, "no eval_swaps / eval_sabre_swaps data",
                    ha="center", va="center", transform=ax_bot.transAxes,
                    fontsize=11, color="#888")

    ax_bot.set_xlabel("Episode")
    ax_bot.set_ylabel("Mean eval SWAPs")
    ax_bot.set_title("Eval Set: AI Compiler vs SABRE")

    plt.tight_layout()
    fig.savefig(OUT_V14_TRAINING, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return OUT_V14_TRAINING


# ---------------------------------------------------------------------------
# Figure 2: V14 PPO loss (policy + value, per update step)
# ---------------------------------------------------------------------------


def plot_v14_loss(history: dict) -> Path:
    policy = np.asarray(history.get("policy_losses", []), dtype=float)
    value = np.asarray(history.get("value_losses", []), dtype=float)

    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False,
        gridspec_kw={"hspace": 0.32},
    )
    fig.suptitle(
        "V14.2 PPO Losses (per update step)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    if policy.size:
        x = np.arange(policy.size)
        win_p = max(5, policy.size // 50)
        ax_p.plot(x, policy, color="#9CB7D4", alpha=0.35, linewidth=0.7, label="raw")
        ax_p.plot(x, smooth(policy, win_p), color="#1F4E79", linewidth=2.0,
                  label=f"moving avg (w={win_p})")
        ax_p.legend(loc="upper right", fontsize=9)
    else:
        ax_p.text(0.5, 0.5, "no policy_losses data",
                  ha="center", va="center", transform=ax_p.transAxes, color="#888")
    ax_p.set_title("Policy Loss")
    ax_p.set_xlabel("Update step")
    ax_p.set_ylabel("Policy loss")

    if value.size:
        x = np.arange(value.size)
        win_v = max(5, value.size // 50)
        ax_v.plot(x, value, color="#E8B4A0", alpha=0.35, linewidth=0.7, label="raw")
        ax_v.plot(x, smooth(value, win_v), color="#A23B25", linewidth=2.0,
                  label=f"moving avg (w={win_v})")
        ax_v.legend(loc="upper right", fontsize=9)
    else:
        ax_v.text(0.5, 0.5, "no value_losses data",
                  ha="center", va="center", transform=ax_v.transAxes, color="#888")
    ax_v.set_title("Value Loss")
    ax_v.set_xlabel("Update step")
    ax_v.set_ylabel("Value loss")

    plt.tight_layout()
    fig.savefig(OUT_V14_LOSS, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return OUT_V14_LOSS


# ---------------------------------------------------------------------------
# Figure 3: V9 vs V14 smoothed SWAPs comparison (single subplot, log-y)
# ---------------------------------------------------------------------------


def plot_v9_v14_compare(v9: dict, v14: dict) -> Path:
    v9_swaps = np.asarray(v9.get("episode_swaps", []), dtype=float)
    v14_swaps = np.asarray(v14.get("episode_swaps", []), dtype=float)

    if v9_swaps.size == 0 or v14_swaps.size == 0:
        raise ValueError("episode_swaps missing for v9 or v14")

    win = 500
    v9_smooth = smooth(v9_swaps, win)
    v14_smooth = smooth(v14_swaps, win)

    # Replace zeros / non-positives with a small epsilon so log scale is safe
    eps = 1e-1
    v9_smooth = np.clip(v9_smooth, eps, None)
    v14_smooth = np.clip(v14_smooth, eps, None)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(v9_smooth.size), v9_smooth,
            color="#C0392B", linewidth=2.0,
            label=f"V9 — PPO baseline (n={v9_swaps.size:,} eps)")
    ax.plot(np.arange(v14_smooth.size), v14_smooth,
            color="#1F8F4A", linewidth=2.0,
            label=("V14.2 — +SABRE cache, staged mask, real integration "
                   f"(n={v14_swaps.size:,} eps)"))

    ax.set_yscale("log")
    ax.set_xlabel("Episode")
    ax.set_ylabel("SWAP count (log scale)")
    ax.set_title(
        f"V9 -> V14.2 Episode SWAP Trajectory (moving avg, window={win})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_V9_V14_COMPARE, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return OUT_V9_V14_COMPARE


# ---------------------------------------------------------------------------
# Figure 4: V14 eval completion rate with stage boundaries
# ---------------------------------------------------------------------------


def plot_v14_completion(history: dict) -> Path:
    completion = np.asarray(history.get("eval_completion", []), dtype=float)
    stages = history.get("curriculum_stages", [])
    n_eps = len(history.get("episode_swaps", [])) or len(stages)

    if completion.size == 0:
        raise ValueError("eval_completion missing or empty in V14 history")

    fig, ax = plt.subplots(figsize=(12, 6))

    n_eval = completion.size
    eval_x = np.arange(1, n_eval + 1)
    ax.plot(eval_x, completion, "-o", color="#1F4E79", linewidth=1.8,
            markersize=6, label="V14.2 eval completion")
    ax.axhline(1.0, color="#1F8F4A", linestyle="--", linewidth=1.2,
               alpha=0.7, label="100% completion")

    # Map curriculum stage boundaries (in episode space) onto eval-step space.
    # Eval rounds are uniformly spaced across training episodes, so episode i
    # corresponds to eval index i * n_eval / n_eps (1-based after rounding).
    stage_legend_seen: set[int] = set()
    if stages and n_eps > 0 and n_eval > 0:
        for s_ep, e_ep, st in stage_segments(stages):
            s_eval = s_ep / n_eps * n_eval
            e_eval = e_ep / n_eps * n_eval
            ax.axvspan(s_eval, e_eval,
                       color=STAGE_COLORS.get(st, "#EEEEEE"),
                       alpha=0.35, zorder=0)
            stage_legend_seen.add(st)

        # Mark stage transition lines for clarity
        boundaries = []
        prev = stages[0]
        for i, st in enumerate(stages):
            if st != prev:
                boundaries.append((i, prev, st))
                prev = st
        for ep_idx, _, new_st in boundaries:
            x_pos = ep_idx / n_eps * n_eval
            ax.axvline(x_pos, color="#444", linestyle=":", linewidth=1.0,
                       alpha=0.6)
            ax.text(x_pos, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] else 0.02,
                    f"->S{new_st}", rotation=90, va="bottom", ha="right",
                    fontsize=8, color="#444")

    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Eval index")
    ax.set_ylabel("Completion rate")
    ax.set_title("V14.2 Eval Completion Rate (with curriculum stage shading)",
                 fontsize=14, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    handles += stage_legend_handles(stage_legend_seen)
    ax.legend(handles=handles, loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout()
    fig.savefig(OUT_V14_COMPLETION, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return OUT_V14_COMPLETION


# ---------------------------------------------------------------------------
# Public top-level API (importable)
# ---------------------------------------------------------------------------


def plot_v14() -> tuple[Path, Path, Path]:
    """Generate the three V14-only plots."""
    ensure_dir()
    v14 = load_history(V14_HISTORY)
    return (
        plot_v14_training(v14),
        plot_v14_loss(v14),
        plot_v14_completion(v14),
    )


def plot_v9_v14_comparison() -> Path:
    ensure_dir()
    v9 = load_history(V9_HISTORY)
    v14 = load_history(V14_HISTORY)
    return plot_v9_v14_compare(v9, v14)


def main() -> None:
    ensure_dir()

    print(f"[plot] loading V9  -> {V9_HISTORY}")
    v9 = load_history(V9_HISTORY)
    print(f"[plot] loading V14 -> {V14_HISTORY}")
    v14 = load_history(V14_HISTORY)

    outputs = [
        plot_v14_training(v14),
        plot_v14_loss(v14),
        plot_v9_v14_compare(v9, v14),
        plot_v14_completion(v14),
    ]

    print("[plot] generated:")
    for p in outputs:
        size_kb = p.stat().st_size / 1024.0
        print(f"  {p}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
