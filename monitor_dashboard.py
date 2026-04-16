import streamlit as st
import json
import time
import os
import pandas as pd

st.set_page_config(page_title="量子电路训练监控", page_icon="🌌", layout="wide")

DATA_PATH = "/root/projects/量子电路/models/v12_tokyo20/history_v7_ibm_tokyo.json"
LOG_PATH = "/root/projects/量子电路/logs/training_v12_ray.log"

placeholder = st.empty()

while True:
    with placeholder.container():
        st.title("🌌 量子电路 PPO 训练监控 (V12 满血版)")

        # --- 读取数据 ---
        try:
            with open(DATA_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            st.error("数据文件加载失败，等待中...")
            time.sleep(3)
            continue

        eval_swaps = data.get("eval_swaps", [])
        eval_sabre = data.get("eval_sabre_swaps", [])
        eval_comp = data.get("eval_completion", [])
        rewards = data.get("episode_rewards", [])
        n = len(rewards)

        if n == 0:
            st.info("等待首轮数据...")
            time.sleep(3)
            continue

        # --- 读取日志最后一行，获取真实进度 ---
        log_ep = n
        try:
            with open(LOG_PATH, "r") as lf:
                lines = lf.readlines()
            for line in reversed(lines):
                if "/50000]" in line:
                    part = line.split("/50000]")[0]
                    num = part.strip().split("[")[-1].strip()
                    log_ep = int(num)
                    break
        except Exception:
            pass

        # --- JSON 文件最后修改时间 ---
        try:
            mtime = os.path.getmtime(DATA_PATH)
            age = int(time.time() - mtime)
        except Exception:
            age = -1

        # --- 状态判定 ---
        if age >= 0 and age < 120:
            status_text = "🟢 正常运行中（数据 %d 秒前更新）" % age
            status_color = "green"
        elif age >= 120 and age < 600:
            status_text = "🟡 可能卡顿（数据 %d 秒未更新）" % age
            status_color = "orange"
        else:
            status_text = "🔴 疑似停止（数据超 %d 秒未更新）" % age
            status_color = "red"

        # --- 核心数字 ---
        best_swap = min(eval_swaps) if eval_swaps else 0
        latest_swap = eval_swaps[-1] if eval_swaps else 0
        latest_sabre = eval_sabre[-1] if eval_sabre else 0
        comp = eval_comp[-1] * 100 if eval_comp else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 已完成局数", "%d" % n)
        c2.metric("🏆 最新 SWAP", "%.1f" % latest_swap, delta=f"vs SABRE: {latest_swap - latest_sabre:.1f}", delta_color="inverse")
        c3.metric("✅ 路由完成率", "%.0f%%" % comp)

        # --- 进度条 ---
        st.progress(min(1.0, log_ep / 50000.0))
        st.caption("训练进度：%d / 50000（约 %.1f%%）" % (log_ep, log_ep / 500.0))

        # --- 运行状态 ---
        st.markdown("### 运行状态")
        st.markdown("**%s**" % status_text)

        # --- 唯一核心图表：Eval SWAP 收敛曲线 ---
        st.markdown("### 📈 SWAP 代价收敛趋势（越低越好）")
        if eval_swaps:
            if eval_sabre and len(eval_sabre) == len(eval_swaps):
                df = pd.DataFrame({
                    "🚀 V12 AI": eval_swaps,
                    "📉 Qiskit SABRE 基线": eval_sabre
                })
            else:
                df = pd.DataFrame({"🚀 V12 AI": eval_swaps})
            st.line_chart(df, use_container_width=True)

        st.caption("每 5 秒自动刷新 | 当前时间：%s" % time.strftime("%H:%M:%S"))

    time.sleep(5)
