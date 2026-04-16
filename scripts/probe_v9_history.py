#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9 History Diagnostic Probe
用于反向诊断跑崩的 JSON 训练日志，解析病理特征。
"""
import json
import statistics

def analyze_history(json_path):
    print(f"🔬 正在解剖: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stages = data.get('curriculum_stages', [])
    swaps = data.get('episode_swaps', [])
    rewards = data.get('episode_rewards', [])
    v_losses = data.get('value_losses', [])
    p_losses = data.get('policy_losses', [])

    if not stages:
        print("❌ 未找到有效数据！")
        return

    n_episodes = len(stages)
    print(f"📊 总局数: {n_episodes}")
    
    # 按照 Stage 进行分组统计
    stage_stats = {}
    for i in range(n_episodes):
        st = stages[i]
        if st not in stage_stats:
            stage_stats[st] = {'swaps': [], 'rewards': [], 'v_loss': [], 'p_loss': [], 'episodes': 0}
        
        stage_stats[st]['episodes'] += 1
        if i < len(swaps): stage_stats[st]['swaps'].append(swaps[i])
        if i < len(rewards): stage_stats[st]['rewards'].append(rewards[i])
        if i < len(v_losses): stage_stats[st]['v_loss'].append(v_losses[i])
        if i < len(p_losses): stage_stats[st]['p_loss'].append(p_losses[i])

    print("\n📈 各 Curriculum Stage 表现:")
    for st in sorted(stage_stats.keys()):
        stat = stage_stats[st]
        eps = stat['episodes']
        avg_swap = sum(stat['swaps'])/len(stat['swaps']) if stat['swaps'] else 0
        avg_rew = sum(stat['rewards'])/len(stat['rewards']) if stat['rewards'] else 0
        avg_v = sum(stat['v_loss'])/len(stat['v_loss']) if stat['v_loss'] else 0
        avg_p = sum(stat['p_loss'])/len(stat['p_loss']) if stat['p_loss'] else 0
        print(f"  Stage {st}: 跑了 {eps} 局")
        print(f"    -> Avg SWAP:   {avg_swap:.2f}")
        print(f"    -> Avg Reward: {avg_rew:.2f}")
        print(f"    -> Avg V_Loss: {avg_v:.4f}")
        print(f"    -> Avg P_Loss: {avg_p:.4f}")
        
    # 末尾 100 局的状态（看是否真的死锁）
    if len(swaps) >= 100:
        recent_swaps = swaps[-100:]
        print("\n⏳ 最后 100 局死锁特征:")
        print(f"  平均 SWAP: {sum(recent_swaps)/100:.2f}")
        print(f"  最高/最低: {max(recent_swaps)} / {min(recent_swaps)}")
        print(f"  标准差 (震荡剧烈度): {statistics.stdev(recent_swaps) if len(recent_swaps) > 1 else 0:.2f}")

if __name__ == '__main__':
    analyze_history('/home/qq/projects/量子电路/models/v9_tokyo20/history_v7_ibm_tokyo.json')
