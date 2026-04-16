import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results_dir = 'results/v12_tokyo20'
with open(f"{results_dir}/ai.json") as f:
    ai_data = json.load(f)
with open(f"{results_dir}/sabre.json") as f:
    sabre_data = json.load(f)

sabre_dict = {x['name']: x['sabre_swaps'] for x in sabre_data}
ai_dict = {x['name']: x['ai_swaps'] for x in ai_data}

keys = sorted(sabre_dict.keys())
x = np.arange(len(keys))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, [sabre_dict.get(k, 0) for k in keys], width, label='SABRE', color='#42A5F5')
ax.bar(x + width/2, [ai_dict.get(k, 0) for k in keys], width, label='AI Router (V12)', color='#EF5350')

ax.set_ylabel('Added SWAP Count')
ax.set_title('AI Router vs SABRE on 20Q (IBM Tokyo)')
ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = Path(results_dir) / 'v12_sabre_comparison.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✅ 图表已保存: {fig_path}")

# Markdown 总结
report = f"# V12 (IBM Tokyo 20Q) AI vs SABRE SWAP 对比\n\n| Circuit | SABRE_SWAPs | AI_SWAPs | 比率 |\n|---|---|---|---|\n"
total_ai = 0
total_sabre = 0
for k in keys:
    ai_v = ai_dict.get(k, 0)
    sa_v = sabre_dict.get(k, 0)
    ratio = f"{(ai_v / sa_v):.2f}x" if sa_v > 0 else "-"
    report += f"| {k} | {sa_v} | {ai_v} | {ratio} |\n"
    total_ai += ai_v
    total_sabre += sa_v
overall = total_ai / total_sabre if total_sabre > 0 else 0
report += f"| **总计** | **{total_sabre}** | **{total_ai}** | **{overall:.2f}x** |\n"
with open(f"{results_dir}/report.md", 'w') as f:
    f.write(report)
