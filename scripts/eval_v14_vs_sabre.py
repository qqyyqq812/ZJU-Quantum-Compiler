#!/usr/bin/env python
"""V14 评测脚本: AI Router vs SABRE 对比。

遵循 .claude/rules/experiment-log.md:
- 生成 eval_report_v<N>.md 到 models/v<N>_<topology>/
- 不写入 docs/，不污染项目结构

用法:
    python scripts/eval_v14_vs_sabre.py --model models/v14_tokyo20/v7_ibm_tokyo_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qiskit import QuantumCircuit, transpile

# 本地导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmarks.circuits import generate_qft, generate_qaoa, generate_random
from src.benchmarks.topologies import get_topology
from src.compiler.pass_manager import AIRouter


BENCHMARK_CIRCUITS = [
    ("QFT-5", lambda n=5: generate_qft(5)),
    ("QFT-10", lambda n=10: generate_qft(10)),
    ("QFT-20", lambda n=20: generate_qft(20)),
    ("QAOA-5", lambda n=5: generate_qaoa(5, p=1)),
    ("QAOA-10", lambda n=10: generate_qaoa(10, p=1)),
    ("Rand-5-20", lambda n=5: generate_random(5, depth=20, seed=42)),
    ("Rand-10-40", lambda n=10: generate_random(10, depth=40, seed=42)),
    ("Rand-20-100", lambda n=20: generate_random(20, depth=100, seed=42)),
]


def eval_single(
    router: AIRouter,
    circuit: QuantumCircuit,
    topology_cm,
    max_steps: int = 2000,
) -> dict:
    """评测单个电路: 返回 AI SWAP 数 + SABRE SWAP 数。"""
    out = router.route_count_only(circuit, max_steps=max_steps)
    ai_swaps = out["ai_swaps"]
    done = out["done"]

    # SABRE baseline
    try:
        sabre_out = transpile(
            circuit,
            coupling_map=topology_cm,
            optimization_level=1,
            routing_method="sabre",
            seed_transpiler=42,
        )
        sabre_swaps = sabre_out.count_ops().get("swap", 0)
    except Exception as e:
        sabre_swaps = -1

    relative = (
        f"{(ai_swaps - sabre_swaps) / max(sabre_swaps, 1) * 100:+.1f}%"
        if sabre_swaps > 0
        else "N/A"
    )
    return {
        "ai_swaps": ai_swaps,
        "sabre_swaps": sabre_swaps,
        "relative": relative,
        "done": done,
        "ai_trace_events": out["trace_len"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path/to/v7_*_best.pt")
    parser.add_argument("--topology", default="ibm_tokyo")
    parser.add_argument("--version", default="v14", help="Output report version tag")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    model_path = Path(args.model)
    cm = get_topology(args.topology)
    if model_path.is_file():
        router = AIRouter(cm, model_path=str(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARN: model not found at {model_path}, falling back to random policy")
        router = AIRouter(cm, model_path=None)
    print(f"Topology: {args.topology} ({cm.size()} qubits)")
    print()

    results = []
    for name, builder in BENCHMARK_CIRCUITS:
        try:
            qc = builder()
        except Exception as e:
            print(f"  Skip {name}: {e}")
            continue
        if qc.num_qubits > cm.size():
            continue

        r = eval_single(router, qc, cm, max_steps=args.max_steps)
        r["circuit"] = name
        r["qubits"] = qc.num_qubits
        r["gates"] = qc.size()
        results.append(r)

        done_mark = "OK" if r["done"] else "INCOMPLETE"
        print(
            f"  {name:12s} qubits={qc.num_qubits:>2d} gates={qc.size():>4d} "
            f"AI={r['ai_swaps']:>3d} SABRE={r['sabre_swaps']:>3d} "
            f"rel={r['relative']:>7s} [{done_mark}]"
        )

    # 写 eval_report
    out_dir = model_path.parent
    report_path = out_dir / f"eval_report_{args.version}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {args.version.upper()} 评测报告\n\n")
        f.write(f"- **模型**: `{model_path}`\n")
        f.write(f"- **拓扑**: {args.topology} ({cm.size()} qubits)\n")
        f.write(f"- **最大路由步数**: {args.max_steps}\n\n")
        f.write(f"## SABRE 对比\n\n")
        f.write(
            "| 电路 | Qubits | Gates | AI SWAP | SABRE SWAP | 相对 | 完成 |\n"
        )
        f.write("|------|--------|-------|---------|------------|------|------|\n")
        for r in results:
            f.write(
                f"| {r['circuit']} | {r['qubits']} | {r['gates']} | "
                f"{r['ai_swaps']} | {r['sabre_swaps']} | {r['relative']} | "
                f"{'OK' if r['done'] else 'N'} |\n"
            )

        # 汇总
        n_wins = sum(
            1 for r in results if r["sabre_swaps"] > 0 and r["ai_swaps"] < r["sabre_swaps"]
        )
        n_ties = sum(
            1 for r in results if r["sabre_swaps"] > 0 and r["ai_swaps"] == r["sabre_swaps"]
        )
        n_total = sum(1 for r in results if r["sabre_swaps"] > 0)
        f.write(
            f"\n## 汇总\n\n"
            f"- 超越 SABRE: **{n_wins}/{n_total}** ({n_wins/max(n_total,1)*100:.0f}%)\n"
            f"- 持平: {n_ties}\n"
            f"- 落后: {n_total - n_wins - n_ties}\n"
        )
        f.write(
            f"\n## 与 V13 的差异 (参照 decisions.md §V14)\n\n"
            f"- V14 §V14-1: SABRE baseline 缓存生效\n"
            f"- V14 §V14-2: 阶段化 Mask 生效\n"
            f"- V14 §V14-3: 奖励分层生效\n"
            f"- V14 §V14-4: pass_manager 真集成 — 本报告的 AI SWAP 数可被外部复现\n"
        )

    print()
    print(f"Report written to: {report_path}")

    # 同时写 JSON
    json_path = out_dir / f"eval_report_{args.version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
