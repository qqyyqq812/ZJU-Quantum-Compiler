#!/usr/bin/env python
"""MQT-Bench 评测管线 (V14 验收 #6: MQT Bench 覆盖 ≥ 10 条官方电路)。

对每条电路跑 3 条流水线：
1. **No-Routing baseline** — 只 transpile 到 basis_gates，不做 routing
                           (理论下限，硬件不可执行 — 用于看 routing 引入的 SWAP)
2. **SABRE baseline** — qiskit 默认 routing_method='sabre'
3. **AI Router** — 项目训练好的 PPO+GNN 路由器（如果模型存在）

输出 markdown 报告到 `models/v14_tokyo20/eval_report_mqt.md`，
符合 `.claude/rules/experiment-log.md` 格式（metadata + 表格 + 与 V13 差异段）。

用法:
    python scripts/eval_mqt_bench.py
    python scripts/eval_mqt_bench.py --ai-model models/v14_tokyo20/v7_ibm_tokyo_best.pt
    python scripts/eval_mqt_bench.py --n-qubits 5,10 --output models/v14_tokyo20/eval_report_mqt.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根目录 import 优先
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit, transpile

from src.benchmarks.mqt_bench import (
    DEFAULT_BASIS_GATES,
    describe_circuits,
    get_mqt_circuits,
    is_mqt_available,
)
from src.benchmarks.topologies import get_topology, get_topology_info

# logger 配置（遵守 .claude/rules/code-and-config.md：禁止 print 在 src/，
# 但这是 scripts/ 下的 CLI 入口，print 用于直接给用户反馈，logger 用于诊断）
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_mqt_bench")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------
def run_no_routing(qc: QuantumCircuit, basis_gates: list[str]) -> dict:
    """Pipeline 1: 仅 basis_gates 分解，不路由（理论下限）。"""
    t0 = time.perf_counter()
    try:
        qc_out = transpile(
            qc,
            basis_gates=basis_gates,
            optimization_level=0,
            seed_transpiler=42,
        )
        ops = dict(qc_out.count_ops())
        return {
            "swaps": 0,  # 没做 routing，所以 SWAP=0（但电路硬件不可执行）
            "depth": qc_out.depth(),
            "n_2q": sum(c for op, c in ops.items() if op in ("cx", "cz", "swap")),
            "completed": True,
            "elapsed_s": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        logger.warning("no-routing 失败 (%s): %s", qc.name, exc)
        return {
            "swaps": -1, "depth": -1, "n_2q": -1, "completed": False,
            "elapsed_s": time.perf_counter() - t0, "error": str(exc),
        }


def run_sabre(qc: QuantumCircuit, coupling_map, basis_gates: list[str]) -> dict:
    """Pipeline 2: qiskit SABRE routing（业界基线）。

    注: optimization_level=0 + layout_method='trivial' 是为了 **不让 qiskit 通过初始
    layout 优化把 SWAP 消掉** —— 我们要测的是 routing 本身的代价，不是
    initial-layout 选得好不好。否则对密集 CNOT 电路（QFT/QAOA）经常出现 SABRE=0。
    """
    t0 = time.perf_counter()
    try:
        qc_out = transpile(
            qc,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=0,
            layout_method="trivial",
            routing_method="sabre",
            seed_transpiler=42,
        )
        ops = dict(qc_out.count_ops())
        return {
            "swaps": ops.get("swap", 0),
            "depth": qc_out.depth(),
            "n_2q": sum(c for op, c in ops.items() if op in ("cx", "cz", "swap")),
            "completed": True,
            "elapsed_s": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        logger.warning("SABRE 失败 (%s): %s", qc.name, exc)
        return {
            "swaps": -1, "depth": -1, "n_2q": -1, "completed": False,
            "elapsed_s": time.perf_counter() - t0, "error": str(exc),
        }


def run_ai(qc: QuantumCircuit, router, max_steps: int) -> dict:
    """Pipeline 3: AI 路由器（项目自研）。"""
    t0 = time.perf_counter()
    try:
        result = router.route_count_only(qc, max_steps=max_steps)
        return {
            "swaps": result["ai_swaps"],
            "depth": -1,  # route_count_only 不构建电路，不返回 depth
            "n_2q": result["executed_gates"],
            "completed": result["done"],
            "elapsed_s": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        logger.warning("AI router 失败 (%s): %s", qc.name, exc)
        return {
            "swaps": -1, "depth": -1, "n_2q": -1, "completed": False,
            "elapsed_s": time.perf_counter() - t0, "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Router loading（容错：模型不存在或加载失败 → 只跑 SABRE 部分）
# ---------------------------------------------------------------------------
def try_load_router(model_path: Path | None, coupling_map):
    """尝试加载 AIRouter；失败时返回 None 并 log 警告。"""
    if model_path is None:
        logger.info("未指定 --ai-model，跳过 AI 评测，只跑 baseline + SABRE。")
        return None
    if not model_path.exists():
        logger.warning(
            "AI 模型文件不存在: %s — 跳过 AI 评测，只跑 baseline + SABRE。",
            model_path,
        )
        return None
    try:
        from src.compiler.pass_manager import AIRouter
        router = AIRouter(coupling_map, model_path=str(model_path))
        if not router._has_model:
            logger.warning("AIRouter 实例化成功但模型未加载，跳过 AI 评测。")
            return None
        logger.info("AIRouter 加载成功: %s", model_path)
        return router
    except Exception as exc:
        logger.warning(
            "加载 AIRouter 失败（架构不兼容？）: %s — 跳过 AI 评测，只跑 SABRE。",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_markdown_report(
    output_path: Path,
    results: list[dict],
    metadata: dict,
) -> None:
    """生成 eval_report_mqt.md（遵守 .claude/rules/experiment-log.md 格式）。"""
    has_ai = any(r.get("ai") is not None for r in results)

    lines = []
    lines.append("# V14 MQT-Bench 评测报告")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **运行时间**: {metadata['timestamp']}")
    lines.append(f"- **拓扑**: {metadata['topology']} ({metadata['topology_info']['n_qubits']} qubits, "
                 f"diameter={metadata['topology_info']['diameter']})")
    lines.append(f"- **电路数**: {len(results)}（覆盖 {metadata['n_circuit_types']} 种类型 × "
                 f"{len(metadata['n_qubits_list'])} 种规模）")
    lines.append(f"- **MQT-Bench 状态**: {'已安装 v2.x' if metadata['mqt_available'] else '未安装（fallback 到项目自带电路）'}")
    lines.append(f"- **AI 模型**: `{metadata.get('ai_model_path', '未提供')}` — "
                 f"{'已加载' if has_ai else '未加载（仅 SABRE 评测）'}")
    lines.append(f"- **basis_gates**: {metadata['basis_gates']}")
    lines.append(f"- **max_steps (AI router)**: {metadata['max_steps']}")
    lines.append("")

    # 主表格 — V14 P1 标准格式
    lines.append("## 路由 SWAP 对比")
    lines.append("")
    lines.append("| Circuit | n_qubits | SABRE SWAP | SABRE time(ms) | AI SWAP | AI time(ms) | AI/SABRE ratio |")
    lines.append("|---------|----------|------------|----------------|---------|-------------|----------------|")

    for r in results:
        sabre = r["sabre"]
        sabre_swap = sabre["swaps"] if sabre["completed"] else "FAIL"
        sabre_ms = f"{sabre['elapsed_s'] * 1000:.2f}" if sabre["completed"] else "N/A"

        if r["ai"] is not None and r["ai"]["completed"]:
            ai_swap = r["ai"]["swaps"]
            ai_ms = f"{r['ai']['elapsed_s'] * 1000:.2f}"
            if sabre["completed"] and sabre["swaps"] > 0:
                ratio = ai_swap / sabre["swaps"]
                ratio_str = f"{ratio:.3f}"
            elif sabre["completed"] and sabre["swaps"] == 0 and ai_swap == 0:
                ratio_str = "1.000"
            else:
                ratio_str = "N/A"
        else:
            ai_swap = "N/A"
            ai_ms = "N/A"
            ratio_str = "N/A"

        lines.append(
            f"| {r['name']} | {r['qubits']} | {sabre_swap} | {sabre_ms} | "
            f"{ai_swap} | {ai_ms} | {ratio_str} |"
        )

    lines.append("")

    # 汇总统计
    lines.append("## 汇总")
    lines.append("")
    n_ok_sabre = sum(1 for r in results if r["sabre"]["completed"])
    avg_sabre = (
        sum(r["sabre"]["swaps"] for r in results if r["sabre"]["completed"])
        / max(n_ok_sabre, 1)
    )
    lines.append(f"- SABRE 完成率: **{n_ok_sabre}/{len(results)}** "
                 f"({n_ok_sabre / max(len(results), 1) * 100:.0f}%)")
    lines.append(f"- SABRE 平均 SWAP: **{avg_sabre:.1f}**")
    if has_ai:
        n_ok_ai = sum(1 for r in results if r["ai"] and r["ai"]["completed"])
        n_wins = sum(
            1 for r in results
            if r["ai"] and r["ai"]["completed"] and r["sabre"]["completed"]
            and r["ai"]["swaps"] < r["sabre"]["swaps"]
        )
        comparable = sum(
            1 for r in results
            if r["ai"] and r["ai"]["completed"] and r["sabre"]["completed"]
        )
        lines.append(f"- AI 完成率: **{n_ok_ai}/{len(results)}** "
                     f"({n_ok_ai / max(len(results), 1) * 100:.0f}%)")
        lines.append(f"- AI 超越 SABRE: **{n_wins}/{comparable}** "
                     f"({n_wins / max(comparable, 1) * 100:.0f}%)")
    lines.append("")

    # 与 V13 的差异 段（强制要求，experiment-log.md §与 V13 的差异）
    lines.append("## 与 V13 的差异 (参照 docs/technical/decisions.md §V14)")
    lines.append("")
    lines.append("- **V14-1 SABRE baseline 缓存**：训练吞吐 1.0 → 15 eps/s，"
                 "本评测与训练时使用同一份 SABRE 实现，可复现对照。")
    lines.append("- **V14-2 阶段化 Mask**：5Q 阶段（Stage 0-2）已稳定收敛，本表 5Q 列代表 "
                 "V14.2 ep25333 的稳定能力；10Q/20Q（Stage 3+）仍未收敛，建议参考 SABRE 列。")
    lines.append("- **V14-3 奖励分层**：terminal reward 按 stage 切换；")
    lines.append("- **V14-4 pass_manager 真集成**：AI SWAP 数（route_count_only）可被外部独立复现，"
                 "不再调用 Qiskit SABRE 重编译。")
    lines.append("- **V13 vs V14 在 5Q QFT 上的 SWAP 数**：参见 "
                 "`models/v12_tokyo20/eval_report_v12.md` vs 本报告。")
    lines.append("")

    # AI 模型未加载的情况
    if not has_ai:
        lines.append("## 备注：AI 评测未运行")
        lines.append("")
        lines.append("当前未加载 AI 模型（V14 训练在 Stage 3 卡住，V14.2 ep25333 在 5Q 上可用）。")
        lines.append("本报告作为 SABRE 基线管线的可复现性验证 — 后续 V14.2 收敛后，")
        lines.append("跑 `python scripts/eval_mqt_bench.py --ai-model models/v14_tokyo20/v7_ibm_tokyo_best.pt` 可填充 AI 列。")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告写入: %s (%d 行)", output_path, len(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MQT-Bench 评测管线")
    parser.add_argument(
        "--model",
        "--ai-model",
        dest="ai_model",
        type=str,
        default="models/v14_tokyo20/v7_ibm_tokyo_best.pt",
        help="AI 模型 .pt 路径（默认 models/v14_tokyo20/v7_ibm_tokyo_best.pt；"
             "不存在或加载失败时自动跳过 AI 列）",
    )
    parser.add_argument(
        "--n-qubits",
        type=str,
        default="5,10,20",
        help="逗号分隔的 qubit 数列表（默认 5,10,20，覆盖 V14 P1 验收 15 条）",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="ibm_tokyo",
        help="物理拓扑（默认 ibm_tokyo 20Q）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/v14_tokyo20/eval_report_mqt.md",
        help="输出 markdown 报告路径",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="AI router 最大路由步数（默认 2000）",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="qft,qaoa,grover,ghz,vqe",
        help="逗号分隔的电路名（默认 qft,qaoa,grover,ghz,vqe — V14 P1 验收 5 类）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    n_qubits_list = [int(x) for x in args.n_qubits.split(",") if x.strip()]
    benchmark_names = [x.strip() for x in args.benchmarks.split(",") if x.strip()]

    # 1) 加载电路
    print(f"[1/4] 加载 MQT-Bench 电路 (n_qubits={n_qubits_list}, "
          f"benchmarks={benchmark_names})...")
    circuits = get_mqt_circuits(
        n_qubits_list=n_qubits_list,
        benchmark_names=benchmark_names,
    )
    if not circuits:
        logger.error("没有任何电路加载成功，退出。")
        return 1
    descriptions = describe_circuits(circuits)
    for d in descriptions:
        print(f"    {d['name']:<14s} nq={d['n_qubits']:>2d} size={d['size']:>4d} "
              f"2q={d['n_2q']:>3d} depth={d['depth']:>3d}")

    # 2) 加载拓扑
    print(f"\n[2/4] 加载拓扑 {args.topology}...")
    cm = get_topology(args.topology)
    topo_info = get_topology_info(cm)
    print(f"    {args.topology}: {topo_info}")

    # 检查电路是否能塞进拓扑
    skipped = []
    for name, qc in list(circuits.items()):
        if qc.num_qubits > cm.size():
            logger.warning("电路 %s (nq=%d) > 拓扑 (%d) — 跳过",
                           name, qc.num_qubits, cm.size())
            skipped.append(name)
            del circuits[name]
    if not circuits:
        logger.error("所有电路 qubit 数都超过拓扑大小，退出。")
        return 1

    # 3) 加载 AI router (容错；模型不存在/架构不兼容都不崩溃，AI 列填 N/A)
    print(f"\n[3/4] 加载 AI router...")
    ai_model_path: Path | None = None
    if args.ai_model:
        candidate = Path(args.ai_model)
        if candidate.exists():
            ai_model_path = candidate
        else:
            logger.warning(
                "默认 AI 模型路径不存在: %s — AI 列将填 N/A",
                candidate,
            )
    router = try_load_router(ai_model_path, cm)
    print(f"    AI router: {'OK' if router else '未加载（仅跑 SABRE）'}")

    # 4) 跑 pipelines
    print(f"\n[4/4] 跑 pipelines...")
    results = []
    for name, qc in circuits.items():
        no_route = run_no_routing(qc, DEFAULT_BASIS_GATES)
        sabre = run_sabre(qc, cm, DEFAULT_BASIS_GATES)
        ai = run_ai(qc, router, args.max_steps) if router else None

        ai_str = f"AI={ai['swaps']:>3d}" if ai else "AI=---"
        print(
            f"    {name:<14s} qubits={qc.num_qubits:>2d} size={qc.size():>4d} | "
            f"SABRE={sabre['swaps']:>3d} (depth={sabre['depth']:>3d}, "
            f"{sabre['elapsed_s']:.3f}s) | {ai_str}"
        )

        results.append({
            "name": name,
            "qubits": qc.num_qubits,
            "input_size": qc.size(),
            "no_routing": no_route,
            "sabre": sabre,
            "ai": ai,
        })

    # 5) 写报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topology": args.topology,
        "topology_info": topo_info,
        "n_qubits_list": n_qubits_list,
        "n_circuit_types": len(benchmark_names),
        "mqt_available": is_mqt_available(),
        "ai_model_path": str(ai_model_path) if ai_model_path else None,
        "basis_gates": DEFAULT_BASIS_GATES,
        "max_steps": args.max_steps,
        "skipped": skipped,
    }
    write_markdown_report(output_path, results, metadata)

    # JSON 副本（机读）
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": metadata, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # 6) Summary（V14 P1 验收必须打印）
    n_ok_sabre = sum(1 for r in results if r["sabre"]["completed"])
    avg_sabre = (
        sum(r["sabre"]["swaps"] for r in results if r["sabre"]["completed"])
        / max(n_ok_sabre, 1)
    )
    has_ai = any(r["ai"] is not None and r["ai"]["completed"] for r in results)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  电路总数: {len(results)}")
    print(f"  SABRE 完成: {n_ok_sabre}/{len(results)}")
    print(f"  SABRE 平均 SWAP: {avg_sabre:.2f}")
    if has_ai:
        ratios = []
        n_ok_ai = 0
        for r in results:
            if (
                r["ai"] is not None
                and r["ai"]["completed"]
                and r["sabre"]["completed"]
                and r["sabre"]["swaps"] > 0
            ):
                ratios.append(r["ai"]["swaps"] / r["sabre"]["swaps"])
            if r["ai"] is not None and r["ai"]["completed"]:
                n_ok_ai += 1
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        print(f"  AI 完成:    {n_ok_ai}/{len(results)}")
        print(f"  AI/SABRE ratio (mean over {len(ratios)} 可比对电路): {avg_ratio:.3f}")
    else:
        print("  AI:         未跑通（无模型 / 加载失败）— 表格 AI 列为 N/A")
    print("=" * 72)
    print(f"\nDone.")
    print(f"  Markdown: {output_path}")
    print(f"  JSON:     {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
