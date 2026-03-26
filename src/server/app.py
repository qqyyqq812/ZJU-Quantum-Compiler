"""
V7.2 量子电路 AI 路由器 — 后端 API
===================================
FastAPI 服务器，提供 AI 路由编译接口。

启动:
    uvicorn src.server.app:app --reload --port 8000

API 端点:
    POST /compile    — 编译量子电路
    GET  /status     — 模型状态
    GET  /benchmarks — 基准测试结果
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.benchmarks.topologies import get_topology
from src.compiler.env import QuantumRoutingEnv
from src.compiler.policy import PolicyNetwork
from src.compiler.inference_v8 import compile_multi_trial
from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa, generate_grover

app = FastAPI(
    title="量子电路 AI 路由器",
    version="7.2",
    description="基于 PPO+GNN+课程学习的量子电路编译路由器",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型
_policy: Optional[PolicyNetwork] = None
_topology_name = "linear_5"


def get_policy() -> PolicyNetwork:
    """延迟加载模型。"""
    global _policy
    if _policy is None:
        cm = get_topology(_topology_name)
        env = QuantumRoutingEnv(coupling_map=cm)
        _policy = PolicyNetwork(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
        )
        # 尝试加载 best 模型
        model_paths = [
            "models/v7_final_v2/v7_linear_5_best.pt",
            "models/v7_final_v2/v7_linear_5.pt",
            "models/v7_final/v7_linear_5_best.pt",
        ]
        for p in model_paths:
            if Path(p).exists():
                _policy.load_state_dict(
                    torch.load(p, map_location="cpu", weights_only=True)
                )
                print(f"✅ 模型加载: {p}")
                break
        _policy.eval()
    return _policy


class CompileRequest(BaseModel):
    circuit_type: str = "random"  # random / qft / qaoa / grover
    n_qubits: int = 5
    depth: int = 3
    seed: int = 42


class CompileResult(BaseModel):
    circuit_type: str
    n_qubits: int
    cx_count: int
    ai_swaps: int
    completed: bool
    steps: int


class StatusResponse(BaseModel):
    model_loaded: bool
    topology: str
    model_path: str


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取服务器状态。"""
    policy = get_policy()
    return StatusResponse(
        model_loaded=policy is not None,
        topology=_topology_name,
        model_path="v7_final_v2",
    )


@app.post("/compile", response_model=CompileResult)
async def compile_circuit(req: CompileRequest):
    """编译量子电路，返回路由结果。"""
    if req.n_qubits > 5:
        raise HTTPException(400, "当前拓扑最大支持 5 qubits")

    # 生成电路
    generators = {
        "random": lambda: generate_random(req.n_qubits, depth=req.depth, seed=req.seed),
        "qft": lambda: generate_qft(req.n_qubits),
        "qaoa": lambda: generate_qaoa(req.n_qubits, p=req.depth),
        "grover": lambda: generate_grover(req.n_qubits),
    }
    gen = generators.get(req.circuit_type)
    if not gen:
        raise HTTPException(400, f"不支持的电路类型: {req.circuit_type}")

    qc = gen()
    cx_count = dict(qc.count_ops()).get("cx", 0)

    # AI 路由 (V8 高并发 MTx100)
    policy = get_policy()
    cm = get_topology(_topology_name)
    
    with torch.no_grad():
        result = compile_multi_trial(qc, policy, cm, n_trials=100)

    return CompileResult(
        circuit_type=req.circuit_type,
        n_qubits=req.n_qubits,
        cx_count=cx_count,
        ai_swaps=result.swaps,
        completed=result.completed,
        steps=result.steps,
    )


@app.get("/benchmarks")
async def run_benchmarks():
    """运行快速基准测试。"""
    policy = get_policy()
    cm = get_topology(_topology_name)

    circuits = [
        ("random_5_d2", generate_random(5, depth=2, seed=42)),
        ("random_5_d3", generate_random(5, depth=3, seed=42)),
        ("random_5_d4", generate_random(5, depth=4, seed=42)),
        ("qft_5", generate_qft(5)),
        ("qaoa_5", generate_qaoa(5, p=1)),
        ("grover_5", generate_grover(5)),
    ]

    results = []
    for name, qc in circuits:
        cx = dict(qc.count_ops()).get("cx", 0)
        with torch.no_grad():
            res = compile_multi_trial(qc, policy, cm, n_trials=100)

        results.append({
            "name": name,
            "cx": cx,
            "ai_swaps": res.swaps,
            "completed": res.completed,
            "steps": res.steps,
        })

    return {"topology": _topology_name, "results": results}
