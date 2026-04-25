"""V15 outer training loop — self-play + supervised network update + eval.

For each iteration:
    1. Generate `games_per_iter` self-play games (CPU).
    2. Push samples into replay buffer.
    3. Run `steps_per_iter` SGD steps on a sampled batch.
    4. Every `eval_interval` iterations: eval vs SABRE on held-out circuits.
    5. Every `checkpoint_interval` iterations: save (best, latest) checkpoints.

Curriculum integration:
    Same scheduler as V14 (src.compiler.curriculum.CurriculumScheduler).
    The trainer reports avg SWAP per stage of self-play games.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.compiler.curriculum import CurriculumScheduler
from src.compiler.env import QuantumRoutingEnv
from src.compiler.v15.network import NetworkConfig, PolicyValueNet
from src.compiler.v15.replay import ReplayBuffer
from src.compiler.v15.selfplay import SelfPlayConfig, play_one_episode
from src.compiler.v15.tree import MCTSConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    iterations: int = 1000
    games_per_iter: int = 100
    steps_per_iter: int = 200
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    log_interval: int = 10
    eval_interval: int = 5
    checkpoint_interval: int = 10
    save_dir: str = "models/v15_tokyo20"


def _batch_to_device(
    xs: list[torch.Tensor],
    edges: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge a list of variable-size graphs into one batched graph.

    Concatenates node features, offsets edge_index per graph, builds `batch` vector.
    """
    batch_indices: list[torch.Tensor] = []
    offset_edges: list[torch.Tensor] = []
    node_offset = 0
    for i, (x, ei) in enumerate(zip(xs, edges, strict=True)):
        n = x.size(0)
        batch_indices.append(torch.full((n,), i, dtype=torch.long))
        offset_edges.append(ei + node_offset)
        node_offset += n
    x_cat = torch.cat(xs, dim=0).to(device)
    ei_cat = torch.cat(offset_edges, dim=1).to(device)
    batch = torch.cat(batch_indices, dim=0).to(device)
    return x_cat, ei_cat, batch


def train_one_iteration(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: TrainConfig,
    device: torch.device,
    rng: random.Random,
) -> dict[str, float]:
    """One outer iteration of supervised network update."""
    network.train()
    pol_losses: list[float] = []
    val_losses: list[float] = []

    if len(buffer) < cfg.batch_size:
        return {"policy_loss": float("nan"), "value_loss": float("nan")}

    for _ in range(cfg.steps_per_iter):
        xs, edges, policies, values = buffer.sample_batch(cfg.batch_size, rng)
        x, ei, batch = _batch_to_device(xs, edges, device)
        policies = policies.to(device)
        values = values.to(device)

        logits, v_pred = network(x, ei, batch)
        # Cross-entropy with target distribution
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(policies * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(v_pred, values)

        loss = (
            cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip)
        optimizer.step()

        pol_losses.append(float(policy_loss.item()))
        val_losses.append(float(value_loss.item()))

    return {
        "policy_loss": float(np.mean(pol_losses)),
        "value_loss": float(np.mean(val_losses)),
    }


def run_training(
    full_cfg: dict[str, Any],
    coupling_map,  # qiskit CouplingMap
) -> dict[str, Any]:
    """Top-level entry point. Driven by parsed yaml dict.

    Args:
        full_cfg: parsed v15_*.yaml as nested dict.
        coupling_map: qiskit CouplingMap matching topology.name.
    Returns:
        dict with training history.
    """
    rng_np = np.random.default_rng(42)
    rng_py = random.Random(42)

    n_qubits = int(full_cfg["topology"]["n_qubits"])
    device = torch.device(full_cfg["hardware"].get("device", "cpu"))

    # --- Network ---
    net_cfg_d = full_cfg["network"]
    # We need n_actions; build a dummy env to ask.
    # Note: SABRE baseline caching is handled internally by env._compute_sabre_baseline
    # via the sabre_cache module (V14-1 optimization), no constructor arg needed.
    cur_d = full_cfg.get("curriculum", {})
    scheduler = CurriculumScheduler(
        max_n_qubits=coupling_map.size(),
        min_episodes_per_stage=int(cur_d.get("min_episodes_per_stage", 100)),
        promotion_patience=int(cur_d.get("promotion_patience", 3000)),
    )
    bootstrap_circuits = scheduler.circuits
    use_sabre_cache = bool(full_cfg["environment"].get("use_sabre_cache", True))
    dummy_env = QuantumRoutingEnv(
        coupling_map=coupling_map,
        max_steps=int(full_cfg["environment"].get("max_steps", 800)),
        soft_mask=bool(full_cfg["environment"].get("soft_mask", True)),
        tabu_size=int(full_cfg["environment"].get("tabu_size", 4)),
        use_sabre_reward=use_sabre_cache,
    )
    dummy_env.set_circuit(bootstrap_circuits[0], topology_name=full_cfg["topology"]["name"])
    dummy_env.reset()
    n_actions = int(dummy_env.n_actions)

    net_cfg = NetworkConfig(
        gnn_in_channels=int(net_cfg_d.get("gnn_in_channels", 9)),
        gnn_hidden=int(net_cfg_d.get("gnn_hidden", 128)),
        gnn_out=int(net_cfg_d.get("gnn_out", 128)),
        graph_pool_dim=int(net_cfg_d.get("graph_pool_dim", 256)),
        policy_head_hidden=int(net_cfg_d.get("policy_head_hidden", 256)),
        value_head_hidden=int(net_cfg_d.get("value_head_hidden", 128)),
        n_actions=n_actions,
        dropout=float(net_cfg_d.get("dropout", 0.1)),
    )

    warmstart = full_cfg.get("paths", {}).get("v14_warmstart")
    if warmstart and Path(warmstart).exists():
        logger.info("Warm-starting V15 from V14 weights: %s", warmstart)
        network = PolicyValueNet.warmstart_from_v14(net_cfg, warmstart)
    else:
        if warmstart:
            logger.warning("V14 warmstart not found at %s; cold-starting", warmstart)
        network = PolicyValueNet(net_cfg)
    network = network.to(device)

    # --- MCTS / SelfPlay configs ---
    mc_d = full_cfg["mcts"]
    mcts_cfg = MCTSConfig(
        n_simulations=int(mc_d.get("n_simulations", 100)),
        c_puct=float(mc_d.get("c_puct", 1.5)),
        dirichlet_alpha=float(mc_d.get("dirichlet_alpha", 0.3)),
        dirichlet_eps=float(mc_d.get("dirichlet_eps", 0.25)),
    )
    sp_d = full_cfg["selfplay"]
    sp_cfg = SelfPlayConfig(
        max_steps_per_game=int(sp_d.get("max_steps_per_game", 800)),
        temperature_warmup=float(mc_d.get("temperature_warmup", 1.0)),
        temperature_play=float(mc_d.get("temperature_play", 0.0)),
        temperature_threshold=int(mc_d.get("temperature_threshold", 30)),
        reward_scheme=str(sp_d.get("reward_scheme", "relative_sabre")),
    )

    # --- Training config ---
    tr_d = full_cfg["training"]
    train_cfg = TrainConfig(
        iterations=int(tr_d.get("iterations", 1000)),
        games_per_iter=int(sp_d.get("games_per_iter", 100)),
        steps_per_iter=int(tr_d.get("steps_per_iter", 200)),
        batch_size=int(tr_d.get("batch_size", 128)),
        learning_rate=float(tr_d.get("learning_rate", 3.0e-4)),
        weight_decay=float(tr_d.get("weight_decay", 1.0e-4)),
        grad_clip=float(tr_d.get("grad_clip", 1.0)),
        policy_loss_weight=float(tr_d.get("policy_loss_weight", 1.0)),
        value_loss_weight=float(tr_d.get("value_loss_weight", 1.0)),
        log_interval=int(tr_d.get("log_interval", 10)),
        eval_interval=int(tr_d.get("eval_interval", 5)),
        checkpoint_interval=int(tr_d.get("checkpoint_interval", 10)),
        save_dir=str(full_cfg["paths"]["save_dir"]),
    )

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Replay ---
    rep_cap = int(full_cfg["replay"].get("capacity", 100000))
    buffer = ReplayBuffer(rep_cap)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    lr_sched_name = tr_d.get("lr_schedule", "cosine")
    if lr_sched_name == "cosine":
        scheduler_lr = CosineAnnealingLR(
            optimizer, T_max=train_cfg.iterations, eta_min=1.0e-5
        )
    else:
        scheduler_lr = None

    history: dict[str, list] = {
        "iter": [],
        "policy_loss": [],
        "value_loss": [],
        "avg_swaps": [],
        "avg_outcome_z": [],
        "completion_rate": [],
        "stage": [],
    }

    for it in range(1, train_cfg.iterations + 1):
        t0 = time.time()
        # ---- Self-play ----
        circuits = scheduler.circuits
        ep_swaps: list[int] = []
        ep_zs: list[float] = []
        ep_completed: list[bool] = []
        for g in range(train_cfg.games_per_iter):
            circ = circuits[g % len(circuits)]
            env = QuantumRoutingEnv(
                coupling_map=coupling_map,
                max_steps=int(full_cfg["environment"].get("max_steps", 800)),
                soft_mask=bool(full_cfg["environment"].get("soft_mask", True)),
                tabu_size=int(full_cfg["environment"].get("tabu_size", 4)),
                use_sabre_reward=use_sabre_cache,
            )
            env.set_circuit(circ, topology_name=full_cfg["topology"]["name"])
            env.set_curriculum_stage(scheduler.current_stage)
            env.reset()
            # SABRE baseline cached by env.reset() via sabre_cache module (V14-1)
            sabre_swaps = int(getattr(env, "_sabre_swaps", 0))
            samples, info = play_one_episode(
                env, network, mcts_cfg, sp_cfg, sabre_swaps, rng=rng_np
            )
            buffer.push_many(samples)
            ep_swaps.append(info["total_swaps"])
            ep_zs.append(info["outcome_z"])
            ep_completed.append(info["completed"])
            # Curriculum scheduler counts each game as one episode
            # (was incorrectly batched per-iter in earlier V15 — slowed promotion 100x)
            scheduler.report_episode(int(info["total_swaps"]))

        avg_sw = float(np.mean(ep_swaps)) if ep_swaps else float("nan")
        # Detect promotion: stage may have advanced inside the inner loop above
        promoted = (
            scheduler.current_stage > history["stage"][-1]
            if history["stage"] else False
        )
        if promoted:
            logger.info(
                "Curriculum promoted to stage %d (%s)",
                scheduler.current_stage,
                scheduler.stage_config.name,
            )

        # ---- Train ----
        losses = train_one_iteration(
            network, optimizer, buffer, train_cfg, device, rng_py
        )
        if scheduler_lr is not None:
            scheduler_lr.step()

        history["iter"].append(it)
        history["policy_loss"].append(losses["policy_loss"])
        history["value_loss"].append(losses["value_loss"])
        history["avg_swaps"].append(avg_sw)
        history["avg_outcome_z"].append(float(np.mean(ep_zs)))
        history["completion_rate"].append(float(np.mean(ep_completed)))
        history["stage"].append(scheduler.current_stage)

        if it % train_cfg.log_interval == 0:
            dt = time.time() - t0
            logger.info(
                "iter %d  stage=%d  swap=%.1f  z=%.3f  comp=%.0f%%  "
                "pol=%.4f  val=%.4f  buf=%d  %.1fs",
                it,
                scheduler.current_stage,
                avg_sw,
                history["avg_outcome_z"][-1],
                100 * history["completion_rate"][-1],
                losses["policy_loss"],
                losses["value_loss"],
                len(buffer),
                dt,
            )

        if it % train_cfg.checkpoint_interval == 0:
            ckpt_path = save_dir / f"checkpoint_v15_iter{it}.pt"
            torch.save(
                {
                    "iter": it,
                    "network_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "curriculum_stage": scheduler.current_stage,
                    "config": full_cfg,
                },
                ckpt_path,
            )
            logger.info("💾 saved %s", ckpt_path)

        # Persist history each iter
        with (save_dir / "history_v15.json").open("w") as f:
            json.dump(history, f, indent=2)

    return history
