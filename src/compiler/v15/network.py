"""V15 PolicyValueNet — shared GraphSAGE backbone + (policy, value) heads.

Layout:
    obs (graph_features, edge_index)
        → GraphSAGEEncoder (V14 V13 already pure-PyTorch, reused)
        → graph_embed (B, 256 = 2*gnn_out)
        → policy_head: MLP → n_actions logits (masked softmax in caller)
        → value_head:  MLP → tanh ∈ [-1, 1]   (matches relative_sabre reward in [-1, 1])

Why we share the GNN backbone with V14:
    V14 ep25333 weights (5Q-converged) can be loaded as warm-start.
    The PolicyValueNet keeps the same `node_embedder + 3×SAGEConv` structure;
    only the heads differ (V14 actor+critic vs V15 policy+value).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.compiler.gnn_encoder import GraphSAGEEncoder


@dataclass(frozen=True)
class NetworkConfig:
    gnn_in_channels: int = 9
    gnn_hidden: int = 128
    gnn_out: int = 128
    graph_pool_dim: int = 256  # = 2 * gnn_out (mean + max pool)
    policy_head_hidden: int = 256
    value_head_hidden: int = 128
    n_actions: int = 0  # caller must set
    dropout: float = 0.1


class PolicyValueNet(nn.Module):
    """V15 AlphaZero-style network.

    Forward returns logits (pre-mask) and value tanh.
    Caller is responsible for applying action mask before softmax.
    """

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        if cfg.n_actions <= 0:
            raise ValueError("NetworkConfig.n_actions must be > 0")
        self.cfg = cfg

        self.encoder = GraphSAGEEncoder(
            in_channels=cfg.gnn_in_channels,
            hidden_channels=cfg.gnn_hidden,
            out_channels=cfg.gnn_out,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(cfg.graph_pool_dim, cfg.policy_head_hidden),
            nn.LayerNorm(cfg.policy_head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.policy_head_hidden, cfg.n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(cfg.graph_pool_dim, cfg.value_head_hidden),
            nn.LayerNorm(cfg.value_head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.value_head_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x: (N, gnn_in_channels) node features for B graphs.
            edge_index: (2, E) edges with `batch` indicating graph membership.
            batch: (N,) graph-id per node. None = single graph.
        Returns:
            logits: (B, n_actions) — pre-mask policy logits.
            value:  (B,) — predicted value in [-1, 1].
        """
        _, graph_embed = self.encoder(x, edge_index, batch)
        logits = self.policy_head(graph_embed)
        value = torch.tanh(self.value_head(graph_embed)).squeeze(-1)
        return logits, value

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Single-graph inference helper.

        Returns probability distribution over actions (mask applied) and scalar value.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                pass  # (N, F) single graph
            logits, value = self.forward(x, edge_index, batch=None)
            # logits: (1, n_actions)
            mask = action_mask.to(logits.device).float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(mask == 0, -1e9)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        return probs, float(value.item())

    @classmethod
    def warmstart_from_v14(
        cls, cfg: NetworkConfig, v14_checkpoint_path: str
    ) -> "PolicyValueNet":
        """Initialize V15 network with V14 GNN backbone weights.

        V14 actor+critic heads are NOT loaded — V15 uses fresh policy+value heads
        because the supervision signal differs (PPO advantage vs MCTS visit/outcome).
        """
        net = cls(cfg)
        try:
            ckpt = torch.load(v14_checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 — surface error
            ckpt = torch.load(v14_checkpoint_path, map_location="cpu")

        state = ckpt.get("policy_state_dict") or ckpt.get("state_dict") or ckpt
        # Filter only encoder weights (V14 named them similarly under `encoder.*` or `gnn.*`)
        encoder_state = {}
        for k, v in state.items():
            for prefix in ("encoder.", "gnn.", "node_embedder."):
                if k.startswith(prefix):
                    new_k = k if k.startswith("encoder.") else f"encoder.{k}"
                    encoder_state[new_k] = v
                    break
        if encoder_state:
            net.load_state_dict(encoder_state, strict=False)
        return net
