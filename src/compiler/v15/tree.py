"""V15 MCTS tree — PUCT selection, Dirichlet root noise, lazy expansion.

Differences from existing src/compiler/mcts.py (V4 inference search):
  1. Pure PyTorch — no torch_geometric (project rule).
  2. Uses LightweightEnv.clone() (O(1)) — replacing copy.deepcopy on QuantumRoutingEnv.
  3. Returns visit-count distribution (for self-play training labels), not just argmax.
  4. Adds Dirichlet noise at root for exploration during self-play.
  5. Hyperparams loaded from yaml (no hard-coded gamma=0.99, etc.).

PUCT formula (AlphaZero):
    a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from src.compiler.light_env import LightweightEnv


@dataclass
class MCTSConfig:
    n_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    discount: float = 1.0  # No discount for routing (terminal-only reward)


class TreeNode:
    """Single MCTS node. Lazy: env is None until first selection reaches here."""

    __slots__ = (
        "env",
        "is_terminal",
        "step_reward",
        "parent",
        "action_taken",
        "prior",
        "children",
        "N",
        "W",
        "valid_actions_mask",
    )

    def __init__(
        self,
        env: LightweightEnv | None,
        parent: "TreeNode | None",
        action_taken: int | None,
        prior: float,
    ) -> None:
        self.env = env
        self.is_terminal = False
        self.step_reward = 0.0
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        # Children indexed by action id; populated during expand()
        self.children: dict[int, TreeNode] = {}
        self.N: int = 0
        self.W: float = 0.0
        self.valid_actions_mask: np.ndarray | None = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child_puct(self, c_puct: float) -> "TreeNode":
        """Pick the child maximizing PUCT score."""
        sqrt_total_n = math.sqrt(max(self.N, 1))
        best_score = -float("inf")
        best_child: TreeNode | None = None
        for child in self.children.values():
            u = c_puct * child.prior * sqrt_total_n / (1 + child.N)
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def expand(self, action_probs: np.ndarray, valid_actions: np.ndarray) -> None:
        """Create children for all valid actions with their priors."""
        self.valid_actions_mask = valid_actions
        for a_idx in np.where(valid_actions > 0)[0]:
            a = int(a_idx)
            self.children[a] = TreeNode(
                env=None,
                parent=self,
                action_taken=a,
                prior=float(action_probs[a]),
            )

    def add_dirichlet_noise(self, alpha: float, eps: float, rng: np.random.Generator) -> None:
        """Mix Dirichlet noise into children priors (root only)."""
        if not self.children:
            return
        actions = list(self.children.keys())
        noise = rng.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise, strict=True):
            self.children[a].prior = (1 - eps) * self.children[a].prior + eps * float(n)

    def backup(self, value: float, discount: float = 1.0) -> None:
        """Propagate `value` from this node up the tree, including step rewards."""
        node: TreeNode | None = self
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            # Routing has terminal-dominant reward, but step_reward (intermediate
            # gate-completion reward) still propagates additively along the path.
            v = node.step_reward + discount * v
            node = node.parent


def _evaluate_with_network(
    env: LightweightEnv,
    network,  # PolicyValueNet
    device: torch.device,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Run policy/value network on env's current state.

    Returns:
        action_probs: (n_actions,) softmax over masked logits
        value: scalar in [-1, 1]
        action_mask: (n_actions,) {0, 1}
    """
    info = env._get_info()  # type: ignore[attr-defined]
    gnn_input = info["gnn_input"]
    graph = gnn_input["graph"]  # extract_physical_graph result

    # graph is the dict produced by gnn_extractor.extract_physical_graph
    # with keys: x (N, 9), edge_index (2, E)  — see gnn_extractor.py
    x = graph["x"] if isinstance(graph, dict) else graph.x
    edge_index = graph["edge_index"] if isinstance(graph, dict) else graph.edge_index

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = x.to(device)
    edge_index = edge_index.to(device)

    mask = env.get_action_mask()
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

    probs, value = network.predict(x, edge_index, mask_t)
    return probs.cpu().numpy(), value, mask


def run_mcts(
    root_env: LightweightEnv,
    network,  # PolicyValueNet
    cfg: MCTSConfig,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, TreeNode]:
    """Run `cfg.n_simulations` PUCT simulations from `root_env` state.

    Returns:
        visit_counts: (n_actions,) raw N counts at the root
        root: the root TreeNode (caller may extract Q values for logging)
    """
    if device is None:
        device = next(network.parameters()).device
    if rng is None:
        rng = np.random.default_rng()

    # Root: clone env once so the original isn't modified
    root_env_copy = root_env.clone()
    root = TreeNode(env=root_env_copy, parent=None, action_taken=None, prior=1.0)

    # Initial expand of root
    probs, _, mask = _evaluate_with_network(root.env, network, device)
    root.expand(probs, mask)
    root.add_dirichlet_noise(cfg.dirichlet_alpha, cfg.dirichlet_eps, rng)

    for _ in range(cfg.n_simulations):
        node = root
        # 1. Select down to a leaf
        while node.is_expanded and not node.is_terminal:
            node = node.select_child_puct(cfg.c_puct)
            # Lazy env materialization
            if node.env is None:
                node.env = node.parent.env.clone()
                _, reward, term, trunc, _ = node.env.step(node.action_taken)
                node.step_reward = float(reward)
                node.is_terminal = bool(term or trunc)
                if node.is_terminal:
                    break

        # 2. Evaluate / Expand
        if node.is_terminal:
            # Terminal value: completed circuit (env.is_done()) gets +1; truncated −1
            if node.env.is_done():
                leaf_value = 1.0
            else:
                # Truncated without completion: heavy penalty
                leaf_value = -1.0
        else:
            probs, leaf_value, mask = _evaluate_with_network(node.env, network, device)
            node.expand(probs, mask)

        # 3. Backup
        node.backup(leaf_value, discount=cfg.discount)

    visit_counts = np.zeros(root_env.n_actions, dtype=np.float32)
    for a, child in root.children.items():
        visit_counts[a] = child.N
    return visit_counts, root


def visits_to_policy(
    visit_counts: np.ndarray, temperature: float
) -> np.ndarray:
    """Convert visit counts to a policy distribution.

    T → 0:    one-hot at argmax(visits)
    T → 1:    proportional to visits
    """
    if temperature < 1e-6:
        policy = np.zeros_like(visit_counts)
        if visit_counts.sum() > 0:
            policy[int(visit_counts.argmax())] = 1.0
        return policy
    powered = visit_counts ** (1.0 / temperature)
    s = powered.sum()
    return powered / s if s > 0 else np.ones_like(visit_counts) / len(visit_counts)
