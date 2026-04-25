"""V15 — AlphaZero-style MCTS+GNN for quantum circuit routing.

References:
- Sinha, A. et al. "QRoute: Hybrid Quantum-Classical Routing." AAAI 2022.
- Park, S. et al. "AlphaRouter: MCTS-based Quantum Circuit Routing." arXiv:2410.05115, 2024.
- AlphaZero: Silver, D. et al. "Mastering chess and shogi by self-play with a general
  reinforcement learning algorithm." Science 2018.

Design notes (see docs/technical/decisions.md §V15):
- Reuse existing V14 infrastructure: LightweightEnv (O(1) clone), GraphSAGE 9D encoder,
  SABRE cache, pass_manager, curriculum scheduler.
- Replace PPO trainer with AlphaZero self-play + tree-search-guided policy improvement.
- Pure PyTorch; no torch_geometric (project rule).
"""
from __future__ import annotations

__all__ = ["network", "tree", "replay", "selfplay", "train"]
