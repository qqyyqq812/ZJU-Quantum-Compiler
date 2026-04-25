"""V15 Replay buffer — fixed-capacity FIFO over (state, π_MCTS, z) tuples."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Sample:
    """One self-play training sample.

    Attributes:
        x: (N, gnn_in_channels) graph node features at the time the action was taken.
        edge_index: (2, E) graph edges.
        policy: (n_actions,) MCTS visit-count distribution (sums to 1 after temperature).
        value: scalar in [-1, 1] — final game outcome backfilled to this state.
    """

    x: np.ndarray
    edge_index: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[Sample] = deque(maxlen=capacity)

    def push(self, sample: Sample) -> None:
        self._buf.append(sample)

    def push_many(self, samples: list[Sample]) -> None:
        self._buf.extend(samples)

    def __len__(self) -> int:
        return len(self._buf)

    def sample_batch(
        self, batch_size: int, rng: random.Random | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample a batch and return python-list-of-tensors form.

        Why list-of-tensors: graphs have variable N (depends on which physical qubits
        are present in the current sub-stage). The training loop is responsible for
        merging them into a batched graph (offsetting edge_index, building `batch` vec).

        Returns:
            xs: list of (N_i, F) tensors
            edge_indices: list of (2, E_i) long tensors
            policies: (B, n_actions) tensor
            values: (B,) tensor
        """
        if rng is None:
            rng = random
        batch = rng.sample(list(self._buf), min(batch_size, len(self._buf)))
        xs = [torch.tensor(s.x, dtype=torch.float32) for s in batch]
        edges = [torch.tensor(s.edge_index, dtype=torch.long) for s in batch]
        policies = torch.tensor(np.stack([s.policy for s in batch]), dtype=torch.float32)
        values = torch.tensor([s.value for s in batch], dtype=torch.float32)
        return xs, edges, policies, values
