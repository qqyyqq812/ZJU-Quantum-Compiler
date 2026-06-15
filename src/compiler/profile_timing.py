"""Small phase timing helper for NPQR runtime measurements."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PhaseStats:
    total_ms: float = 0.0
    count: int = 0
    min_ms: float | None = None
    max_ms: float | None = None

    def add(self, elapsed_ms: float) -> None:
        self.total_ms += float(elapsed_ms)
        self.count += 1
        self.min_ms = elapsed_ms if self.min_ms is None else min(self.min_ms, elapsed_ms)
        self.max_ms = elapsed_ms if self.max_ms is None else max(self.max_ms, elapsed_ms)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_ms": self.total_ms,
            "count": self.count,
            "min_ms": 0.0 if self.min_ms is None else self.min_ms,
            "max_ms": 0.0 if self.max_ms is None else self.max_ms,
        }


@dataclass
class PhaseProfiler:
    """Accumulates wall-clock phase timings with near-zero disabled overhead."""

    enabled: bool = True
    phases: dict[str, PhaseStats] = field(default_factory=lambda: defaultdict(PhaseStats))

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[phase].add((time.perf_counter() - started) * 1000.0)

    def add(self, phase: str, elapsed_ms: float) -> None:
        if self.enabled:
            self.phases[phase].add(elapsed_ms)

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {phase: stats.as_dict() for phase, stats in sorted(self.phases.items())}


def maybe_measure(profiler: PhaseProfiler | None, phase: str):
    """Return a measurement context manager only when profiling is active."""
    return profiler.measure(phase) if profiler is not None else nullcontext()
