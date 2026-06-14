from __future__ import annotations

from src.benchmarks.circuits import generate_random
from src.compiler.npqr_runtime import NPQRRuntimeConfig, should_precheck_frontier_rescue


def _refined_config() -> NPQRRuntimeConfig:
    return NPQRRuntimeConfig(
        frontier_min_unique_pair_ratio=0.485,
        frontier_max_cx_like=120,
        frontier_max_repeat_pair_ratio=0.6,
    )


def test_random_d14_matches_stage72_refined_trigger_shape():
    triggered, metrics = should_precheck_frontier_rescue(
        generate_random(10, depth=14, seed=42),
        _refined_config(),
    )

    assert triggered is True
    assert metrics["cx_like_count"] >= 40
    assert metrics["unique_two_qubit_pairs"] >= 25
    assert metrics["pair_entropy"] >= 4.0
    assert metrics["depth"] <= 30


def test_random_shallow_and_depth_sentinel_stay_outside_refined_trigger():
    shallow_triggered, shallow_metrics = should_precheck_frontier_rescue(
        generate_random(10, depth=8, seed=42),
        _refined_config(),
    )
    depth_triggered, depth_metrics = should_precheck_frontier_rescue(
        generate_random(10, depth=16, seed=42),
        _refined_config(),
    )

    assert shallow_triggered is False
    assert shallow_metrics["cx_like_count"] < 40
    assert depth_triggered is False
    assert depth_metrics["depth"] > 30
