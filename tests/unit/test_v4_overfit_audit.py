"""Tests for the research-only V4 overfitting audit statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from exp_v4_overfit_audit import (  # noqa: E402
    circular_block_indices,
    cscv_probability_of_backtest_overfitting,
    local_gap_grid,
    newey_west_mean_test,
    white_reality_check,
)


def test_circular_blocks_have_requested_length_and_valid_indices() -> None:
    rng = np.random.default_rng(7)
    indices = circular_block_indices(23, block_size=5, rng=rng)

    assert len(indices) == 23
    assert np.all(indices >= 0)
    assert np.all(indices < 23)


def test_newey_west_detects_positive_independent_mean() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(loc=0.004, scale=0.01, size=2_000)

    result = newey_west_mean_test(values, max_lag=10)

    assert result["mean"] > 0.003
    assert result["t_stat"] > 5.0
    assert result["p_value_one_sided"] < 0.001


def test_white_reality_check_penalizes_candidate_search() -> None:
    rng = np.random.default_rng(13)
    noise = rng.normal(scale=0.01, size=(1_000, 8))
    noise[:, 3] += 0.003

    result = white_reality_check(
        noise,
        selected_column=3,
        block_size=10,
        bootstrap_samples=1_000,
        seed=17,
    )

    assert result["candidate_count"] == 8
    assert result["selected_mean"] > 0.002
    assert result["p_value"] < 0.05


def test_cscv_flags_a_regime_specific_winner() -> None:
    returns = np.zeros((800, 4), dtype=float)
    returns[:400, 1] = 0.01
    returns[400:, 1] = -0.01
    returns[:400, 2] = -0.01
    returns[400:, 2] = 0.01
    returns[:, 3] = 0.0001

    result = cscv_probability_of_backtest_overfitting(returns, slices=8)

    assert result["splits"] == 70
    assert result["pbo"] > 0.45


def test_local_gap_grid_is_centered_and_has_27_unique_points() -> None:
    grid = local_gap_grid()

    assert len(grid) == 27
    assert len(set(grid.values())) == 27
    assert any(
        params.slow_gap == 0.0075 and params.fast_5d_gap == 0.0225 and params.fast_3d_gap == 0.01125
        for params in grid.values()
    )
