"""Tests for the research-only V3-G overfitting audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from exp_v3g_overfit_audit import (  # noqa: E402
    GateParams,
    gate_allows,
    historical_score_periods,
    local_gate_neighbors,
)


def shock_path() -> np.ndarray:
    close = np.full(80, 110.0)
    close[-21:] = np.linspace(100.0, 104.0, 21)
    close[-5] = 99.0
    close[-4:] = (100.5, 102.0, 103.0, 104.0)
    return close


def test_gate_can_allow_low_position_positive_momentum_shock() -> None:
    close = shock_path()

    assert gate_allows(close, GateParams())
    assert not gate_allows(close, GateParams(strict_drop_filter=True))


def test_gate_rejects_shock_after_large_prior_advance() -> None:
    close = shock_path()
    close[-61] = 90.0

    assert not gate_allows(close, GateParams())


def test_historical_period_grid_has_29_unique_pairs_and_production() -> None:
    periods = historical_score_periods()

    assert len(periods) == 29
    assert len(set(periods)) == 29
    assert (10, 20) in periods


def test_local_gate_neighbors_are_unique_and_include_production() -> None:
    neighbors = local_gate_neighbors()

    assert len(neighbors) >= 10
    assert len(set(neighbors.values())) == len(neighbors)
    assert GateParams() in neighbors.values()
