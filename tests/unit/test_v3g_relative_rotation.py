"""Pure-signal tests for the V3-G relative-weakness rotation experiment."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from exp_v3g_relative_rotation import (  # noqa: E402
    RotationParams,
    should_early_rotate,
)


def test_rank_flip_alone_does_not_rotate() -> None:
    decision = should_early_rotate(
        holding="silver",
        leader="gold",
        leader_score=0.10,
        holding_score=0.095,
        holding_drawdown=-0.01,
        leader_count_last3=2,
        days_since_early_rotation=99,
        params=RotationParams(),
    )

    assert not decision.triggered


def test_standard_signal_requires_relative_and_absolute_weakness() -> None:
    decision = should_early_rotate(
        holding="silver",
        leader="gold",
        leader_score=0.11,
        holding_score=0.08,
        holding_drawdown=-0.05,
        leader_count_last3=2,
        days_since_early_rotation=99,
        params=RotationParams(),
    )

    assert decision.triggered
    assert decision.kind == "standard"


def test_fast_signal_bypasses_rank_persistence() -> None:
    decision = should_early_rotate(
        holding="silver",
        leader="gold",
        leader_score=0.09,
        holding_score=0.08,
        holding_drawdown=-0.08,
        leader_count_last3=1,
        days_since_early_rotation=99,
        params=RotationParams(),
    )

    assert decision.triggered
    assert decision.kind == "fast"


def test_three_day_lock_blocks_positive_momentum_reversal() -> None:
    decision = should_early_rotate(
        holding="gold",
        leader="silver",
        leader_score=0.12,
        holding_score=0.06,
        holding_drawdown=-0.08,
        leader_count_last3=3,
        days_since_early_rotation=2,
        params=RotationParams(),
    )

    assert not decision.triggered
    assert decision.reason == "minimum_hold"


def test_three_day_lock_releases_when_momentum_is_negative() -> None:
    decision = should_early_rotate(
        holding="gold",
        leader="silver",
        leader_score=0.05,
        holding_score=-0.01,
        holding_drawdown=-0.08,
        leader_count_last3=1,
        days_since_early_rotation=2,
        params=RotationParams(),
    )

    assert decision.triggered
    assert decision.kind == "fast"


def test_precious_pair_scope_rejects_cross_theme_rotation() -> None:
    decision = should_early_rotate(
        holding="161226",
        leader="501018",
        leader_score=0.12,
        holding_score=0.05,
        holding_drawdown=-0.08,
        leader_count_last3=3,
        days_since_early_rotation=99,
        params=RotationParams(scope_assets=("518880", "161226")),
    )

    assert not decision.triggered
    assert decision.reason == "outside_scope"


def test_group_scope_accepts_rotation_within_any_declared_group() -> None:
    decision = should_early_rotate(
        holding="513100",
        leader="159941",
        leader_score=0.08,
        holding_score=0.07,
        holding_drawdown=-0.01,
        leader_count_last3=1,
        days_since_early_rotation=99,
        params=RotationParams(
            relative_gap=0.005,
            holding_drawdown=0.0,
            persistence_hits=1,
            fast_drawdown=None,
            scope_groups=(("518880", "161226"), ("513100", "159941")),
        ),
    )

    assert decision.triggered
