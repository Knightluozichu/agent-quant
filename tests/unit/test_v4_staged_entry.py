"""Pure tests for the research-only V4 staged-entry confirmation."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import qixing_v4 as v4  # noqa: E402
from exp_v4_stop import staged_top_up_allowed  # noqa: E402


def factor(
    *,
    eligible: bool = True,
    slow: float = 0.05,
    trend: float = 0.01,
) -> v4.AssetFactors:
    return v4.AssetFactors(
        code="silver",
        eligible=eligible,
        slow_momentum=slow,
        return_3d=0.01,
        return_5d=0.02,
        acceleration_5d=0.0,
        trend_strength=trend,
        vol_adjusted_5d=0.0,
        drawdown_5d=0.0,
    )


def test_top_up_requires_two_days_and_same_daily_target() -> None:
    factors = {"silver": factor()}

    assert not staged_top_up_allowed(
        holding="silver",
        holding_age=1,
        confirmation_days=2,
        selected_target="silver",
        factors=factors,
        guard_triggered=False,
    )
    assert not staged_top_up_allowed(
        holding="silver",
        holding_age=2,
        confirmation_days=2,
        selected_target="gold",
        factors=factors,
        guard_triggered=False,
    )
    assert staged_top_up_allowed(
        holding="silver",
        holding_age=2,
        confirmation_days=2,
        selected_target="silver",
        factors=factors,
        guard_triggered=False,
    )


def test_top_up_requires_eligible_positive_slow_and_short_trend() -> None:
    for held in (
        factor(eligible=False),
        factor(slow=0.0),
        factor(trend=0.0),
    ):
        assert not staged_top_up_allowed(
            holding="silver",
            holding_age=2,
            confirmation_days=2,
            selected_target="silver",
            factors={"silver": held},
            guard_triggered=False,
        )


def test_guard_has_priority_over_top_up() -> None:
    assert not staged_top_up_allowed(
        holding="silver",
        holding_age=2,
        confirmation_days=2,
        selected_target="silver",
        factors={"silver": factor()},
        guard_triggered=True,
    )
