from __future__ import annotations

import pytest
from scripts.exp_v4_momentum_failure import (
    replay_documented_failure,
    threshold_neighborhood,
)
from scripts.exp_v4_stop import momentum_failure_decision
from scripts.qixing_v4 import AssetFactors


def _factor(
    code: str,
    *,
    slow: float,
    return_3d: float,
    return_5d: float,
    trend: float = 0.01,
    eligible: bool = True,
) -> AssetFactors:
    return AssetFactors(
        code=code,
        eligible=eligible,
        slow_momentum=slow,
        return_3d=return_3d,
        return_5d=return_5d,
        acceleration_5d=0.0,
        trend_strength=trend,
        vol_adjusted_5d=0.0,
        drawdown_5d=0.0,
    )


def test_failure_breaker_does_not_override_an_exit_v4_already_makes() -> None:
    decision = momentum_failure_decision(
        mode="stale_leader_cash",
        holding="silver",
        v4_would_exit=True,
        one_day=-0.08,
        three_day=-0.12,
        candidates=[],
        factors={},
    )

    assert not decision.triggered


def test_failure_breaker_exits_stale_leader_after_first_observed_shock() -> None:
    decision = momentum_failure_decision(
        mode="stale_leader_cash",
        holding="silver",
        v4_would_exit=False,
        one_day=-0.05,
        three_day=-0.05,
        candidates=[("silver", 0.20), ("gold", 0.05)],
        factors={},
    )

    assert decision.triggered
    assert decision.exit_to_cash
    assert decision.reasons == ("failure_1d", "no_qualified_alternative")


def test_failure_breaker_rotates_to_a_qualified_alternative() -> None:
    factors = {
        "silver": _factor(
            "silver", slow=0.15, return_3d=-0.04, return_5d=-0.01
        ),
        "gold": _factor(
            "gold", slow=0.05, return_3d=0.02, return_5d=0.04
        ),
    }
    decision = momentum_failure_decision(
        mode="stale_leader_best_or_cash",
        holding="silver",
        v4_would_exit=False,
        one_day=-0.05,
        three_day=-0.05,
        candidates=[("silver", 0.15), ("gold", 0.05)],
        factors=factors,
    )

    assert decision.triggered
    assert decision.target == "gold"
    assert not decision.exit_to_cash
    assert decision.reasons == ("failure_1d", "qualified_alternative")


def test_failure_breaker_rejects_a_weak_alternative_and_uses_cash() -> None:
    factors = {
        "silver": _factor(
            "silver", slow=0.15, return_3d=-0.04, return_5d=-0.01
        ),
        "gold": _factor(
            "gold", slow=-0.01, return_3d=0.02, return_5d=0.04
        ),
    }
    decision = momentum_failure_decision(
        mode="stale_leader_best_or_cash",
        holding="silver",
        v4_would_exit=False,
        one_day=-0.05,
        three_day=-0.05,
        candidates=[("silver", 0.15), ("gold", -0.01)],
        factors=factors,
    )

    assert decision.triggered
    assert decision.target is None
    assert decision.exit_to_cash


def test_failure_breaker_uses_three_day_loss_without_future_data() -> None:
    decision = momentum_failure_decision(
        mode="stale_leader_cash",
        holding="silver",
        v4_would_exit=False,
        one_day=-0.02,
        three_day=-0.10,
        candidates=[],
        factors={},
    )

    assert decision.triggered
    assert decision.reasons[0] == "failure_3d"


def test_failure_breaker_has_no_signal_before_the_first_shock() -> None:
    decision = momentum_failure_decision(
        mode="stale_leader_cash",
        holding="silver",
        v4_would_exit=False,
        one_day=0.01,
        three_day=0.03,
        candidates=[],
        factors={},
    )

    assert not decision.triggered


def test_documented_failure_switches_after_not_before_the_first_loss() -> None:
    replay = replay_documented_failure()

    assert replay["first_shock_is_unavoidable"]
    assert replay["v4_stale_holding_return"] == pytest.approx(-0.15222)
    assert replay["breaker_cash_return"] == pytest.approx(-0.05)
    assert replay["breaker_qualified_gold_return"] == pytest.approx(-0.00193)


def test_threshold_robustness_is_a_five_point_one_axis_neighborhood() -> None:
    neighborhood = threshold_neighborhood()

    assert len(neighborhood) == 5
    assert neighborhood["center"] == (-0.05, -0.10)
    assert set(neighborhood.values()) == {
        (-0.05, -0.10),
        (-0.04, -0.10),
        (-0.06, -0.10),
        (-0.05, -0.08),
        (-0.05, -0.12),
    }
