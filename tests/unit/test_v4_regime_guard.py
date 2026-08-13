from __future__ import annotations

import numpy as np
import pytest
from scripts.exp_v4_regime_guard import (
    FloatArray,
    RegimeGuardParams,
    absolute_trend_votes,
    confirmed_tail_exposure,
    conjunctive_exposure,
    expanding_thresholds,
    parameter_neighborhood,
    preregistered_variants,
    regime_exposure,
    risk_votes,
    tail_decay_confirmation,
)


def test_absolute_trend_requires_two_independent_horizons_by_default() -> None:
    assert absolute_trend_votes(
        return_20d=0.01, return_60d=-0.02, ma20_slope_5d=0.01
    ) == 2
    assert absolute_trend_votes(
        return_20d=-0.01, return_60d=-0.02, ma20_slope_5d=0.01
    ) == 1


def test_expanding_threshold_excludes_current_observation() -> None:
    values: FloatArray = np.asarray(
        [1.0, 2.0, 3.0, 100.0], dtype=np.float64
    )
    thresholds = expanding_thresholds(
        values, quantile=0.5, minimum_history=3, default=float("inf")
    )

    assert np.isinf(thresholds[:3]).all()
    assert thresholds[3] == pytest.approx(2.0)


def test_each_orthogonal_stress_family_contributes_only_one_vote() -> None:
    params = RegimeGuardParams()
    count, reasons = risk_votes(
        breadth_20d=0.20,
        median_return_20d=-0.10,
        correlation_concentration=0.60,
        correlation_threshold=0.50,
        downside_vol_acceleration=2.0,
        downside_vol_threshold=1.5,
        rank_churn_10d=5.0,
        churn_threshold=3.0,
        leader_confidence=0.5,
        confidence_threshold=1.0,
        params=params,
    )

    assert count == 4
    assert reasons == (
        "weak_breadth",
        "correlation_concentration",
        "downside_vol_acceleration",
        "unstable_leadership",
    )


def test_two_risk_votes_set_half_exposure() -> None:
    exposure, active, streak, reasons = regime_exposure(
        absolute_votes=3,
        votes=2,
        stale_holding_shock=False,
        portfolio_drawdown=-0.02,
        risk_off_active=False,
        recovery_streak=0,
        params=RegimeGuardParams(),
    )

    assert exposure == pytest.approx(0.5)
    assert not active
    assert streak == 0
    assert "yellow_regime" in reasons


def test_three_risk_votes_set_zero_exposure() -> None:
    exposure, active, streak, reasons = regime_exposure(
        absolute_votes=3,
        votes=3,
        stale_holding_shock=False,
        portfolio_drawdown=-0.02,
        risk_off_active=False,
        recovery_streak=0,
        params=RegimeGuardParams(),
    )

    assert exposure == 0.0
    assert active
    assert streak == 0
    assert "red_regime" in reasons


def test_observed_stale_holding_shock_overrides_high_absolute_momentum() -> None:
    exposure, active, _streak, reasons = regime_exposure(
        absolute_votes=3,
        votes=0,
        stale_holding_shock=True,
        portfolio_drawdown=0.0,
        risk_off_active=False,
        recovery_streak=0,
        params=RegimeGuardParams(),
    )

    assert exposure == 0.0
    assert active
    assert reasons == ("stale_holding_shock",)


def test_zero_risk_state_requires_two_healthy_observations_to_recover() -> None:
    params = RegimeGuardParams(recovery_days=2)
    first = regime_exposure(
        absolute_votes=3,
        votes=0,
        stale_holding_shock=False,
        portfolio_drawdown=-0.05,
        risk_off_active=True,
        recovery_streak=0,
        params=params,
    )
    second = regime_exposure(
        absolute_votes=3,
        votes=0,
        stale_holding_shock=False,
        portfolio_drawdown=-0.05,
        risk_off_active=first[1],
        recovery_streak=first[2],
        params=params,
    )

    assert first[:3] == (0.0, True, 1)
    assert second[:3] == (1.0, False, 0)


def test_drawdown_alone_does_not_force_exit_without_confirming_risk_state() -> None:
    exposure, active, _streak, _reasons = regime_exposure(
        absolute_votes=3,
        votes=0,
        stale_holding_shock=False,
        portfolio_drawdown=-0.30,
        risk_off_active=False,
        recovery_streak=0,
        params=RegimeGuardParams(),
    )

    assert exposure == 1.0
    assert not active


def test_conjunctive_guard_ignores_a_single_noisy_warning() -> None:
    exposure, reasons = conjunctive_exposure(
        risk_vote_reasons=("correlation_concentration",),
        stale_holding_shock=False,
        params=RegimeGuardParams(mode="conjunctive"),
    )

    assert exposure == 1.0
    assert reasons == ()


def test_conjunctive_guard_requires_orthogonal_confirmation() -> None:
    systemic, systemic_reasons = conjunctive_exposure(
        risk_vote_reasons=(
            "correlation_concentration",
            "downside_vol_acceleration",
        ),
        stale_holding_shock=False,
        params=RegimeGuardParams(mode="conjunctive"),
    )
    rotation, rotation_reasons = conjunctive_exposure(
        risk_vote_reasons=("weak_breadth", "unstable_leadership"),
        stale_holding_shock=False,
        params=RegimeGuardParams(mode="conjunctive"),
    )

    assert systemic == 0.0
    assert systemic_reasons == ("confirmed_systemic_stress",)
    assert rotation == pytest.approx(0.5)
    assert rotation_reasons == ("confirmed_rotation_trap",)


def test_tail_decay_requires_all_four_conditions() -> None:
    params = RegimeGuardParams(mode="confirmed_tail")

    assert tail_decay_confirmation(
        volatility_20d=0.50,
        momentum_decay_5d=-0.03,
        price_below_ma10=True,
        momentum_score=0.05,
        params=params,
    )
    assert not tail_decay_confirmation(
        volatility_20d=0.40,
        momentum_decay_5d=-0.03,
        price_below_ma10=True,
        momentum_score=0.05,
        params=params,
    )


def test_confirmed_tail_carries_shock_across_rotation_at_reduced_risk() -> None:
    params = RegimeGuardParams(mode="confirmed_tail")

    exposure, reasons = confirmed_tail_exposure(
        holding_shock=True,
        holding_changed=True,
        tail_decay_confirmed=False,
        params=params,
    )

    assert exposure == pytest.approx(0.70)
    assert reasons == ("shock_carryover_to_new_holding",)


def test_confirmed_tail_exits_when_shocked_holding_is_unchanged() -> None:
    params = RegimeGuardParams(mode="confirmed_tail")

    exposure, reasons = confirmed_tail_exposure(
        holding_shock=True,
        holding_changed=False,
        tail_decay_confirmed=False,
        params=params,
    )

    assert exposure == 0.0
    assert reasons == ("stale_holding_shock",)


def test_confirmed_tail_caps_high_volatility_decay_without_a_shock() -> None:
    params = RegimeGuardParams(mode="confirmed_tail")

    exposure, reasons = confirmed_tail_exposure(
        holding_shock=False,
        holding_changed=False,
        tail_decay_confirmed=True,
        params=params,
    )

    assert exposure == pytest.approx(0.70)
    assert reasons == ("high_volatility_momentum_decay",)


def test_variants_are_mechanism_ablations_not_weight_grid() -> None:
    variants = preregistered_variants()

    assert set(variants) == {
        "absolute_only",
        "breadth_only",
        "systemic_only",
        "leadership_only",
        "v4_rg",
    }
    assert {params.mode for params in variants.values()} == {
        "absolute",
        "breadth",
        "systemic",
        "leadership",
        "full",
    }


def test_neighborhood_is_center_plus_eight_one_axis_perturbations() -> None:
    neighbors = parameter_neighborhood()

    assert len(neighbors) == 9
    assert neighbors["center"] == RegimeGuardParams(mode="confirmed_tail")
    assert neighbors["volatility_80pct"].tail_volatility == pytest.approx(0.36)
    assert neighbors["exposure_120pct"].tail_exposure == pytest.approx(0.84)
    assert neighbors["decay_120pct"].tail_momentum_decay == pytest.approx(-0.024)
    assert neighbors["weakness_120pct"].tail_absolute_weak == pytest.approx(0.096)
