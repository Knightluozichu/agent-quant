"""Tests for the theory-first V4 risk-budget research overlay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import exp_v4_risk_budget as rb  # noqa: E402


def test_cppi_exposure_is_full_at_peak_and_zero_at_floor() -> None:
    params = rb.RiskBudgetParams(mode="cppi")

    assert rb.cppi_exposure(100.0, 100.0, params) == 1.0
    assert np.isclose(rb.cppi_exposure(95.0, 100.0, params), 15.0 / 19.0)
    assert rb.cppi_exposure(80.0, 100.0, params) == 0.0
    assert rb.cppi_exposure(70.0, 100.0, params) == 0.0


def test_shock_brake_caps_next_period_without_using_future_return() -> None:
    params = rb.RiskBudgetParams(mode="shock")

    assert rb.shock_exposure(-0.049, -0.099, params) == 1.0
    assert rb.shock_exposure(-0.05, -0.08, params) == 0.5
    assert rb.shock_exposure(-0.01, -0.10, params) == 0.0


def test_risk_budget_takes_the_most_conservative_observable_cap() -> None:
    params = rb.RiskBudgetParams(mode="risk_budget")

    exposure = rb.target_exposure(
        wealth=95.0,
        peak=100.0,
        one_day=-0.05,
        three_day=-0.08,
        annualized_volatility=0.20,
        params=params,
    )

    assert exposure == 0.5


def test_documented_three_day_shock_is_reduced_but_not_erased() -> None:
    scenario = rb.replay_documented_shock()

    assert scenario["exposure_used"] == [1.0, 0.5, 0.0]
    assert np.isclose(scenario["v4_holding_return"], -0.15222)
    assert np.isclose(scenario["risk_budget_return"], -0.088)
    assert scenario["risk_budget_return"] > -0.10
    assert scenario["first_gap_is_unavoidable"] is True


def test_preregistered_candidates_do_not_scan_parameters() -> None:
    candidates = rb.preregistered_candidates()

    assert list(candidates) == ["vol30", "cppi20", "shock_brake", "v4_risk_budget"]
    assert candidates["vol30"] == rb.RiskBudgetParams(mode="volatility")
    assert candidates["cppi20"] == rb.RiskBudgetParams(mode="cppi")
    assert candidates["shock_brake"] == rb.RiskBudgetParams(mode="shock")
    assert candidates["v4_risk_budget"] == rb.RiskBudgetParams(mode="risk_budget")


def test_position_budget_has_a_five_percent_deductible_and_ten_percent_limit() -> None:
    params = rb.RiskBudgetParams(mode="position_budget")

    assert rb.position_budget_exposure(-0.049, params) == 1.0
    assert rb.position_budget_exposure(-0.05, params) == 1.0
    assert np.isclose(rb.position_budget_exposure(-0.075, params), 0.5)
    assert rb.position_budget_exposure(-0.10, params) == 0.0


def test_position_budget_resets_only_after_v4_has_changed_the_holding() -> None:
    params = rb.phase2_position_budget_candidate()

    assert rb.position_target_exposure(
        position_drawdown=-0.08,
        one_day=-0.06,
        three_day=-0.11,
        holding_changed=False,
        params=params,
    ) == 0.0
    assert rb.position_target_exposure(
        position_drawdown=-0.08,
        one_day=-0.06,
        three_day=-0.11,
        holding_changed=True,
        params=params,
    ) == 1.0


def test_episode_budget_persists_across_holdings_until_half_recovery() -> None:
    params = rb.phase3_episode_budget_candidate()

    assert rb.episode_target_exposure(
        portfolio_drawdown=-0.09,
        episode_active=False,
        shock_cap=1.0,
        params=params,
    ) == (1.0, False)
    assert rb.episode_target_exposure(
        portfolio_drawdown=-0.10,
        episode_active=False,
        shock_cap=1.0,
        params=params,
    ) == (0.5, True)
    assert rb.episode_target_exposure(
        portfolio_drawdown=-0.08,
        episode_active=True,
        shock_cap=1.0,
        params=params,
    ) == (0.5, True)
    assert rb.episode_target_exposure(
        portfolio_drawdown=-0.05,
        episode_active=True,
        shock_cap=1.0,
        params=params,
    ) == (1.0, False)
    assert rb.episode_target_exposure(
        portfolio_drawdown=-0.20,
        episode_active=True,
        shock_cap=1.0,
        params=params,
    ) == (0.25, True)


def test_episode_budget_never_overrides_a_stricter_asset_shock_cap() -> None:
    exposure, active = rb.episode_target_exposure(
        portfolio_drawdown=-0.12,
        episode_active=True,
        shock_cap=0.0,
        params=rb.phase3_episode_budget_candidate(),
    )

    assert exposure == 0.0
    assert active is True


def test_constant_risk_budget_is_derived_from_twenty_percent_mdd_reduction() -> None:
    candidates = rb.phase4_constant_budget_candidates()

    assert list(candidates) == ["v4_constant80", "v4_constant80_shock"]
    assert candidates["v4_constant80"].constant_exposure == 0.80
    assert candidates["v4_constant80_shock"].constant_exposure == 0.80
    assert rb.constant_target_exposure(
        one_day=-0.01,
        three_day=-0.02,
        params=candidates["v4_constant80_shock"],
    ) == 0.80
    assert rb.constant_target_exposure(
        one_day=-0.05,
        three_day=-0.08,
        params=candidates["v4_constant80_shock"],
    ) == 0.50
    assert rb.constant_target_exposure(
        one_day=-0.01,
        three_day=-0.10,
        params=candidates["v4_constant80_shock"],
    ) == 0.0


def test_constant_eighty_shock_budget_limits_documented_path_below_eight_percent() -> None:
    scenario = rb.replay_documented_shock()

    assert scenario["constant80_shock_exposure_used"] == [0.8, 0.5, 0.0]
    assert np.isclose(scenario["constant80_shock_return"], -0.0784)


def test_volatility_shock_composition_adds_no_new_parameter() -> None:
    params = rb.phase5_volatility_shock_candidate()

    assert params.mode == "volatility_shock"
    assert params.volatility_target == 0.30
    assert params.ewma_decay == 0.94
    assert rb.target_exposure(
        wealth=100.0,
        peak=100.0,
        one_day=-0.01,
        three_day=-0.02,
        annualized_volatility=0.60,
        params=params,
    ) == 0.5
    assert rb.target_exposure(
        wealth=100.0,
        peak=100.0,
        one_day=-0.05,
        three_day=-0.08,
        annualized_volatility=0.20,
        params=params,
    ) == 0.5
    assert rb.target_exposure(
        wealth=100.0,
        peak=100.0,
        one_day=-0.01,
        three_day=-0.10,
        annualized_volatility=0.20,
        params=params,
    ) == 0.0


def test_volatility_shock_robustness_is_a_seven_point_one_axis_neighborhood() -> None:
    neighbors = rb.volatility_shock_neighbors()

    assert list(neighbors) == [
        "center",
        "volatility_80pct",
        "volatility_120pct",
        "decay_0.90",
        "decay_0.97",
        "shock_80pct",
        "shock_120pct",
    ]
    assert len(set(neighbors.values())) == 7
    assert neighbors["center"] == rb.phase5_volatility_shock_candidate()
