"""Research and overfitting audit for strict V3 plus the frozen V4 overlay.

Strict V3.5 deliberately excludes every V3-G gate and exemption.  It keeps the
strict five-day drop exclusion in both the scheduled selector and V4's factor
eligibility, then adds only V4's full-pool fast/slow handoff mechanism.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from scripts.exp_v3g_overfit_audit import (
        GateParams,
        gate_allows,
        run_gate_strategy,
        select_target,
    )
    from scripts.exp_v4_overfit_audit import (
        _annual_attribution,
        _bootstrap_interval,
        _event_evidence,
        _log_returns,
        _rolling_attribution,
        cscv_probability_of_backtest_overfitting,
        historical_candidate_params,
        local_gap_grid,
        newey_west_mean_test,
        white_reality_check,
    )
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v3g_overfit_audit import (
        GateParams,
        gate_allows,
        run_gate_strategy,
        select_target,
    )
    from exp_v4_overfit_audit import (
        _annual_attribution,
        _bootstrap_interval,
        _event_evidence,
        _log_returns,
        _rolling_attribution,
        cscv_probability_of_backtest_overfitting,
        historical_candidate_params,
        local_gap_grid,
        newey_west_mean_test,
        white_reality_check,
    )


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "strict_v35_audit.json"
STRATEGY_NAME = "严格V3.5"
STRATEGY_ID = "STRICT_V3_5"
DESIGN_DATE = "2026-08-12"
STRICT_V3_PARAMS = GateParams(strict_drop_filter=True, exempt_buffer=False)
STRICT_V35_PARAMS = v4.V4_PARAMS


def strict_drop_allows(close: np.ndarray) -> bool:
    """Apply strict V3's five-day single-drop exclusion with no release gate."""
    return bool(gate_allows(np.asarray(close, dtype=float), STRICT_V3_PARAMS))


def strict_v35_components() -> dict[str, bool | int]:
    """Return the frozen definition, including explicit V3-G exclusions."""
    return {
        "strict_drop_filter": True,
        "v4_full_pool_fast_slow": True,
        "v4_confirmation_hits": STRICT_V35_PARAMS.confirmation_hits,
        "v4_confirmation_window": STRICT_V35_PARAMS.confirmation_window,
        "v4_minimum_hold_days": STRICT_V35_PARAMS.minimum_hold_days,
        "v3g_ret60_gate": False,
        "v3g_momentum_release": False,
        "v3g_holding_buffer_exemption": False,
        "v3g_realtime_filter_approximation": False,
    }


def strict_gate_payload() -> dict[str, bool | float | int]:
    """Expose only active strict-V3 fields, not dormant V3-G dataclass fields."""
    return {
        "drop_threshold": STRICT_V3_PARAMS.drop_threshold,
        "drop_lookback": STRICT_V3_PARAMS.drop_lookback,
        "strict_drop_filter": True,
    }


def _scenario_factor(
    code: str,
    *,
    eligible: bool,
    slow: float,
    return_3d: float,
    return_5d: float,
) -> v4.AssetFactors:
    return v4.AssetFactors(
        code=code,
        eligible=eligible,
        slow_momentum=slow,
        return_3d=return_3d,
        return_5d=return_5d,
        acceleration_5d=0.0,
        trend_strength=0.01,
        vol_adjusted_5d=0.0,
        drawdown_5d=min(return_3d, return_5d, 0.0),
    )


def replay_documented_shock_scenario() -> dict[str, Any]:
    """Mechanically replay the user's three-day silver/gold shock narrative."""
    daily_factors = (
        {
            "518880": _scenario_factor(
                "518880", eligible=True, slow=0.06,
                return_3d=0.01, return_5d=0.03,
            ),
            "161226": _scenario_factor(
                "161226", eligible=False, slow=0.14,
                return_3d=-0.05, return_5d=0.08,
            ),
        },
        {
            "518880": _scenario_factor(
                "518880", eligible=True, slow=0.07,
                return_3d=0.03, return_5d=0.04,
            ),
            "161226": _scenario_factor(
                "161226", eligible=False, slow=0.05,
                return_3d=-0.12, return_5d=-0.08,
            ),
        },
        {
            "518880": _scenario_factor(
                "518880", eligible=True, slow=0.09,
                return_3d=0.06, return_5d=0.07,
            ),
            "161226": _scenario_factor(
                "161226", eligible=False, slow=0.01,
                return_3d=-0.15, return_5d=-0.15,
            ),
        },
    )
    decisions = []
    for day, (factors, hits) in enumerate(
        zip(daily_factors, (0, 1, 2), strict=True), start=1
    ):
        decision = v4.decide_full_pool_handoff(
            holding="161226",
            factors=factors,
            params=STRICT_V35_PARAMS,
            signal_hits=hits,
            days_since_rotation=10_000,
        )
        decisions.append({"day": day, **asdict(decision)})
    silver_wealth = float(np.prod([0.95, 0.92, 0.97]))
    gold_wealth = float(np.prod([1.01, 1.02, 1.03]))
    return {
        "scenario": "silver daily -5%/-8%/-3%; gold +1%/+2%/+3%",
        "decisions": decisions,
        "silver_cumulative_return": silver_wealth - 1.0,
        "gold_cumulative_return": gold_wealth - 1.0,
        "gold_relative_wealth_advantage": gold_wealth / silver_wealth - 1.0,
        "known_defect_fixed": False,
        "mechanism": (
            "strict eligibility blocks silver as a new target but does not force-sell "
            "an ineligible holding; V4 still waits for fast/slow consensus and two hits"
        ),
    }


def _component_payload(params: v4.FullPoolParams) -> dict[str, bool | int]:
    components = strict_v35_components()
    components["v4_full_pool_fast_slow"] = params.enabled
    return components


def run_strict_strategy(
    data: dict[str, Any],
    params: v4.FullPoolParams,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Replay strict V3 with an optional V4 overlay and no V3-G behavior."""
    original_select = rq.select_target
    original_drop = rq.check_single_day_drop
    original_realtime_filter = rr.apply_server_realtime_filter

    def strict_select(
        inner_data: dict[str, Any],
        idx_map: dict[str, int],
        holding: str | None,
    ) -> tuple[str, list[tuple[str, float]], float, bool]:
        return cast(
            "tuple[str, list[tuple[str, float]], float, bool]",
            select_target(inner_data, idx_map, holding, STRICT_V3_PARAMS),
        )

    def no_realtime_gate(
        _data: dict[str, Any],
        _idx_map: dict[str, int],
        candidates: list[tuple[str, float]],
    ) -> tuple[list[tuple[str, float]], list[dict[str, Any]]]:
        return candidates, []

    try:
        rq.select_target = strict_select
        rq.check_single_day_drop = strict_drop_allows
        rr.apply_server_realtime_filter = no_realtime_gate
        result: dict[str, Any] = run_full_pool_strategy(
            data,
            params,
            cost_multiplier=cost_multiplier,
        )
    finally:
        rq.select_target = original_select
        rq.check_single_day_drop = original_drop
        rr.apply_server_realtime_filter = original_realtime_filter

    result["strategy_id"] = STRATEGY_ID if params.enabled else "STRICT_V3"
    result["strategy_name"] = STRATEGY_NAME if params.enabled else "严格V3"
    result["strict_gate_params"] = strict_gate_payload()
    result["strict_components"] = _component_payload(params)
    return result


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": result.get("strategy_id"),
        "params": result["params"],
        "metrics": result["metrics"],
    }


def _rename_annual_fields(annual: dict[str, Any]) -> dict[str, Any]:
    for row in annual["years"]:
        row["strict_v3_return"] = row.pop("v3g_return")
        row["strict_v35_return"] = row.pop("v4_return")
        row["strict_v35_relative_to_strict_v3"] = row.pop(
            "v4_relative_to_v3g"
        )
    return annual


def _named_selection_counts(
    result: dict[str, Any], names: list[str]
) -> dict[str, Any]:
    result["selection_counts"] = {
        names[int(key)]: value
        for key, value in result["selection_counts"].items()
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="严格V3.5 research audit")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 500:
        raise ValueError("use at least 500 bootstrap samples")
    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("strict V3.5 audit requires all downsize layers disabled")

    data = rq.load_data()
    strict_cache: dict[
        tuple[v4.FullPoolParams, float], dict[str, Any]
    ] = {}

    def evaluate_strict(
        params: v4.FullPoolParams, cost: float = 1.0
    ) -> dict[str, Any]:
        key = (params, cost)
        if key not in strict_cache:
            strict_cache[key] = run_strict_strategy(
                data, params, cost_multiplier=cost
            )
        return strict_cache[key]

    strict_v3 = evaluate_strict(v4.FullPoolParams.disabled())
    strict_v35 = evaluate_strict(STRICT_V35_PARAMS)

    parameterized_strict_v3 = run_gate_strategy(data, STRICT_V3_PARAMS)
    strict_parity = bool(np.allclose(
        strict_v3["equity_curve"]["equity"],
        parameterized_strict_v3["equity_curve"]["equity"],
    ))
    if not strict_parity:
        raise RuntimeError("strict V3 baseline parity failed")

    dates, strict_v3_returns = _log_returns(strict_v3)
    selected_dates, strict_v35_returns = _log_returns(strict_v35)
    if dates.tolist() != selected_dates.tolist():
        raise RuntimeError("strict V3 and strict V3.5 date alignment drift")
    selected_excess = strict_v35_returns - strict_v3_returns
    data_end = str(dates[-1])[:10]

    historical_params = historical_candidate_params()
    historical_results = {
        name: evaluate_strict(params)
        for name, params in historical_params.items()
    }
    historical_names = list(historical_results)
    candidate_excess: list[np.ndarray] = []
    candidate_absolute: list[np.ndarray] = [strict_v3_returns]
    absolute_names = ["strict_v3"]
    for name, result in historical_results.items():
        candidate_dates, candidate_returns = _log_returns(result)
        if candidate_dates.tolist() != dates.tolist():
            raise RuntimeError(f"candidate date drift: {name}")
        candidate_excess.append(candidate_returns - strict_v3_returns)
        candidate_absolute.append(candidate_returns)
        absolute_names.append(name)
    excess_matrix = np.column_stack(candidate_excess)
    absolute_matrix = np.column_stack(candidate_absolute)
    selected_column = historical_names.index("consensus_strict")

    white = {
        str(block): white_reality_check(
            excess_matrix,
            selected_column=selected_column,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260900 + block,
        )
        for block in (5, 20, 60)
    }
    consensus_candidate_names = [
        "consensus",
        "consensus_confirm2",
        "consensus_strict",
        "consensus_very_strict",
    ]
    consensus_columns = [
        historical_names.index(name) for name in consensus_candidate_names
    ]
    consensus_white = {
        str(block): white_reality_check(
            excess_matrix[:, consensus_columns],
            selected_column=consensus_candidate_names.index("consensus_strict"),
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260910 + block,
        )
        for block in (5, 20, 60)
    }

    bootstrap = {
        str(block): _bootstrap_interval(
            selected_excess,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260920 + block,
        )
        for block in (5, 20, 60)
    }
    newey_west = {
        str(lag): newey_west_mean_test(selected_excess, max_lag=lag)
        for lag in (5, 20, 60)
    }
    all_pbo = _named_selection_counts(
        cscv_probability_of_backtest_overfitting(absolute_matrix, slices=8),
        absolute_names,
    )
    consensus_absolute_names = ["strict_v3", *consensus_candidate_names]
    consensus_absolute_columns = [
        absolute_names.index(name) for name in consensus_absolute_names
    ]
    consensus_pbo = _named_selection_counts(
        cscv_probability_of_backtest_overfitting(
            absolute_matrix[:, consensus_absolute_columns], slices=8
        ),
        consensus_absolute_names,
    )

    strict_v3_final = float(strict_v3["metrics"]["final_value"])
    surface_rows: list[dict[str, Any]] = []
    for name, params in local_gap_grid().items():
        result = evaluate_strict(params)
        metrics = result["metrics"]
        surface_rows.append({
            "name": name,
            "params": result["params"],
            "final_value": float(metrics["final_value"]),
            "relative_to_strict_v3": float(
                metrics["final_value"] / strict_v3_final - 1.0
            ),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "early_rotations": int(metrics["early_rotations"]),
        })
    relative_values = np.asarray([
        row["relative_to_strict_v3"] for row in surface_rows
    ])
    center_name = "slow1.0_fast51.0_fast31.0"
    center_final = next(
        row["final_value"] for row in surface_rows if row["name"] == center_name
    )
    ordered_finals = sorted(
        (row["final_value"] for row in surface_rows), reverse=True
    )
    surface = {
        "definition": "3x3x3 independent gap perturbation at 80%/100%/120%",
        "points": len(surface_rows),
        "points_beating_strict_v3": int(np.sum(relative_values > 0.0)),
        "beat_rate": float(np.mean(relative_values > 0.0)),
        "minimum_relative_to_strict_v3": float(np.min(relative_values)),
        "median_relative_to_strict_v3": float(np.median(relative_values)),
        "maximum_relative_to_strict_v3": float(np.max(relative_values)),
        "strict_v35_center_rank_by_final": ordered_finals.index(center_final) + 1,
        "rows": surface_rows,
    }

    structural_params = {
        "hold_2": replace(STRICT_V35_PARAMS, minimum_hold_days=2),
        "hold_3": STRICT_V35_PARAMS,
        "hold_4": replace(STRICT_V35_PARAMS, minimum_hold_days=4),
        "confirmation_1": replace(STRICT_V35_PARAMS, confirmation_hits=1),
        "confirmation_2": STRICT_V35_PARAMS,
    }
    structural = {
        name: _compact(evaluate_strict(params))
        for name, params in structural_params.items()
    }

    cost_pressure: dict[str, dict[str, Any]] = {
        f"{cost:.0f}x": {
            "strict_v3": _compact(
                evaluate_strict(v4.FullPoolParams.disabled(), cost)
            ),
            "strict_v35": _compact(evaluate_strict(STRICT_V35_PARAMS, cost)),
        }
        for cost in (0.0, 1.0, 2.0, 3.0)
    }
    for row in cost_pressure.values():
        row["strict_v35_relative_final"] = float(
            row["strict_v35"]["metrics"]["final_value"]
            / row["strict_v3"]["metrics"]["final_value"]
            - 1.0
        )
    segments = {
        label: {
            "strict_v3": rr.segment_metrics(strict_v3["equity_curve"], start, end),
            "strict_v35": rr.segment_metrics(
                strict_v35["equity_curve"], start, end
            ),
        }
        for label, start, end in (
            ("retrospective_2020_2023", "2020-06-19", "2023-12-29"),
            ("retrospective_2024_current", "2024-01-01", data_end),
        )
    }

    current_v3g = run_full_pool_strategy(data, v4.FullPoolParams.disabled())
    current_v4 = run_full_pool_strategy(data, v4.V4_PARAMS)
    four_way = {
        "strict_v3": _compact(strict_v3),
        "strict_v35": _compact(strict_v35),
        "current_v3g": _compact(current_v3g),
        "current_v4": _compact(current_v4),
    }

    annual = _rename_annual_fields(_annual_attribution(
        dates, strict_v3_returns, strict_v35_returns
    ))
    rolling = _rolling_attribution(dates, selected_excess)
    events = _event_evidence(strict_v35["rotation_events"])
    terminal_relative = float(
        strict_v35["metrics"]["final_value"] / strict_v3_final - 1.0
    )

    statistical_pass = bool(
        float(white["20"]["p_value"]) < 0.05
        and float(bootstrap["20"]["ci95"][0]) > 0.0
    )
    neighborhood_pass = bool(cast("float", surface["beat_rate"]) >= 0.80)
    pbo_warning = bool(
        float(all_pbo["pbo"]) >= 0.50
        or float(consensus_pbo["pbo"]) >= 0.50
    )
    temporal_warning = bool(
        annual["positive_relative_years"] < annual["year_count"] / 2
        or rolling["positive_rate"] < 0.50
        or float(annual["post_2024_share_of_total_relative_log_return"]) > 0.80
    )
    cost_stress_pass = bool(all(
        float(cost_pressure[label]["strict_v35_relative_final"]) > 0.0
        and float(
            cost_pressure[label]["strict_v35"]["metrics"]["max_drawdown"]
        ) > -0.50
        for label in ("2x", "3x")
    ))
    if (
        not statistical_pass
        or not neighborhood_pass
        or pbo_warning
        or temporal_warning
        or not cost_stress_pass
    ):
        classification = "high_overfit_risk"
    else:
        classification = "medium_high_overfit_risk"

    payload = {
        "meta": {
            "strategy_name": STRATEGY_NAME,
            "strategy_id": STRATEGY_ID,
            "definition": "strict V3 base plus frozen production V4 overlay only",
            "design_date": DESIGN_DATE,
            "data_start": str(dates[0])[:10],
            "data_end": data_end,
            "observations": len(dates),
            "live_oos_observations": 0,
            "historical_oos_label_valid": False,
            "historical_oos_reason": (
                "strict V3.5 was proposed after the full history and prior V4/V3-G "
                "audits were observed"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "minimum_reused_v4_candidate_trials": len(historical_names),
            "trial_count_is_lower_bound": True,
            "strict_v3_parity": strict_parity,
            "strict_components": strict_v35_components(),
            "strict_gate_params": strict_gate_payload(),
            "v4_overlay_params": asdict(STRICT_V35_PARAMS),
            "signal_time_rule": "all factor arrays are sliced through decision day T",
            "execution_model_warning": (
                "historical official close is a same-price 14:50 execution proxy; "
                "historical intraday snapshots are unavailable"
            ),
        },
        "four_way_comparison": four_way,
        "incremental_to_strict_v3": {
            "terminal_relative": terminal_relative,
            "annual": annual,
            "rolling_252d": rolling,
            "newey_west": newey_west,
            "moving_block_bootstrap": bootstrap,
            "white_reality_check": white,
            "white_reality_check_consensus_family": consensus_white,
        },
        "selection_bias": {
            "candidate_names": historical_names,
            "all_recorded_candidates_cscv": all_pbo,
            "consensus_family_cscv": consensus_pbo,
        },
        "parameter_robustness": {
            "gap_surface": surface,
            "structural_neighbors": structural,
        },
        "cost_pressure": cost_pressure,
        "retrospective_segments": segments,
        "rotation_event_evidence": events,
        "documented_shock_scenario": replay_documented_shock_scenario(),
        "assessment": {
            "classification": classification,
            "research_status": "research_candidate_only",
            "statistical_increment_pass": statistical_pass,
            "parameter_neighborhood_pass": neighborhood_pass,
            "pbo_warning": pbo_warning,
            "temporal_concentration_warning": temporal_warning,
            "cost_stress_pass": cost_stress_pass,
            "cost_path_warning": (
                "cost changes portfolio drawdown, which changes risk actions and later "
                "trades; the 3x collapse is a path-dependent stress failure"
            ),
            "production_recommendation": (
                "do not replace production from this retrospective result; freeze the "
                "candidate and collect untouched shadow observations"
            ),
        },
    }

    print("\n严格V3.5 audit")
    print(
        f"data={payload['meta']['data_start']}..{data_end} n={len(dates)} "
        "live_OOS=0"
    )
    print("\nstrategy             final    CAGR Sharpe     MDD legs early")
    for name, result in four_way.items():
        metrics = result["metrics"]
        print(
            f"{name:<18} {metrics['final_value']:>9,.0f} "
            f"{metrics['cagr']:>6.1%} {metrics['sharpe']:>6.3f} "
            f"{metrics['max_drawdown']:>7.1%} {metrics['trade_legs']:>4} "
            f"{metrics['early_rotations']:>5}"
        )
    print(
        f"\nstrict V3.5 relative={terminal_relative:+.1%}; "
        f"White p20={white['20']['p_value']:.3f}; "
        f"bootstrap95={bootstrap['20']['ci95'][0]:+.1%}.."
        f"{bootstrap['20']['ci95'][1]:+.1%}"
    )
    print(
        f"PBO all={all_pbo['pbo']:.1%} consensus={consensus_pbo['pbo']:.1%}; "
        f"local beat={surface['points_beating_strict_v3']}/"
        f"{surface['points']} center rank="
        f"{surface['strict_v35_center_rank_by_final']}/{surface['points']}"
    )
    print(
        f"positive years={annual['positive_relative_years']}/{annual['year_count']}; "
        f"rolling wins={rolling['positive_windows']}/{rolling['windows']}; "
        f"assessment={classification}"
    )
    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
