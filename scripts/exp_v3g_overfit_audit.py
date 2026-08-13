"""Mathematical overfitting audit for production V3-G.

The audit separates the G gate's incremental evidence from the already tuned
V3 momentum core.  Production files are not modified; a parameterized selector
is replayed through the canonical daily research/risk engine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from scripts.exp_v4_overfit_audit import (
        _annual_attribution,
        _bootstrap_interval,
        _log_returns,
        _rolling_attribution,
        cscv_probability_of_backtest_overfitting,
        newey_west_mean_test,
        white_reality_check,
    )
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v4_overfit_audit import (
        _annual_attribution,
        _bootstrap_interval,
        _log_returns,
        _rolling_attribution,
        cscv_probability_of_backtest_overfitting,
        newey_west_mean_test,
        white_reality_check,
    )


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3g_overfit_audit.json"
V3G_LAUNCH_DATE = "2026-08-10"
RECORDED_VARIANT_EVALUATIONS = 111


@dataclass(frozen=True)
class GateParams:
    """Typed research parameterization of the current V3-G selector."""

    drop_threshold: float = 0.03
    drop_lookback: int = 5
    ret60_threshold: float = 0.01
    gate_momentum_short: int = 10
    gate_momentum_long: int = 20
    gate_short_weight: float = 0.5
    gate_momentum_threshold: float = 0.0
    exempt_buffer: bool = True
    strict_drop_filter: bool = False
    score_short: int = 10
    score_long: int = 20
    score_short_weight: float = 0.5


def _period_return(close: np.ndarray, period: int) -> float:
    if len(close) <= period or close[-period - 1] <= 0.0:
        return 0.0
    return float(close[-1] / close[-period - 1] - 1.0)


def _has_drop(close: np.ndarray, params: GateParams) -> bool:
    if len(close) < params.drop_lookback + 1:
        return False
    return any(
        close[index] / close[index - 1] - 1.0 < -params.drop_threshold
        for index in range(-params.drop_lookback, 0)
    )


def gate_allows(close: np.ndarray, params: GateParams) -> bool:
    """Return whether the drop filter admits one candidate at T."""
    if not _has_drop(close, params):
        return True
    if params.strict_drop_filter:
        return False
    if len(close) <= 61:
        return False
    if _period_return(close, 60) >= params.ret60_threshold:
        return False
    short = _period_return(close, params.gate_momentum_short)
    long = _period_return(close, params.gate_momentum_long)
    momentum = params.gate_short_weight * short + (1.0 - params.gate_short_weight) * long
    return momentum > params.gate_momentum_threshold


def _score(close: np.ndarray, params: GateParams) -> float:
    return params.score_short_weight * _period_return(close, params.score_short) + (
        1.0 - params.score_short_weight
    ) * _period_return(close, params.score_long)


def select_target(
    data: dict[str, Any],
    idx_map: dict[str, int],
    holding: str | None,
    params: GateParams,
) -> tuple[str, list[tuple[str, float]], float, bool]:
    """Parameterized, T-sliced equivalent of production V3-G selection."""
    a_share_weak = (
        rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
        if rq.USE_A_SHARE_FILTER
        else False
    )
    candidates: list[tuple[str, float]] = []
    for code in rq.ETF_POOL:
        if code not in idx_map or (code == rq.A_SHARE_ETF and a_share_weak):
            continue
        index = idx_map[code]
        frame = data[code]
        close = np.asarray(frame["close"].values[: index + 1], dtype=float)
        volume = np.asarray(frame["volume"].values[: index + 1], dtype=float)
        if len(close) < 121:
            continue
        if rq.USE_SHORT_MOM_FILTER and not rq.check_short_momentum(close):
            continue
        if rq.USE_VOL_SPIKE_FILTER and not rq.check_volume_spike(volume, close):
            continue
        if rq.USE_DROP_FILTER and not gate_allows(close, params):
            continue
        if (
            rq.USE_LONG_MOM_FILTER
            and code in ("513100", "159915")
            and len(close) > rq.LONG_MOM_PERIOD
            and _period_return(close, rq.LONG_MOM_PERIOD) < 0.0
        ):
            continue
        score = _score(close, params)
        if score > 0.0:
            candidates.append((code, score))
    candidates.sort(key=lambda item: -item[1])

    if rq.USE_CATEGORY_SWITCH and candidates:
        score_map = dict(candidates)
        category_scores = {
            category: float(np.mean([score_map[code] for code in codes if code in score_map]))
            for category, codes in rq.CATEGORIES.items()
            if any(code in score_map for code in codes)
        }
        if category_scores:
            best_category = max(category_scores, key=category_scores.__getitem__)
            category_codes = set(rq.CATEGORIES[best_category])
            candidates = [item for item in candidates if item[0] in category_codes]

    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0.0
    threshold = 0.0 if best_score > 0.10 else 0.05
    if (
        params.exempt_buffer
        and not params.strict_drop_filter
        and holding
        and holding != rq.DEFENSE
        and holding in idx_map
    ):
        held_close = np.asarray(data[holding]["close"].values[: idx_map[holding] + 1], dtype=float)
        if _has_drop(held_close, params) and gate_allows(held_close, params):
            threshold = 0.0

    if holding and holding != rq.DEFENSE:
        current_score = dict(candidates).get(holding, -999.0)
        if current_score > 0.0:
            target = best_target if best_score > current_score + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target, candidates, best_score, a_share_weak


def historical_score_periods() -> list[tuple[int, int]]:
    short = (3, 5, 8, 10, 13, 15, 20)
    long = (13, 15, 20, 25, 30)
    return [(left, right) for left in short for right in long if right > left]


def local_gate_neighbors() -> dict[str, GateParams]:
    candidates: list[tuple[str, GateParams]] = []
    for threshold in (0.0, 0.005, 0.01, 0.015, 0.02):
        candidates.append((f"ret60_{threshold:.3f}", GateParams(ret60_threshold=threshold)))
    for threshold in (0.025, 0.03, 0.035):
        candidates.append((f"drop_{threshold:.3f}", GateParams(drop_threshold=threshold)))
    for lookback in (4, 5, 6):
        candidates.append((f"lookback_{lookback}", GateParams(drop_lookback=lookback)))
    for threshold in (-0.01, 0.0, 0.01):
        candidates.append(
            (f"gate_momentum_{threshold:+.2f}", GateParams(gate_momentum_threshold=threshold))
        )
    candidates.append(("without_buffer_exemption", GateParams(exempt_buffer=False)))
    unique: dict[GateParams, str] = {}
    for name, params in candidates:
        unique.setdefault(params, name)
    return {name: params for params, name in unique.items()}


def run_gate_strategy(data: dict[str, Any], params: GateParams) -> dict[str, Any]:
    """Replay one selector variant through the canonical V3-G daily engine."""
    original_select = rq.select_target
    original_realtime_filter = rr.apply_server_realtime_filter
    decisions: list[dict[str, Any]] = []

    def wrapped(
        inner_data: dict[str, Any],
        idx_map: dict[str, int],
        holding: str | None,
    ) -> tuple[str, list[tuple[str, float]], float, bool]:
        result = select_target(inner_data, idx_map, holding, params)
        code = next((item for item in rq.ETF_POOL if item in idx_map), None)
        trade_date = inner_data[code].iloc[idx_map[code]]["trade_date"] if code else None
        decisions.append(
            {
                "date": str(trade_date),
                "holding": holding,
                "target": result[0],
            }
        )
        return result

    try:
        rq.select_target = wrapped
        rr.apply_server_realtime_filter = lambda _data, _idx_map, candidates: (candidates, [])
        result: dict[str, Any] = run_full_pool_strategy(data, v4.FullPoolParams.disabled())
    finally:
        rq.select_target = original_select
        rr.apply_server_realtime_filter = original_realtime_filter
    result["gate_params"] = asdict(params)
    result["decision_log"] = decisions
    return result


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate_params": result["gate_params"],
        "metrics": result["metrics"],
    }


def _decision_divergences(
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    eligible_dates: set[str],
) -> int:
    targets = {row["date"]: row["target"] for row in baseline}
    return sum(
        targets.get(row["date"]) != row["target"]
        for row in challenger
        if row["date"] in targets and row["date"] in eligible_dates
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-G mathematical overfit audit")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 500:
        raise ValueError("use at least 500 bootstrap samples")
    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V3-G audit requires all downsize layers disabled")

    data = rq.load_data()
    canonical = run_full_pool_strategy(data, v4.FullPoolParams.disabled())
    cache: dict[GateParams, dict[str, Any]] = {}

    def evaluate(params: GateParams) -> dict[str, Any]:
        if params not in cache:
            cache[params] = run_gate_strategy(data, params)
        return cache[params]

    production_params = GateParams()
    strict_params = GateParams(strict_drop_filter=True, exempt_buffer=False)
    production = evaluate(production_params)
    strict = evaluate(strict_params)
    if not np.allclose(
        canonical["equity_curve"]["equity"],
        production["equity_curve"]["equity"],
    ):
        raise RuntimeError("parameterized V3-G does not match canonical V3-G")

    dates, strict_returns = _log_returns(strict)
    production_dates, production_returns = _log_returns(production)
    if dates.tolist() != production_dates.tolist():
        raise RuntimeError("V3 and V3-G date alignment drift")
    excess = production_returns - strict_returns
    data_end = str(dates[-1])[:10]
    live_oos = int(np.sum(np.asarray([str(date)[:10] > V3G_LAUNCH_DATE for date in dates])))

    ret60_params = {
        f"ret60_{threshold:.2f}": GateParams(ret60_threshold=threshold)
        for threshold in (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
    }
    ret60_results = {name: evaluate(params) for name, params in ret60_params.items()}
    ret60_names = list(ret60_results)
    ret60_excess: list[np.ndarray] = []
    ret60_absolute: list[np.ndarray] = [strict_returns]
    for result in ret60_results.values():
        _, candidate_returns = _log_returns(result)
        ret60_excess.append(candidate_returns - strict_returns)
        ret60_absolute.append(candidate_returns)
    ret60_excess_matrix = np.column_stack(ret60_excess)
    ret60_absolute_matrix = np.column_stack(ret60_absolute)
    selected_column = ret60_names.index("ret60_0.01")
    white = {
        str(block): white_reality_check(
            ret60_excess_matrix,
            selected_column=selected_column,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260830 + block,
        )
        for block in (5, 20, 60)
    }
    gate_pbo = cscv_probability_of_backtest_overfitting(ret60_absolute_matrix, slices=8)
    gate_names = ["strict_v3", *ret60_names]
    gate_pbo["selection_counts"] = {
        gate_names[int(key)]: value for key, value in gate_pbo["selection_counts"].items()
    }

    bootstrap = {
        str(block): _bootstrap_interval(
            excess,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260840 + block,
        )
        for block in (5, 20, 60)
    }
    hac = {str(lag): newey_west_mean_test(excess, max_lag=lag) for lag in (5, 20, 60)}
    annual = _annual_attribution(dates, strict_returns, production_returns)
    for row in annual["years"]:
        row["strict_v3_return"] = row.pop("v3g_return")
        row["v3g_return"] = row.pop("v4_return")
        row["v3g_relative_to_strict_v3"] = row.pop("v4_relative_to_v3g")
    rolling = _rolling_attribution(dates, excess)
    rebalance_dates = {str(date) for date in rr.common_dates(data)[130:][:: rq.REBALANCE_DAYS]}

    strict_final = float(strict["metrics"]["final_value"])
    local_rows: list[dict[str, Any]] = []
    for name, params in local_gate_neighbors().items():
        result = evaluate(params)
        metrics = result["metrics"]
        local_rows.append(
            {
                "name": name,
                "params": asdict(params),
                "final_value": float(metrics["final_value"]),
                "relative_to_strict_v3": float(metrics["final_value"] / strict_final - 1.0),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
            }
        )
    local_relative = np.asarray([row["relative_to_strict_v3"] for row in local_rows], dtype=float)

    score_params: dict[str, GateParams] = {
        f"period_{short}_{long}": replace(production_params, score_short=short, score_long=long)
        for short, long in historical_score_periods()
    }
    for weight in (0.3, 0.4, 0.5, 0.6, 0.7):
        score_params[f"weight_{weight:.1f}"] = replace(production_params, score_short_weight=weight)
    unique_score_params: dict[GateParams, str] = {}
    for name, params in score_params.items():
        unique_score_params.setdefault(params, name)
    score_params = {name: params for params, name in unique_score_params.items()}
    score_results = {name: evaluate(params) for name, params in score_params.items()}
    score_names = list(score_results)
    score_returns = []
    score_rows = []
    for name, result in score_results.items():
        _, returns = _log_returns(result)
        score_returns.append(returns)
        metrics = result["metrics"]
        score_rows.append(
            {
                "name": name,
                "params": result["gate_params"],
                "final_value": float(metrics["final_value"]),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
            }
        )
    score_matrix = np.column_stack(score_returns)
    score_pbo = cscv_probability_of_backtest_overfitting(score_matrix, slices=8)
    score_pbo["selection_counts"] = {
        score_names[int(key)]: value for key, value in score_pbo["selection_counts"].items()
    }
    production_score_name = next(
        name for name, params in score_params.items() if params == production_params
    )
    ordered_score_final = sorted((row["final_value"] for row in score_rows), reverse=True)
    production_score_final = float(production["metrics"]["final_value"])
    within_ten_percent = int(
        sum(row["final_value"] >= production_score_final * 0.90 for row in score_rows)
    )

    gate_stat_pass = bool(white["20"]["p_value"] < 0.05 and float(bootstrap["20"]["ci95"][0]) > 0.0)
    local_pass = float(np.mean(local_relative > 0.0)) >= 0.80
    historical_validation_passed = False
    if live_oos <= 1 and (
        not gate_stat_pass
        or not local_pass
        or float(score_pbo["pbo"]) >= 0.50
        or not historical_validation_passed
    ):
        classification = "high_overfit_risk"
    else:
        classification = "requires_live_oos_review"

    payload = {
        "meta": {
            "strategy": "production V3-G versus strict-drop V3",
            "data_start": str(dates[0])[:10],
            "data_end": data_end,
            "observations": len(dates),
            "v3g_launch_date": V3G_LAUNCH_DATE,
            "live_oos_observations": live_oos,
            "recorded_variant_evaluations_lower_bound": RECORDED_VARIANT_EVALUATIONS,
            "recorded_trials_are_correlated": True,
            "current_white_trial_universe": len(ret60_names),
            "bootstrap_samples": args.bootstrap_samples,
            "baseline_definition": (
                "same 10/20 momentum, risk and execution; only restore strict drop exclusion"
            ),
            "canonical_parity": True,
            "execution_model_warning": (
                "historical official close is a same-price 14:50 execution proxy"
            ),
        },
        "strict_v3": _compact(strict),
        "v3g": _compact(production),
        "gate_increment": {
            "terminal_relative_to_strict_v3": (production_score_final / strict_final - 1.0),
            "decision_divergences": _decision_divergences(
                strict["decision_log"],
                production["decision_log"],
                rebalance_dates,
            ),
            "annual": annual,
            "rolling_252d": rolling,
            "newey_west": hac,
            "moving_block_bootstrap": bootstrap,
            "white_reality_check_ret60_scan": white,
            "ret60_cscv": gate_pbo,
        },
        "gate_parameter_robustness": {
            "neighbors": len(local_rows),
            "neighbors_beating_strict_v3": int(np.sum(local_relative > 0.0)),
            "beat_rate": float(np.mean(local_relative > 0.0)),
            "minimum_relative": float(np.min(local_relative)),
            "median_relative": float(np.median(local_relative)),
            "maximum_relative": float(np.max(local_relative)),
            "rows": local_rows,
        },
        "momentum_selection": {
            "candidate_count": len(score_rows),
            "production_name": production_score_name,
            "production_rank_by_final": (ordered_score_final.index(production_score_final) + 1),
            "candidates_within_10pct_of_production": within_ten_percent,
            "cscv": score_pbo,
            "rows": score_rows,
        },
        "historical_governance_evidence": {
            "gate_final_validate_passed": False,
            "permutation_percentile": 91.66666666666666,
            "strict_rolling_checks_passed": "0/4",
            "expanded_peer_pool": (
                "historical gate gain did not reproduce consistently; recorded as 0%/-13.2%"
            ),
            "scope_warning": (
                "historical validation used earlier H3/downsize variants and is supporting "
                "governance evidence, not the current-path statistic"
            ),
        },
        "assessment": {
            "classification": classification,
            "gate_statistical_increment_pass": gate_stat_pass,
            "gate_local_neighborhood_pass": local_pass,
            "momentum_pbo_warning": float(score_pbo["pbo"]) >= 0.50,
            "interpretation": (
                "V3-G combines a sparsely triggered gate with a momentum core selected "
                "on the same history; untouched post-launch evidence is effectively absent."
            ),
        },
    }

    print("\nV3-G overfit audit")
    print(
        f"data={payload['meta']['data_start']}..{data_end} n={len(dates)} "
        f"live_OOS={live_oos} recorded_trials>={RECORDED_VARIANT_EVALUATIONS}"
    )
    print(
        f"strict V3={strict_final:,.0f} V3-G={production_score_final:,.0f} "
        f"gate relative={payload['gate_increment']['terminal_relative_to_strict_v3']:+.1%} "
        f"divergences={payload['gate_increment']['decision_divergences']}"
    )
    print(
        f"White p20={white['20']['p_value']:.3f} "
        f"bootstrap95={bootstrap['20']['ci95'][0]:+.1%}.."
        f"{bootstrap['20']['ci95'][1]:+.1%} gate_PBO={gate_pbo['pbo']:.1%}"
    )
    print(
        f"gate neighbors={int(np.sum(local_relative > 0.0))}/{len(local_rows)} "
        f"momentum rank={payload['momentum_selection']['production_rank_by_final']}/"
        f"{len(score_rows)} momentum_PBO={score_pbo['pbo']:.1%}"
    )
    print(
        f"positive years={annual['positive_relative_years']}/{annual['year_count']} "
        f"post2024 share={annual['post_2024_share_of_total_relative_log_return']:.1%} "
        f"assessment={classification}"
    )
    if args.save:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
