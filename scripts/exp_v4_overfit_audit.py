"""Mathematical overfitting audit for the production V4 overlay.

This research file does not alter V4.  It treats V3-G as the benchmark and
audits only V4's incremental evidence using the candidate set that was already
compared during V4 research, local parameter perturbations, contiguous-time
cross-validation, and dependence-aware bootstrap inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
except ModuleNotFoundError:
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_overfit_audit.json"
INITIAL_CAPITAL = 100_000.0
V4_LAUNCH_DATE = "2026-08-11"
FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]


def circular_block_indices(
    length: int,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> IntArray:
    """Draw one circular moving-block bootstrap sample."""
    if length < 1:
        raise ValueError("length must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    blocks = math.ceil(length / block_size)
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_size)
    return cast("IntArray", np.asarray(
        np.concatenate([(start + offsets) % length for start in starts])[:length],
        dtype=np.int64,
    ))


def newey_west_mean_test(values: np.ndarray, *, max_lag: int) -> dict[str, float]:
    """One-sided HAC test that the serially dependent mean is positive."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) < 2:
        raise ValueError("values must be a one-dimensional sample")
    if max_lag < 0:
        raise ValueError("max_lag cannot be negative")
    n = len(sample)
    mean = float(np.mean(sample))
    centered = sample - mean
    long_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_variance, 0.0) / n)
    t_stat = (
        math.inf if mean > 0.0 else 0.0
    ) if standard_error == 0.0 else mean / standard_error
    p_value = 0.5 * math.erfc(t_stat / math.sqrt(2.0))
    return {
        "observations": float(n),
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "p_value_one_sided": p_value,
        "max_lag": float(max_lag),
    }


def white_reality_check(
    excess_returns: np.ndarray,
    *,
    selected_column: int,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """White-style max-mean test with cross-candidate block resampling."""
    matrix = np.asarray(excess_returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("excess_returns must be a non-empty 2D matrix")
    if not 0 <= selected_column < matrix.shape[1]:
        raise ValueError("selected_column out of range")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    n, candidates = matrix.shape
    means = np.mean(matrix, axis=0)
    observed = math.sqrt(n) * float(np.max(means))
    centered = matrix - means
    rng = np.random.default_rng(seed)
    boot_max: FloatArray = np.empty(bootstrap_samples, dtype=float)
    for sample_index in range(bootstrap_samples):
        indices = circular_block_indices(n, block_size=block_size, rng=rng)
        boot_max[sample_index] = math.sqrt(n) * float(
            np.max(np.mean(centered[indices], axis=0))
        )
    p_value = float((1 + np.sum(boot_max >= observed)) / (bootstrap_samples + 1))
    return {
        "observations": n,
        "candidate_count": candidates,
        "block_size": block_size,
        "bootstrap_samples": bootstrap_samples,
        "selected_mean": float(means[selected_column]),
        "best_mean": float(np.max(means)),
        "selected_rank": int(
            candidates - np.argsort(np.argsort(means))[selected_column]
        ),
        "observed_max_stat": observed,
        "bootstrap_p95": float(np.quantile(boot_max, 0.95)),
        "p_value": p_value,
    }


def _mean_score(values: np.ndarray) -> FloatArray:
    return cast("FloatArray", np.asarray(np.mean(values, axis=0), dtype=np.float64))


def _rank_percentile(values: np.ndarray, selected: int) -> float:
    selected_value = values[selected]
    lower = float(np.sum(values < selected_value))
    equal = float(np.sum(values == selected_value))
    average_rank = lower + (equal - 1.0) / 2.0
    return average_rank / max(len(values) - 1, 1)


def cscv_probability_of_backtest_overfitting(
    return_matrix: np.ndarray,
    *,
    slices: int = 8,
) -> dict[str, Any]:
    """Estimate PBO with combinatorially symmetric contiguous time folds."""
    matrix = np.asarray(return_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("return_matrix needs at least two strategies")
    if slices < 4 or slices % 2:
        raise ValueError("slices must be an even integer of at least four")
    blocks: list[IntArray] = [np.asarray(block, dtype=np.int64) for block in np.array_split(
        np.arange(matrix.shape[0]), slices
    )]
    percentiles: list[float] = []
    selected_columns: list[int] = []
    for train_blocks in itertools.combinations(range(slices), slices // 2):
        train_set = set(train_blocks)
        train_index = np.concatenate([blocks[index] for index in train_blocks])
        test_index = np.concatenate([
            blocks[index] for index in range(slices) if index not in train_set
        ])
        train_scores = _mean_score(matrix[train_index])
        selected = int(np.argmax(train_scores))
        test_scores = _mean_score(matrix[test_index])
        percentiles.append(_rank_percentile(test_scores, selected))
        selected_columns.append(selected)
    values: FloatArray = np.asarray(percentiles, dtype=float)
    return {
        "slices": slices,
        "splits": len(values),
        "pbo": float(np.mean(values < 0.5)),
        "median_oos_rank_percentile": float(np.median(values)),
        "mean_oos_rank_percentile": float(np.mean(values)),
        "selection_counts": dict(Counter(selected_columns)),
    }


def local_gap_grid() -> dict[str, v4.FullPoolParams]:
    """Predefined ±20% three-dimensional neighborhood around V4 gaps."""
    grid: dict[str, v4.FullPoolParams] = {}
    for slow_scale, fast5_scale, fast3_scale in itertools.product(
        (0.8, 1.0, 1.2), repeat=3
    ):
        name = (
            f"slow{slow_scale:.1f}_fast5{fast5_scale:.1f}_fast3{fast3_scale:.1f}"
        )
        grid[name] = replace(
            v4.V4_PARAMS,
            slow_gap=v4.V4_PARAMS.slow_gap * slow_scale,
            fast_5d_gap=v4.V4_PARAMS.fast_5d_gap * fast5_scale,
            fast_3d_gap=v4.V4_PARAMS.fast_3d_gap * fast3_scale,
        )
    return grid


def historical_candidate_params() -> dict[str, v4.FullPoolParams]:
    """Deduplicate the minimum candidate universe recorded in V4 research."""
    candidates: list[tuple[str, v4.FullPoolParams]] = [
        ("slow", v4.FullPoolParams(mode="slow")),
        ("slow_confirm2", v4.FullPoolParams(mode="slow", confirmation_hits=2)),
        ("slow_gap1_confirm2", v4.FullPoolParams(
            mode="slow", slow_gap=0.01, confirmation_hits=2
        )),
        ("fast", v4.FullPoolParams(mode="fast")),
        ("fast_confirm2", v4.FullPoolParams(mode="fast", confirmation_hits=2)),
        ("consensus", v4.FullPoolParams(mode="consensus")),
        ("consensus_confirm2", v4.FullPoolParams(
            mode="consensus", confirmation_hits=2
        )),
        ("consensus_strict", v4.V4_PARAMS),
        ("consensus_very_strict", v4.FullPoolParams(
            mode="consensus",
            slow_gap=0.01,
            fast_5d_gap=0.03,
            fast_3d_gap=0.015,
            confirmation_hits=2,
        )),
        ("or", v4.FullPoolParams(mode="or")),
    ]
    for gap in (0.005, 0.01, 0.02, 0.03, 0.05):
        for hits in (1, 2):
            candidates.append((
                f"slow_gap{gap:.1%}_hits{hits}",
                v4.FullPoolParams(mode="slow", slow_gap=gap, confirmation_hits=hits),
            ))
    unique: dict[v4.FullPoolParams, str] = {}
    for name, params in candidates:
        unique.setdefault(params, name)
    return {name: params for params, name in unique.items()}


def _log_returns(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    curve = result["equity_curve"]
    equity = np.asarray(curve["equity"], dtype=float)
    if np.any(equity <= 0.0):
        raise RuntimeError("non-positive equity cannot be log-transformed")
    values = np.concatenate(([INITIAL_CAPITAL], equity))
    dates = np.asarray(curve["trade_date"])
    return dates, np.diff(np.log(values))


def _bootstrap_interval(
    values: np.ndarray,
    *,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    means: FloatArray = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        indices = circular_block_indices(len(sample), block_size=block_size, rng=rng)
        means[index] = float(np.mean(sample[indices]))
    transformed = np.expm1(means * 252.0)
    return {
        "block_size": block_size,
        "bootstrap_samples": bootstrap_samples,
        "annualized_relative_return": float(np.expm1(np.mean(sample) * 252.0)),
        "ci90": [float(value) for value in np.quantile(transformed, (0.05, 0.95))],
        "ci95": [float(value) for value in np.quantile(transformed, (0.025, 0.975))],
        "probability_positive": float(np.mean(transformed > 0.0)),
    }


def _annual_attribution(
    dates: np.ndarray,
    baseline_returns: np.ndarray,
    selected_returns: np.ndarray,
) -> dict[str, Any]:
    years = np.asarray([int(str(date)[:4]) for date in dates])
    rows: list[dict[str, Any]] = []
    excess = selected_returns - baseline_returns
    for year in sorted(set(years.tolist())):
        mask = years == year
        rows.append({
            "year": year,
            "observations": int(np.sum(mask)),
            "v3g_return": float(np.expm1(np.sum(baseline_returns[mask]))),
            "v4_return": float(np.expm1(np.sum(selected_returns[mask]))),
            "v4_relative_to_v3g": float(np.expm1(np.sum(excess[mask]))),
            "relative_log_contribution": float(np.sum(excess[mask])),
        })
    total_log = float(np.sum(excess))
    post_2024_log = float(np.sum(excess[years >= 2024]))
    leave_one_year_out = {
        str(row["year"]): float(np.expm1(total_log - row["relative_log_contribution"]))
        for row in rows
    }
    return {
        "years": rows,
        "positive_relative_years": int(sum(
            row["v4_relative_to_v3g"] > 0.0 for row in rows
        )),
        "year_count": len(rows),
        "post_2024_share_of_total_relative_log_return": (
            post_2024_log / total_log if total_log != 0.0 else 0.0
        ),
        "leave_one_year_out_relative_return": leave_one_year_out,
    }


def _rolling_attribution(
    dates: np.ndarray,
    excess_returns: np.ndarray,
    *,
    window: int = 252,
    step: int = 63,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(excess_returns) - window + 1, step):
        end = start + window
        rows.append({
            "start": str(dates[start])[:10],
            "end": str(dates[end - 1])[:10],
            "relative_return": float(np.expm1(np.sum(excess_returns[start:end]))),
        })
    values = np.asarray([row["relative_return"] for row in rows], dtype=float)
    return {
        "window_trading_days": window,
        "step_trading_days": step,
        "windows": len(rows),
        "positive_windows": int(np.sum(values > 0.0)),
        "positive_rate": float(np.mean(values > 0.0)) if len(values) else 0.0,
        "median_relative_return": float(np.median(values)) if len(values) else 0.0,
        "minimum_relative_return": float(np.min(values)) if len(values) else 0.0,
        "maximum_relative_return": float(np.max(values)) if len(values) else 0.0,
        "rows": rows,
    }


def _binomial_upper_tail(wins: int, trials: int) -> float:
    if trials < 1:
        return 1.0
    return float(sum(
        math.comb(trials, value) * 0.5**trials
        for value in range(wins, trials + 1)
    ))


def _event_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon in (5, 10, 20):
        key = f"ex_post_actual_relative_{horizon}d"
        values: FloatArray = np.asarray([
            float(event[key]) for event in events if key in event
        ], dtype=float)
        positive = np.sort(values[values > 0.0])[::-1]
        wins = int(np.sum(values > 0.0))
        horizons[str(horizon)] = {
            "observations": len(values),
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "win_rate": float(np.mean(values > 0.0)) if len(values) else 0.0,
            "sign_test_p_value_one_sided": _binomial_upper_tail(wins, len(values)),
            "top5_share_of_positive_sum": (
                float(np.sum(positive[:5]) / np.sum(positive))
                if len(positive) and np.sum(positive) > 0.0
                else 0.0
            ),
        }
    return {
        "events": len(events),
        "events_by_year": dict(Counter(str(event["date"])[:4] for event in events)),
        "horizons": horizons,
        "dependence_warning": (
            "event horizons overlap and sign-test observations are not independent"
        ),
    }


def _compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 mathematical overfit audit")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V4 audit requires all downsize layers disabled")
    if args.bootstrap_samples < 500:
        raise ValueError("use at least 500 bootstrap samples")

    data = rq.load_data()
    cache: dict[v4.FullPoolParams, dict[str, Any]] = {}

    def evaluate(params: v4.FullPoolParams) -> dict[str, Any]:
        if params not in cache:
            cache[params] = run_full_pool_strategy(data, params, cost_multiplier=1.0)
        return cache[params]

    baseline_params = v4.FullPoolParams.disabled()
    baseline = evaluate(baseline_params)
    selected = evaluate(v4.V4_PARAMS)
    dates, baseline_log_returns = _log_returns(baseline)
    selected_dates, selected_log_returns = _log_returns(selected)
    if dates.tolist() != selected_dates.tolist():
        raise RuntimeError("V3-G and V4 date alignment drift")
    data_end = str(dates[-1])[:10]
    live_oos_observations = int(np.sum(np.asarray([
        str(date)[:10] > V4_LAUNCH_DATE for date in dates
    ])))

    historical_params = historical_candidate_params()
    historical_results = {
        name: evaluate(params) for name, params in historical_params.items()
    }
    historical_names = list(historical_results)
    candidate_excess: list[np.ndarray] = []
    candidate_absolute: list[np.ndarray] = [baseline_log_returns]
    candidate_absolute_names = ["baseline"]
    for name in historical_names:
        candidate_dates, candidate_returns = _log_returns(historical_results[name])
        if candidate_dates.tolist() != dates.tolist():
            raise RuntimeError(f"candidate date drift: {name}")
        candidate_excess.append(candidate_returns - baseline_log_returns)
        candidate_absolute.append(candidate_returns)
        candidate_absolute_names.append(name)
    excess_matrix = np.column_stack(candidate_excess)
    absolute_matrix = np.column_stack(candidate_absolute)
    selected_column = historical_names.index("consensus_strict")

    white_checks = {
        str(block): white_reality_check(
            excess_matrix,
            selected_column=selected_column,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260811 + block,
        )
        for block in (5, 20, 60)
    }
    consensus_excess_names = [
        "consensus",
        "consensus_confirm2",
        "consensus_strict",
        "consensus_very_strict",
    ]
    consensus_excess_columns = [
        historical_names.index(name) for name in consensus_excess_names
    ]
    consensus_excess_matrix = excess_matrix[:, consensus_excess_columns]
    consensus_selected_column = consensus_excess_names.index("consensus_strict")
    consensus_white_checks = {
        str(block): white_reality_check(
            consensus_excess_matrix,
            selected_column=consensus_selected_column,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260821 + block,
        )
        for block in (5, 20, 60)
    }
    selected_excess = selected_log_returns - baseline_log_returns
    bootstrap_intervals = {
        str(block): _bootstrap_interval(
            selected_excess,
            block_size=block,
            bootstrap_samples=args.bootstrap_samples,
            seed=20260812 + block,
        )
        for block in (5, 20, 60)
    }
    newey_west = {
        str(lag): newey_west_mean_test(selected_excess, max_lag=lag)
        for lag in (5, 20, 60)
    }

    all_pbo = cscv_probability_of_backtest_overfitting(absolute_matrix, slices=8)
    all_pbo["selection_counts"] = {
        candidate_absolute_names[int(key)]: value
        for key, value in all_pbo["selection_counts"].items()
    }
    consensus_names = [
        "baseline",
        "consensus",
        "consensus_confirm2",
        "consensus_strict",
        "consensus_very_strict",
    ]
    consensus_columns = [candidate_absolute_names.index(name) for name in consensus_names]
    consensus_pbo = cscv_probability_of_backtest_overfitting(
        absolute_matrix[:, consensus_columns], slices=8
    )
    consensus_pbo["selection_counts"] = {
        consensus_names[int(key)]: value
        for key, value in consensus_pbo["selection_counts"].items()
    }

    grid_results = {
        name: evaluate(params) for name, params in local_gap_grid().items()
    }
    baseline_final = float(baseline["metrics"]["final_value"])
    surface_rows = []
    for name, result in grid_results.items():
        metrics = result["metrics"]
        surface_rows.append({
            "name": name,
            "params": result["params"],
            "final_value": float(metrics["final_value"]),
            "relative_to_v3g": float(metrics["final_value"] / baseline_final - 1.0),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "early_rotations": int(metrics["early_rotations"]),
        })
    center_name = "slow1.0_fast51.0_fast31.0"
    center_final = next(
        row["final_value"] for row in surface_rows if row["name"] == center_name
    )
    ordered_final = sorted(
        (row["final_value"] for row in surface_rows), reverse=True
    )
    relative_values = np.asarray([
        row["relative_to_v3g"] for row in surface_rows
    ], dtype=float)
    neighborhood_beat_rate = float(np.mean(relative_values > 0.0))
    surface_summary = {
        "definition": "3x3x3 independent gap perturbation at 80%/100%/120%",
        "points": len(surface_rows),
        "points_beating_v3g_final": int(np.sum(relative_values > 0.0)),
        "beat_rate": neighborhood_beat_rate,
        "median_relative_to_v3g": float(np.median(relative_values)),
        "minimum_relative_to_v3g": float(np.min(relative_values)),
        "maximum_relative_to_v3g": float(np.max(relative_values)),
        "v4_center_rank_by_final": ordered_final.index(center_final) + 1,
        "rows": surface_rows,
    }

    structural_params = {
        "hold_2": replace(v4.V4_PARAMS, minimum_hold_days=2),
        "hold_3": v4.V4_PARAMS,
        "hold_4": replace(v4.V4_PARAMS, minimum_hold_days=4),
        "confirmation_1": replace(v4.V4_PARAMS, confirmation_hits=1),
        "confirmation_2": v4.V4_PARAMS,
    }
    structural = {
        name: _compact_metrics(evaluate(params))
        for name, params in structural_params.items()
    }

    annual = _annual_attribution(
        dates,
        baseline_log_returns,
        selected_log_returns,
    )
    rolling = _rolling_attribution(dates, selected_excess)
    event_evidence = _event_evidence(selected["rotation_events"])

    statistical_pass = bool(
        white_checks["20"]["p_value"] < 0.05
        and float(bootstrap_intervals["20"]["ci95"][0]) > 0.0
    )
    neighborhood_pass = neighborhood_beat_rate >= 0.80
    pbo_warning = bool(
        all_pbo["pbo"] >= 0.50 or consensus_pbo["pbo"] >= 0.50
    )
    if live_oos_observations == 0 and (
        not statistical_pass or not neighborhood_pass or pbo_warning
    ):
        assessment = "high_overfit_risk"
    elif live_oos_observations == 0:
        assessment = "medium_high_overfit_risk"
    else:
        assessment = "requires_live_oos_review"

    payload = {
        "meta": {
            "strategy": "production V4 incremental to canonical V3-G",
            "data_start": str(dates[0])[:10],
            "data_end": data_end,
            "observations": len(dates),
            "v4_launch_date": V4_LAUNCH_DATE,
            "live_oos_observations": live_oos_observations,
            "historical_oos_label_valid": False,
            "historical_oos_reason": (
                "2024-2026 data participated in V4 design and selection"
            ),
            "minimum_recorded_nonbaseline_trials": len(historical_names),
            "trial_count_is_lower_bound": True,
            "bootstrap_samples": args.bootstrap_samples,
            "v4_params": asdict(v4.V4_PARAMS),
            "no_lookahead_in_signal_code": (
                "factor arrays are sliced through T; audit uses only realized return paths"
            ),
            "execution_model_warning": (
                "historical official close is used as a 14:50 same-price execution proxy; "
                "true intraday snapshots are unavailable"
            ),
        },
        "baseline": _compact_metrics(baseline),
        "v4": _compact_metrics(selected),
        "incremental": {
            "terminal_relative_to_v3g": (
                float(selected["metrics"]["final_value"]) / baseline_final - 1.0
            ),
            "annual": annual,
            "rolling_252d": rolling,
            "newey_west": newey_west,
            "moving_block_bootstrap": bootstrap_intervals,
            "white_reality_check": white_checks,
            "white_reality_check_consensus_family": consensus_white_checks,
        },
        "selection_bias": {
            "candidate_names": historical_names,
            "all_recorded_candidates_cscv": all_pbo,
            "consensus_family_cscv": consensus_pbo,
        },
        "parameter_robustness": {
            "gap_surface": surface_summary,
            "structural_neighbors": structural,
        },
        "rotation_event_evidence": event_evidence,
        "assessment": {
            "classification": assessment,
            "statistical_increment_pass": statistical_pass,
            "parameter_neighborhood_pass": neighborhood_pass,
            "pbo_warning": pbo_warning,
            "interpretation": (
                "The audit can identify overfit risk but cannot validate V4 without "
                "untouched post-launch observations."
            ),
        },
    }

    print("\nV4 overfit audit")
    print(
        f"data={payload['meta']['data_start']}..{data_end} "
        f"n={len(dates)} live_OOS={live_oos_observations}"
    )
    print(
        f"V3-G={baseline_final:,.0f} V4={selected['metrics']['final_value']:,.0f} "
        f"relative={payload['incremental']['terminal_relative_to_v3g']:+.1%}"
    )
    print(
        f"historical nonbaseline trials >= {len(historical_names)}; "
        f"White RC p(block20)={white_checks['20']['p_value']:.3f}; "
        f"consensus-only={consensus_white_checks['20']['p_value']:.3f}; "
        f"bootstrap95={bootstrap_intervals['20']['ci95'][0]:+.1%}.."
        f"{bootstrap_intervals['20']['ci95'][1]:+.1%}"
    )
    print(
        f"PBO all={all_pbo['pbo']:.1%} consensus={consensus_pbo['pbo']:.1%}; "
        f"local beat={surface_summary['beat_rate']:.1%}; "
        f"center rank={surface_summary['v4_center_rank_by_final']}/27"
    )
    print(
        f"positive years={annual['positive_relative_years']}/{annual['year_count']}; "
        f"post-2024 log contribution={annual['post_2024_share_of_total_relative_log_return']:.1%}; "
        f"252d rolling wins={rolling['positive_windows']}/{rolling['windows']}"
    )
    print(
        f"early rotations={event_evidence['events']}; "
        f"assessment={assessment}"
    )

    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
