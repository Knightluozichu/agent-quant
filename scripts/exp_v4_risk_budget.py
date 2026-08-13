"""Theory-first drawdown-budget research built on the frozen V4 signal path.

V4 continues to choose the risky ETF sleeve.  This file studies an independent
exposure controller: volatility targeting, a CPPI-style rolling drawdown floor,
an asset-shock brake, and one pre-registered combination.  It is research only
and deliberately does not modify the production selector, risk layer, or V4.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from scripts.exp_v4_overfit_audit import circular_block_indices
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v4_overfit_audit import circular_block_indices


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_risk_budget_research.json"
INITIAL_CAPITAL = 100_000.0
Mode = Literal[
    "volatility", "cppi", "shock", "risk_budget", "position_budget",
    "episode_budget", "constant", "constant_shock",
    "volatility_shock",
]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class RiskBudgetParams:
    """Pre-registered exposure rules; none of these parameters is scanned."""

    mode: Mode = "risk_budget"
    volatility_target: float = 0.30
    ewma_decay: float = 0.94
    floor_ratio: float = 0.80
    cppi_multiplier: float = 5.0
    shock_1d: float = -0.05
    shock_3d: float = -0.10
    post_shock_exposure: float = 0.50
    position_deductible: float = 0.05
    position_loss_limit: float = 0.10
    episode_trigger: float = 0.10
    episode_recovery: float = 0.05
    episode_severe: float = 0.20
    episode_exposure: float = 0.50
    episode_severe_exposure: float = 0.25
    constant_exposure: float = 0.80


def preregistered_candidates() -> dict[str, RiskBudgetParams]:
    """Return the fixed mechanism attribution set in declared order."""
    return {
        "vol30": RiskBudgetParams(mode="volatility"),
        "cppi20": RiskBudgetParams(mode="cppi"),
        "shock_brake": RiskBudgetParams(mode="shock"),
        "v4_risk_budget": RiskBudgetParams(mode="risk_budget"),
    }


def phase2_position_budget_candidate() -> RiskBudgetParams:
    """One theory revision after phase 1 exposed permanent CPPI cash lock."""
    return RiskBudgetParams(mode="position_budget")


def phase3_episode_budget_candidate() -> RiskBudgetParams:
    """Cross-holding accident state with a minimum participation rate."""
    return RiskBudgetParams(mode="episode_budget")


def phase4_constant_budget_candidates() -> dict[str, RiskBudgetParams]:
    """First-order homogeneous risk scaling, with one shock attribution variant."""
    return {
        "v4_constant80": RiskBudgetParams(mode="constant"),
        "v4_constant80_shock": RiskBudgetParams(mode="constant_shock"),
    }


def phase5_volatility_shock_candidate() -> RiskBudgetParams:
    """Compose the two orthogonal phase-1 controls without adding parameters."""
    return RiskBudgetParams(mode="volatility_shock")


def volatility_shock_neighbors() -> dict[str, RiskBudgetParams]:
    """Single-axis robustness perturbations around the frozen phase-5 center."""
    center = phase5_volatility_shock_candidate()
    return {
        "center": center,
        "volatility_80pct": replace(center, volatility_target=0.24),
        "volatility_120pct": replace(center, volatility_target=0.36),
        "decay_0.90": replace(center, ewma_decay=0.90),
        "decay_0.97": replace(center, ewma_decay=0.97),
        "shock_80pct": replace(center, shock_1d=-0.04, shock_3d=-0.08),
        "shock_120pct": replace(center, shock_1d=-0.06, shock_3d=-0.12),
    }


def _validate_params(params: RiskBudgetParams) -> None:
    if not 0.0 < params.volatility_target < 1.0:
        raise ValueError("volatility_target must be in (0, 1)")
    if not 0.0 < params.ewma_decay < 1.0:
        raise ValueError("ewma_decay must be in (0, 1)")
    if not 0.0 < params.floor_ratio < 1.0:
        raise ValueError("floor_ratio must be in (0, 1)")
    if params.cppi_multiplier <= 0.0:
        raise ValueError("cppi_multiplier must be positive")
    if not 0.0 <= params.post_shock_exposure <= 1.0:
        raise ValueError("post_shock_exposure must be in [0, 1]")
    if not 0.0 < params.position_deductible < params.position_loss_limit < 1.0:
        raise ValueError("position loss levels must satisfy 0 < deductible < limit < 1")
    if not 0.0 < params.episode_recovery < params.episode_trigger:
        raise ValueError("episode recovery must be below its trigger")
    if not params.episode_trigger < params.episode_severe < 1.0:
        raise ValueError("episode severe level must be above its trigger")
    if not 0.0 <= params.episode_severe_exposure <= params.episode_exposure <= 1.0:
        raise ValueError("episode exposures must be ordered inside [0, 1]")
    if not 0.0 < params.constant_exposure <= 1.0:
        raise ValueError("constant exposure must be in (0, 1]")


def cppi_exposure(
    wealth: float,
    peak: float,
    params: RiskBudgetParams,
) -> float:
    """CPPI risky weight against a rolling peak floor, clipped to no leverage."""
    _validate_params(params)
    if wealth <= 0.0 or peak <= 0.0:
        return 0.0
    floor = params.floor_ratio * peak
    cushion = max(wealth - floor, 0.0)
    return float(np.clip(params.cppi_multiplier * cushion / wealth, 0.0, 1.0))


def shock_exposure(
    one_day: float,
    three_day: float,
    params: RiskBudgetParams,
) -> float:
    """Risk cap for the next holding period after a T-observable asset shock."""
    _validate_params(params)
    if three_day <= params.shock_3d:
        return 0.0
    if one_day <= params.shock_1d:
        return params.post_shock_exposure
    return 1.0


def position_budget_exposure(
    position_drawdown: float,
    params: RiskBudgetParams,
) -> float:
    """Deductible loss insurance: full risk to 5%, then linear to zero at 10%."""
    _validate_params(params)
    loss = max(-position_drawdown, 0.0)
    if loss <= params.position_deductible:
        return 1.0
    if loss >= params.position_loss_limit:
        return 0.0
    remaining = params.position_loss_limit - loss
    span = params.position_loss_limit - params.position_deductible
    return float(np.clip(remaining / span, 0.0, 1.0))


def position_target_exposure(
    *,
    position_drawdown: float,
    one_day: float,
    three_day: float,
    holding_changed: bool,
    params: RiskBudgetParams,
) -> float:
    """Reset the insurance budget only when frozen V4 actually changes holding."""
    if holding_changed:
        return 1.0
    return min(
        position_budget_exposure(position_drawdown, params),
        shock_exposure(one_day, three_day, params),
    )


def episode_target_exposure(
    *,
    portfolio_drawdown: float,
    episode_active: bool,
    shock_cap: float,
    params: RiskBudgetParams,
) -> tuple[float, bool]:
    """Maintain a cross-holding drawdown state with recovery hysteresis."""
    _validate_params(params)
    loss = max(-portfolio_drawdown, 0.0)
    if episode_active and loss <= params.episode_recovery:
        episode_active = False
    if loss >= params.episode_trigger:
        episode_active = True
    if loss >= params.episode_severe:
        exposure = params.episode_severe_exposure
    elif episode_active:
        exposure = params.episode_exposure
    else:
        exposure = 1.0
    return min(exposure, shock_cap), episode_active


def constant_target_exposure(
    *,
    one_day: float,
    three_day: float,
    params: RiskBudgetParams,
) -> float:
    """Constant V4 allocation, optionally capped by a T-observable shock."""
    _validate_params(params)
    if params.mode == "constant_shock":
        return min(
            params.constant_exposure,
            shock_exposure(one_day, three_day, params),
        )
    return params.constant_exposure


def volatility_exposure(
    annualized_volatility: float,
    params: RiskBudgetParams,
) -> float:
    """Unlevered inverse-volatility weight."""
    _validate_params(params)
    if annualized_volatility <= 0.0:
        return 1.0
    return float(np.clip(
        params.volatility_target / annualized_volatility, 0.0, 1.0
    ))


def target_exposure(
    *,
    wealth: float,
    peak: float,
    one_day: float,
    three_day: float,
    annualized_volatility: float,
    params: RiskBudgetParams,
) -> float:
    """Compute the next-period exposure from information observable at T."""
    if params.mode == "volatility":
        return volatility_exposure(annualized_volatility, params)
    if params.mode == "volatility_shock":
        return min(
            volatility_exposure(annualized_volatility, params),
            shock_exposure(one_day, three_day, params),
        )
    if params.mode == "cppi":
        return cppi_exposure(wealth, peak, params)
    if params.mode == "shock":
        return shock_exposure(one_day, three_day, params)
    if params.mode == "risk_budget":
        return min(
            cppi_exposure(wealth, peak, params),
            shock_exposure(one_day, three_day, params),
        )
    if params.mode in ("constant", "constant_shock"):
        return constant_target_exposure(
            one_day=one_day, three_day=three_day, params=params
        )
    if params.mode in ("position_budget", "episode_budget"):
        raise ValueError(f"{params.mode} requires stateful controller inputs")
    raise ValueError(f"unsupported mode: {params.mode}")


def replay_documented_shock() -> dict[str, Any]:
    """Replay -5%/-8%/-3%; today's signal changes only tomorrow's exposure."""
    returns = [-0.05, -0.08, -0.03]
    three_day_signals = [-0.05, (0.95 * 0.92 - 1.0), 0.95 * 0.92 * 0.97 - 1.0]
    params = RiskBudgetParams(mode="risk_budget")
    exposure = 1.0
    wealth = 1.0
    peak = 1.0
    used: list[float] = []
    for daily_return, three_day in zip(returns, three_day_signals, strict=True):
        used.append(exposure)
        wealth *= 1.0 + exposure * daily_return
        peak = max(peak, wealth)
        exposure = target_exposure(
            wealth=wealth,
            peak=peak,
            one_day=daily_return,
            three_day=three_day,
            annualized_volatility=0.20,
            params=params,
        )
    constant_params = RiskBudgetParams(mode="constant_shock")
    constant_exposure = constant_params.constant_exposure
    constant_wealth = 1.0
    constant_used: list[float] = []
    for daily_return, three_day in zip(returns, three_day_signals, strict=True):
        constant_used.append(constant_exposure)
        constant_wealth *= 1.0 + constant_exposure * daily_return
        constant_exposure = constant_target_exposure(
            one_day=daily_return,
            three_day=three_day,
            params=constant_params,
        )
    return {
        "holding_returns": returns,
        "exposure_used": used,
        "v4_holding_return": float(np.prod(1.0 + np.asarray(returns)) - 1.0),
        "risk_budget_return": wealth - 1.0,
        "position_budget_return": wealth - 1.0,
        "constant80_shock_exposure_used": constant_used,
        "constant80_shock_return": constant_wealth - 1.0,
        "first_gap_is_unavoidable": True,
        "execution_rule": (
            "T-day 14:50 signal controls only the exposure after that observation"
        ),
    }


def _curve_returns(curve: pd.DataFrame) -> FloatArray:
    equity = np.asarray(curve["equity"], dtype=float)
    return cast("FloatArray", np.asarray(
        np.diff(np.concatenate(([INITIAL_CAPITAL], equity))) / np.concatenate(
            ([INITIAL_CAPITAL], equity[:-1])
        ),
        dtype=np.float64,
    ))


def _date_index(frame: pd.DataFrame) -> dict[pd.Timestamp, int]:
    return {
        pd.Timestamp(value).normalize(): int(index)
        for index, value in enumerate(frame["trade_date"])
    }


def observable_inputs(
    data: dict[str, pd.DataFrame],
    curve: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all T-observable safe, shock, and position-risk inputs."""
    if rq.DEFENSE not in data:
        raise RuntimeError(f"defense series missing: {rq.DEFENSE}")
    maps = {code: _date_index(frame) for code, frame in data.items()}
    safe_close = np.asarray(data[rq.DEFENSE]["close"], dtype=float)
    safe: list[float] = []
    held_1d: list[float] = []
    held_3d: list[float] = []
    position_drawdowns: list[float] = []
    holding_changes: list[bool] = []
    holdings = curve["holding"].astype(str).tolist()
    position_peak = 0.0
    for position, raw_date in enumerate(curve["trade_date"]):
        td = pd.Timestamp(raw_date).normalize()
        safe_index = maps[rq.DEFENSE][td]
        safe.append(
            float(safe_close[safe_index] / safe_close[safe_index - 1] - 1.0)
            if safe_index > 0 else 0.0
        )
        if position == 0:
            held_1d.append(0.0)
            held_3d.append(0.0)
        else:
            held = holdings[position - 1]
            if held not in data or td not in maps[held]:
                held_1d.append(0.0)
                held_3d.append(0.0)
            else:
                index = maps[held][td]
                close = np.asarray(data[held]["close"], dtype=float)
                held_1d.append(
                    float(close[index] / close[index - 1] - 1.0)
                    if index >= 1 else 0.0
                )
                held_3d.append(
                    float(close[index] / close[index - 3] - 1.0)
                    if index >= 3 else 0.0
                )

        current_holding = holdings[position]
        changed = position == 0 or current_holding != holdings[position - 1]
        holding_changes.append(changed)
        if current_holding not in data or td not in maps[current_holding]:
            position_peak = 0.0
            position_drawdowns.append(0.0)
            continue
        current_index = maps[current_holding][td]
        current_price = float(data[current_holding].iloc[current_index]["close"])
        if changed or position_peak <= 0.0:
            position_peak = current_price
        else:
            position_peak = max(position_peak, current_price)
        position_drawdowns.append(
            current_price / position_peak - 1.0 if position_peak > 0.0 else 0.0
        )
    return (
        np.asarray(safe, dtype=float),
        np.asarray(held_1d, dtype=float),
        np.asarray(held_3d, dtype=float),
        np.asarray(position_drawdowns, dtype=float),
        np.asarray(holding_changes, dtype=bool),
    )


def _annual_rows(curve: pd.DataFrame) -> list[dict[str, Any]]:
    indexed = curve.copy()
    indexed["year"] = indexed["trade_date"].dt.year
    prior = INITIAL_CAPITAL
    rows: list[dict[str, Any]] = []
    for year, group in indexed.groupby("year", sort=True):
        end = float(group["equity"].iloc[-1])
        rows.append({
            "year": int(cast("int", year)),
            "return": end / prior - 1.0,
            "end_value": end,
        })
        prior = end
    return rows


def _rolling_252(curve: pd.DataFrame) -> dict[str, Any]:
    equity = np.concatenate((
        np.asarray([INITIAL_CAPITAL], dtype=float),
        np.asarray(curve["equity"], dtype=float),
    ))
    if len(equity) <= 252:
        return {"windows": 0, "negative_windows": 0}
    values = equity[252:] / equity[:-252] - 1.0
    return {
        "window_days": 252,
        "windows": len(values),
        "negative_windows": int(np.sum(values < 0.0)),
        "positive_rate": float(np.mean(values > 0.0)),
        "minimum_return": float(np.min(values)),
        "median_return": float(np.median(values)),
        "maximum_return": float(np.max(values)),
    }


def _risk_metrics(curve: pd.DataFrame) -> dict[str, Any]:
    metrics = dict(rr.curve_metrics(curve, initial_capital=INITIAL_CAPITAL))
    returns = _curve_returns(curve)
    equity = np.asarray(curve["equity"], dtype=float)
    cagr = float(metrics["cagr"])
    max_drawdown = float(metrics["max_drawdown"])
    drawdowns = equity / np.maximum.accumulate(
        np.concatenate(([INITIAL_CAPITAL], equity))
    )[1:] - 1.0
    tail_count = max(math.ceil(0.05 * len(returns)), 1)
    metrics.update({
        "calmar": float(
            cagr / abs(max_drawdown)
            if max_drawdown < 0.0 else math.inf
        ),
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdowns)))),
        "daily_expected_shortfall_5pct": float(
            np.mean(np.sort(returns)[:tail_count])
        ),
        "worst_1d_return": float(np.min(returns)),
    })
    return metrics


def run_overlay(
    baseline: dict[str, Any],
    safe_returns: np.ndarray,
    held_1d: np.ndarray,
    held_3d: np.ndarray,
    position_drawdowns: np.ndarray,
    holding_changes: np.ndarray,
    params: RiskBudgetParams,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Scale the frozen V4 sleeve and place unused capital in its defense ETF."""
    _validate_params(params)
    base_curve = baseline["equity_curve"]
    base_returns = _curve_returns(base_curve)
    if not (
        len(base_returns)
        == len(safe_returns)
        == len(held_1d)
        == len(held_3d)
        == len(position_drawdowns)
        == len(holding_changes)
    ):
        raise ValueError("overlay inputs are not aligned")

    wealth = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    exposure = (
        params.constant_exposure
        if params.mode in ("constant", "constant_shock")
        else 1.0
    )
    ewma_variance = (params.volatility_target**2) / 252.0
    controller_cost = 0.0
    turnover = 0.0
    episode_active = False
    episode_days = 0
    rows: list[dict[str, Any]] = []
    dates = list(base_curve["trade_date"])
    fee_per_reallocation = 2.0 * (rq.FEE + rq.SLIPPAGE) * cost_multiplier
    initial_safe_cost = (
        wealth
        * (1.0 - exposure)
        * (rq.FEE + rq.SLIPPAGE)
        * cost_multiplier
    )
    wealth -= initial_safe_cost
    controller_cost += initial_safe_cost
    turnover += 1.0 - exposure

    for index, td in enumerate(dates):
        used_exposure = exposure
        risky_value = wealth * used_exposure * (1.0 + base_returns[index])
        safe_value = wealth * (1.0 - used_exposure) * (1.0 + safe_returns[index])
        wealth = risky_value + safe_value
        drifted_exposure = risky_value / wealth if wealth > 0.0 else 0.0
        peak = max(peak, wealth)
        ewma_variance = (
            params.ewma_decay * ewma_variance
            + (1.0 - params.ewma_decay) * base_returns[index] ** 2
        )
        annualized_volatility = math.sqrt(max(ewma_variance, 0.0) * 252.0)
        portfolio_drawdown = wealth / peak - 1.0 if peak > 0.0 else 0.0
        if params.mode == "position_budget":
            next_exposure = position_target_exposure(
                position_drawdown=float(position_drawdowns[index]),
                one_day=float(held_1d[index]),
                three_day=float(held_3d[index]),
                holding_changed=bool(holding_changes[index]),
                params=params,
            )
        elif params.mode == "episode_budget":
            next_exposure, episode_active = episode_target_exposure(
                portfolio_drawdown=portfolio_drawdown,
                episode_active=episode_active,
                shock_cap=shock_exposure(
                    float(held_1d[index]), float(held_3d[index]), params
                ),
                params=params,
            )
        else:
            next_exposure = target_exposure(
                wealth=wealth,
                peak=peak,
                one_day=float(held_1d[index]),
                three_day=float(held_3d[index]),
                annualized_volatility=annualized_volatility,
                params=params,
            )
        reallocation = abs(next_exposure - drifted_exposure)
        cost = wealth * reallocation * fee_per_reallocation
        wealth -= cost
        controller_cost += cost
        turnover += reallocation
        episode_days += int(episode_active)
        rows.append({
            "trade_date": pd.Timestamp(td),
            "equity": wealth,
            "holding": base_curve.iloc[index]["holding"],
            "exposure_used": used_exposure,
            "next_exposure": next_exposure,
            "drifted_exposure": drifted_exposure,
            "annualized_volatility": annualized_volatility,
            "held_1d_return": float(held_1d[index]),
            "held_3d_return": float(held_3d[index]),
            "position_drawdown": float(position_drawdowns[index]),
            "holding_changed": bool(holding_changes[index]),
            "portfolio_drawdown": portfolio_drawdown,
            "episode_active": episode_active,
            "controller_cost": cost,
        })
        exposure = next_exposure

    curve = pd.DataFrame(rows)
    metrics = _risk_metrics(curve)
    annual = _annual_rows(curve)
    metrics.update({
        "cost_multiplier": cost_multiplier,
        "average_exposure": float(curve["exposure_used"].mean()),
        "minimum_exposure": float(curve["exposure_used"].min()),
        "days_below_full_exposure": int(np.sum(curve["exposure_used"] < 0.999)),
        "zero_exposure_days": int(np.sum(curve["exposure_used"] <= 0.001)),
        "controller_turnover": turnover,
        "controller_cost": controller_cost,
        "episode_days": episode_days,
        "negative_years": int(sum(row["return"] < 0.0 for row in annual)),
    })
    return {
        "params": asdict(params),
        "metrics": metrics,
        "annual": annual,
        "rolling_252": _rolling_252(curve),
        "equity_curve": curve,
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result.get("params", asdict(v4.V4_PARAMS)),
        "metrics": result["metrics"],
        "annual": result.get("annual", _annual_rows(result["equity_curve"])),
        "rolling_252": result.get(
            "rolling_252", _rolling_252(result["equity_curve"])
        ),
    }


def _recent_comparison(
    baseline: dict[str, Any],
    selected: dict[str, Any],
    *,
    days: int = 10,
) -> list[dict[str, Any]]:
    base_curve = baseline["equity_curve"]
    selected_curve = selected["equity_curve"]
    base_returns = _curve_returns(base_curve)
    selected_returns = _curve_returns(selected_curve)
    rows: list[dict[str, Any]] = []
    for index in range(max(len(base_curve) - days, 0), len(base_curve)):
        selected_row = selected_curve.iloc[index]
        rows.append({
            "date": str(pd.Timestamp(base_curve.iloc[index]["trade_date"]).date()),
            "holding": str(base_curve.iloc[index]["holding"]),
            "v4_return": float(base_returns[index]),
            "selected_return": float(selected_returns[index]),
            "exposure_used": float(selected_row["exposure_used"]),
            "next_exposure": float(selected_row["next_exposure"]),
            "held_1d_return": float(selected_row["held_1d_return"]),
            "held_3d_return": float(selected_row["held_3d_return"]),
            "annualized_volatility": float(
                selected_row["annualized_volatility"]
            ),
        })
    return rows


def _bootstrap_comparison(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    *,
    bootstrap_samples: int,
    block_size: int,
    seed: int,
) -> dict[str, Any]:
    """Paired block bootstrap of realized policy returns, without re-optimization."""
    base = _curve_returns(baseline["equity_curve"])
    candidate = _curve_returns(challenger["equity_curve"])
    if len(base) != len(candidate):
        raise ValueError("bootstrap paths are not aligned")
    rng = np.random.default_rng(seed)
    retention: list[float] = []
    drawdown_improvement: list[float] = []
    joint_pass = 0
    safety_pass = 0
    positive = 0
    for _ in range(bootstrap_samples):
        indices = circular_block_indices(
            len(base), block_size=block_size, rng=rng
        )
        base_sample = base[indices]
        candidate_sample = candidate[indices]
        base_wealth: FloatArray = np.cumprod(1.0 + base_sample)
        candidate_wealth: FloatArray = np.cumprod(1.0 + candidate_sample)
        base_cagr = float(base_wealth[-1] ** (252.0 / len(base)) - 1.0)
        candidate_cagr = float(
            candidate_wealth[-1] ** (252.0 / len(candidate)) - 1.0
        )
        base_peak = np.maximum.accumulate(np.concatenate(([1.0], base_wealth)))[1:]
        candidate_peak = np.maximum.accumulate(
            np.concatenate(([1.0], candidate_wealth))
        )[1:]
        base_mdd = abs(float(np.min(base_wealth / base_peak - 1.0)))
        candidate_mdd = abs(float(
            np.min(candidate_wealth / candidate_peak - 1.0)
        ))
        cagr_retention = (
            candidate_cagr / base_cagr if base_cagr > 0.0 else 0.0
        )
        mdd_gain = 1.0 - candidate_mdd / base_mdd if base_mdd > 0.0 else 0.0
        retention.append(cagr_retention)
        drawdown_improvement.append(mdd_gain)
        positive += int(candidate_wealth[-1] > 1.0)
        joint_pass += int(cagr_retention >= 0.80 and mdd_gain >= 0.20)
        safety_pass += int(candidate_cagr > 0.0 and candidate_mdd <= 0.20)
    return {
        "block_size": block_size,
        "bootstrap_samples": bootstrap_samples,
        "probability_positive_terminal": positive / bootstrap_samples,
        "probability_joint_80pct_cagr_20pct_mdd_rule": joint_pass / bootstrap_samples,
        "probability_positive_cagr_and_mdd_at_most_20pct": (
            safety_pass / bootstrap_samples
        ),
        "cagr_retention_ci95": np.asarray(
            np.quantile(retention, (0.025, 0.975)), dtype=float
        ).tolist(),
        "mdd_improvement_ci95": np.asarray(
            np.quantile(drawdown_improvement, (0.025, 0.975)), dtype=float
        ).tolist(),
        "scope_warning": (
            "resamples realized fixed-policy returns; it does not reselect parameters"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 risk-budget research")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 500:
        raise ValueError("use at least 500 bootstrap samples")
    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V4 risk-budget research expects disabled legacy downsizing")

    data = rq.load_data()
    baseline_by_cost = {
        multiplier: run_full_pool_strategy(
            data, v4.V4_PARAMS, cost_multiplier=multiplier
        )
        for multiplier in (0.0, 1.0, 2.0, 3.0)
    }
    phase1_candidates = preregistered_candidates()
    candidates = {
        **phase1_candidates,
        "v4_position_budget": phase2_position_budget_candidate(),
        "v4_episode_budget": phase3_episode_budget_candidate(),
        **phase4_constant_budget_candidates(),
        "v4_vol30_shock": phase5_volatility_shock_candidate(),
    }
    overlay_by_cost: dict[float, dict[str, dict[str, Any]]] = {}
    for cost, baseline in baseline_by_cost.items():
        inputs = observable_inputs(data, baseline["equity_curve"])
        overlay_by_cost[cost] = {
            name: run_overlay(
                baseline, *inputs, params, cost_multiplier=cost
            )
            for name, params in candidates.items()
        }

    baseline = baseline_by_cost[1.0]
    baseline["metrics"] = {
        **baseline["metrics"],
        **{
            key: value
            for key, value in _risk_metrics(baseline["equity_curve"]).items()
            if key not in baseline["metrics"]
        },
    }
    named = overlay_by_cost[1.0]
    base_metrics = baseline["metrics"]
    base_annual = _annual_rows(baseline["equity_curve"])
    base_negative_years = sum(row["return"] < 0.0 for row in base_annual)
    shock_scenario = replay_documented_shock()

    acceptance: dict[str, Any] = {}
    for name, result in named.items():
        metrics = result["metrics"]
        cagr_retention = float(metrics["cagr"] / base_metrics["cagr"])
        mdd_improvement = float(
            1.0 - abs(metrics["max_drawdown"]) / abs(base_metrics["max_drawdown"])
        )
        cost2 = overlay_by_cost[2.0][name]["metrics"]
        checks = {
            "cagr_retention_at_least_80pct": cagr_retention >= 0.80,
            "mdd_reduction_at_least_20pct": mdd_improvement >= 0.20,
            "no_additional_negative_calendar_year": (
                metrics["negative_years"] <= base_negative_years
            ),
            "positive_at_2x_cost": cost2["final_value"] > INITIAL_CAPITAL,
            "mdd_below_40pct_at_2x_cost": cost2["max_drawdown"] > -0.40,
            "documented_shock_below_10pct": (
                name not in (
                    "v4_risk_budget", "v4_position_budget", "v4_episode_budget",
                    "v4_constant80_shock",
                    "v4_vol30_shock",
                )
                or shock_scenario["risk_budget_return"] > -0.10
            ),
        }
        acceptance[name] = {
            "cagr_retention": cagr_retention,
            "mdd_improvement": mdd_improvement,
            "checks": checks,
            "passed": all(checks.values()),
        }

    segments = {
        label: {
            "baseline_v4": rr.segment_metrics(baseline["equity_curve"], start, end),
            **{
                name: rr.segment_metrics(result["equity_curve"], start, end)
                for name, result in named.items()
            },
        }
        for label, start, end in (
            ("retrospective_2020_2023", "2020-06-19", "2023-12-29"),
            ("retrospective_2024_current", "2024-01-01", "2026-08-11"),
        )
    }
    cost_pressure = {
        f"{cost:.0f}x": {
            "baseline_v4": _compact(baseline_result),
            **{
                name: _compact(result)
                for name, result in overlay_by_cost[cost].items()
            },
        }
        for cost, baseline_result in baseline_by_cost.items()
    }
    selected_name = "v4_vol30_shock"
    baseline_inputs = observable_inputs(data, baseline["equity_curve"])
    neighbor_results = {
        name: run_overlay(
            baseline,
            *baseline_inputs,
            params,
            cost_multiplier=1.0,
        )
        for name, params in volatility_shock_neighbors().items()
    }
    neighbor_rows: list[dict[str, Any]] = []
    for name, result in neighbor_results.items():
        metrics = result["metrics"]
        neighbor_rows.append({
            "name": name,
            "params": result["params"],
            "final_value": float(metrics["final_value"]),
            "cagr": float(metrics["cagr"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "sharpe": float(metrics["sharpe"]),
            "average_exposure": float(metrics["average_exposure"]),
            "negative_years": int(metrics["negative_years"]),
            "safety_profile_passed": bool(
                metrics["cagr"] > 0.0
                and metrics["max_drawdown"] >= -0.20
                and metrics["negative_years"] == 0
            ),
        })
    neighbor_passes = sum(row["safety_profile_passed"] for row in neighbor_rows)
    neighborhood = {
        "definition": "one-axis center plus +/-20% risk and shock perturbations",
        "points": len(neighbor_rows),
        "safety_profile_passes": neighbor_passes,
        "safety_profile_pass_rate": neighbor_passes / len(neighbor_rows),
        "minimum_cagr": min(row["cagr"] for row in neighbor_rows),
        "maximum_mdd_magnitude": max(
            abs(row["max_drawdown"]) for row in neighbor_rows
        ),
        "rows": neighbor_rows,
    }
    bootstrap = {
        str(block): _bootstrap_comparison(
            baseline,
            named[selected_name],
            bootstrap_samples=args.bootstrap_samples,
            block_size=block,
            seed=20261000 + block,
        )
        for block in (5, 20, 60)
    }

    selected_pass = bool(acceptance[selected_name]["passed"])
    selected_metrics = named[selected_name]["metrics"]
    selected_cost2 = overlay_by_cost[2.0][selected_name]["metrics"]
    safety_checks = {
        "positive_cagr": selected_metrics["cagr"] > 0.0,
        "mdd_at_most_20pct": selected_metrics["max_drawdown"] >= -0.20,
        "no_negative_calendar_year": selected_metrics["negative_years"] == 0,
        "positive_at_2x_cost": selected_cost2["final_value"] > INITIAL_CAPITAL,
        "mdd_below_40pct_at_2x_cost": selected_cost2["max_drawdown"] > -0.40,
        "documented_shock_below_10pct": (
            shock_scenario["risk_budget_return"] > -0.10
        ),
    }
    safety_pass = bool(all(safety_checks.values()))
    neighborhood_pass = bool(
        neighborhood["safety_profile_pass_rate"] >= 0.80
    )
    bootstrap_safety_probability = float(
        bootstrap["20"]["probability_positive_cagr_and_mdd_at_most_20pct"]
    )
    robustness_pass = bool(
        neighborhood_pass and bootstrap_safety_probability >= 0.80
    )
    classification = (
        "shadow_candidate_strict_constraints_require_oos"
        if selected_pass and robustness_pass
        else (
            "provisional_shadow_candidate_high_uncertainty"
            if safety_pass
            else "failed_preregistered_constraints"
        )
    )
    payload = {
        "meta": {
            "strategy": "frozen V4 signal path plus theory-first exposure budgeting",
            "data_start": str(baseline["equity_curve"]["trade_date"].iloc[0].date()),
            "data_end": str(baseline["equity_curve"]["trade_date"].iloc[-1].date()),
            "observations": len(baseline["equity_curve"]),
            "design_date": "2026-08-12",
            "live_oos_observations": 0,
            "no_parameter_scan": True,
            "candidate_count": len(candidates),
            "current_recorded_path_evaluations": (
                len(candidates) + len(neighbor_results)
            ),
            "adaptive_research_sequence": True,
            "phase1_candidate_count": len(phase1_candidates),
            "phase2_theory_revision_count": 1,
            "phase2_reason": (
                "phase 1 CPPI met drawdown control but lost return through persistent "
                "cash lock; phase 2 resets its loss budget only on a V4 holding change"
            ),
            "phase3_theory_revision_count": 1,
            "phase3_reason": (
                "the maximum drawdown crossed many V4 holdings, so phase 3 preserves "
                "one accident state across rotations and retains minimum participation"
            ),
            "phase4_theory_revision_count": 1,
            "phase4_reason": (
                "without a validated timing predictor, first-order risk and drawdown are "
                "approximately homogeneous in exposure; 80% is derived from the 20% "
                "MDD-reduction objective rather than selected from a return scan"
            ),
            "phase5_composition_count": 1,
            "phase5_reason": (
                "volatility control reduced historical MDD while shock control addressed "
                "the documented clustered-loss path; their minimum adds no parameter"
            ),
            "v4_params": asdict(v4.V4_PARAMS),
            "safe_sleeve": rq.DEFENSE,
            "controller_cost_model": (
                "two ETF legs per exposure change, in addition to scaled V4 sleeve costs"
            ),
            "no_lookahead": (
                "exposure used on day T was fixed at T-1; T observations only set the "
                "next period exposure"
            ),
            "execution_warning": (
                "official close is a same-price 14:50 proxy; gap losses before the "
                "observation cannot be prevented"
            ),
            "frozen_path_approximation_warning": (
                "the exposure controller scales the archived V4 sleeve return and does "
                "not feed its lower equity path back into risk_overrides; an integrated "
                "engine replay is required before any implementation review"
            ),
        },
        "theory": {
            "objective": (
                "retain at least 80% of V4 CAGR while reducing MDD magnitude at least "
                "20%, without adding negative years or failing 2x costs"
            ),
            "volatility_rule": "x=min(1, 30% / EWMA annualized volatility), lambda=0.94",
            "cppi_rule": (
                "floor=80% of running peak; x=min(1, 5*(wealth-floor)/wealth)"
            ),
            "shock_rule": (
                "next-period x<=50% after held asset 1d<=-5%; x=0 after 3d<=-10%"
            ),
            "position_budget_rule": (
                "full exposure through 5% position drawdown, linear to zero at 10%, "
                "then reset only after frozen V4 changes the holding"
            ),
            "episode_budget_rule": (
                "at 10% portfolio drawdown use 50% exposure until recovery to 5%; "
                "at 20% use 25%; combine with the stricter asset-shock cap"
            ),
            "constant_budget_rule": (
                "80% frozen V4 sleeve plus 20% defense, daily rebalanced; the shock "
                "variant applies the pre-registered 50%/0% next-period caps"
            ),
            "volatility_shock_rule": (
                "x=min(1, 30%/EWMA volatility, 1d/3d shock cap)"
            ),
            "guarantee_limit": (
                "daily discrete execution creates gap risk; the 80% floor is soft, not "
                "a guaranteed payoff"
            ),
            "primary_references": [
                "https://doi.org/10.3386/w22208",
                "https://doi.org/10.1016/0165-1889(92)90043-E",
            ],
        },
        "baseline_v4": _compact(baseline),
        "candidates": {name: _compact(result) for name, result in named.items()},
        "acceptance": acceptance,
        "cost_pressure": cost_pressure,
        "retrospective_segments": segments,
        "documented_shock_scenario": shock_scenario,
        "paired_block_bootstrap_selected": {
            "selected": selected_name,
            "results": bootstrap,
        },
        "parameter_robustness": neighborhood,
        "recent_10_trading_days": _recent_comparison(
            baseline, named[selected_name]
        ),
        "assessment": {
            "classification": classification,
            "preregistered_constraints_passed": selected_pass,
            "safety_profile_checks": safety_checks,
            "safety_profile_passed": safety_pass,
            "local_neighborhood_pass_rate": neighborhood[
                "safety_profile_pass_rate"
            ],
            "local_neighborhood_passed": neighborhood_pass,
            "bootstrap_20d_safety_probability": bootstrap_safety_probability,
            "robustness_passed": robustness_pass,
            "production_status": "research_only_no_production_change",
            "interpretation": (
                "A historical pass can nominate a frozen shadow path, but zero untouched "
                "post-design observations cannot validate the controller."
            ),
        },
    }

    print("\nV4 theory-first risk budget")
    print("variant             final    CAGR Sharpe     MDD Calmar avgExp negY pass")
    display = {"baseline_v4": baseline, **named}
    for name, result in display.items():
        metrics = result["metrics"]
        avg_exposure = metrics.get("average_exposure", 1.0)
        negative_years = metrics.get("negative_years", base_negative_years)
        passed = acceptance.get(name, {}).get("passed", "-")
        print(
            f"{name:<18} {metrics['final_value']:>9,.0f} "
            f"{metrics['cagr']:>6.1%} {metrics['sharpe']:>6.3f} "
            f"{metrics['max_drawdown']:>7.1%} {metrics['calmar']:>6.2f} "
            f"{avg_exposure:>6.1%} {negative_years:>4} {passed!s:>5}"
        )
    selected = acceptance[selected_name]
    print(
        f"\n{selected_name} retention={selected['cagr_retention']:.1%} "
        f"MDD improvement={selected['mdd_improvement']:.1%} "
        f"shock={shock_scenario['risk_budget_return']:+.1%} "
        f"assessment={classification}"
    )
    print(
        f"bootstrap20 joint pass probability="
        f"{bootstrap['20']['probability_joint_80pct_cagr_20pct_mdd_rule']:.1%}"
    )
    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
