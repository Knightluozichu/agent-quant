"""Theory-first V4 regime guard research.

The canonical V4 sleeve is frozen.  This module studies an orthogonal risk
state made from absolute trend, cross-asset breadth, leadership stability,
correlation concentration, downside-volatility acceleration, and observable
holding shocks.  A final sparse tail-confirmation revision carries shock risk
across a V4 rotation and reuses historical V3 tail-risk priors.  It is research
only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import exp_v4_momentum_failure as mf
    from scripts import exp_v4_risk_budget as rb
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from scripts.exp_v4_overfit_audit import circular_block_indices
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import exp_v4_momentum_failure as mf
    import exp_v4_risk_budget as rb
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v4_overfit_audit import circular_block_indices


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_regime_guard.json"
INITIAL_CAPITAL = 100_000.0
GuardMode = Literal[
    "absolute",
    "breadth",
    "systemic",
    "leadership",
    "full",
    "conjunctive",
    "confirmed_tail",
]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class RegimeGuardParams:
    mode: GuardMode = "full"
    absolute_votes_required: int = 2
    breadth_floor: float = 1.0 / 3.0
    stress_quantile: float = 0.80
    confidence_quantile: float = 0.20
    quantile_min_history: int = 60
    yellow_votes: int = 2
    red_votes: int = 3
    yellow_exposure: float = 0.50
    shock_1d: float = -0.05
    shock_3d: float = -0.10
    tail_volatility: float = 0.45
    tail_momentum_decay: float = -0.02
    tail_absolute_weak: float = 0.08
    tail_exposure: float = 0.70
    drawdown_yellow: float = 0.10
    drawdown_red: float = 0.15
    recovery_days: int = 2


def validate_params(params: RegimeGuardParams) -> None:
    if params.absolute_votes_required not in (1, 2, 3):
        raise ValueError("absolute_votes_required must be 1, 2, or 3")
    if not 0.0 < params.breadth_floor < 1.0:
        raise ValueError("breadth_floor must be in (0, 1)")
    if not 0.50 <= params.stress_quantile < 1.0:
        raise ValueError("stress_quantile must be in [0.5, 1)")
    if not 0.0 < params.confidence_quantile <= 0.50:
        raise ValueError("confidence_quantile must be in (0, 0.5]")
    if params.yellow_votes >= params.red_votes:
        raise ValueError("yellow_votes must be below red_votes")
    if not 0.0 < params.yellow_exposure < 1.0:
        raise ValueError("yellow_exposure must be in (0, 1)")
    if params.tail_volatility <= 0.0:
        raise ValueError("tail_volatility must be positive")
    if params.tail_momentum_decay >= 0.0:
        raise ValueError("tail_momentum_decay must be negative")
    if params.tail_absolute_weak <= 0.0:
        raise ValueError("tail_absolute_weak must be positive")
    if not 0.0 < params.tail_exposure < 1.0:
        raise ValueError("tail_exposure must be in (0, 1)")
    if not 0.0 < params.drawdown_yellow < params.drawdown_red < 1.0:
        raise ValueError("drawdown limits must be ordered in (0, 1)")
    if params.recovery_days < 1:
        raise ValueError("recovery_days must be positive")


def absolute_trend_votes(
    *, return_20d: float, return_60d: float, ma20_slope_5d: float
) -> int:
    """Three low-degree-of-freedom absolute-trend votes."""
    return sum((return_20d > 0.0, return_60d > 0.0, ma20_slope_5d > 0.0))


def expanding_thresholds(
    values: FloatArray,
    *,
    quantile: float,
    minimum_history: int,
    default: float,
) -> FloatArray:
    """Prior-only expanding quantile; current T never enters its own threshold."""
    result: FloatArray = np.full(len(values), default, dtype=np.float64)
    for index in range(minimum_history, len(values)):
        history = values[:index]
        finite = history[np.isfinite(history)]
        if len(finite) >= minimum_history:
            result[index] = float(np.quantile(finite, quantile))
    return result


def risk_votes(
    *,
    breadth_20d: float,
    median_return_20d: float,
    correlation_concentration: float,
    correlation_threshold: float,
    downside_vol_acceleration: float,
    downside_vol_threshold: float,
    rank_churn_10d: float,
    churn_threshold: float,
    leader_confidence: float,
    confidence_threshold: float,
    params: RegimeGuardParams,
) -> tuple[int, tuple[str, ...]]:
    """Four orthogonal stress families, each contributing at most one vote."""
    validate_params(params)
    reasons: list[str] = []
    if breadth_20d < params.breadth_floor or median_return_20d < 0.0:
        reasons.append("weak_breadth")
    if correlation_concentration > correlation_threshold:
        reasons.append("correlation_concentration")
    if downside_vol_acceleration > downside_vol_threshold:
        reasons.append("downside_vol_acceleration")
    if (
        rank_churn_10d > churn_threshold
        and leader_confidence < confidence_threshold
    ):
        reasons.append("unstable_leadership")
    return len(reasons), tuple(reasons)


def regime_exposure(
    *,
    absolute_votes: int,
    votes: int,
    stale_holding_shock: bool,
    portfolio_drawdown: float,
    risk_off_active: bool,
    recovery_streak: int,
    params: RegimeGuardParams,
) -> tuple[float, bool, int, tuple[str, ...]]:
    """Discrete causal guard with two-day hysteretic recovery from zero risk."""
    validate_params(params)
    absolute_ok = absolute_votes >= params.absolute_votes_required
    reasons: list[str] = []
    red = stale_holding_shock or votes >= params.red_votes
    if stale_holding_shock:
        reasons.append("stale_holding_shock")
    if votes >= params.red_votes:
        reasons.append("red_regime")
    if (
        portfolio_drawdown <= -params.drawdown_red
        and votes >= params.yellow_votes
    ):
        red = True
        reasons.append("drawdown_red_confirmation")

    if red:
        return 0.0, True, 0, tuple(reasons)

    if risk_off_active:
        recovery_streak = recovery_streak + 1 if absolute_ok and votes <= 1 else 0
        if recovery_streak < params.recovery_days:
            return 0.0, True, recovery_streak, ("recovery_hysteresis",)
        risk_off_active = False
        recovery_streak = 0
        reasons.append("recovered")

    exposure = 1.0
    if votes >= params.yellow_votes:
        exposure = params.yellow_exposure
        reasons.append("yellow_regime")
    elif not absolute_ok:
        exposure = params.yellow_exposure
        reasons.append("weak_absolute_trend")
    if (
        portfolio_drawdown <= -params.drawdown_yellow
        and votes >= 1
    ):
        exposure = min(exposure, params.yellow_exposure)
        reasons.append("drawdown_yellow_confirmation")
    return exposure, risk_off_active, recovery_streak, tuple(reasons)


def preregistered_variants() -> dict[str, RegimeGuardParams]:
    """Mechanism ablations, not a return-ranked parameter scan."""
    return {
        "absolute_only": RegimeGuardParams(mode="absolute"),
        "breadth_only": RegimeGuardParams(mode="breadth"),
        "systemic_only": RegimeGuardParams(mode="systemic"),
        "leadership_only": RegimeGuardParams(mode="leadership"),
        "v4_rg": RegimeGuardParams(mode="full"),
    }


def parameter_neighborhood() -> dict[str, RegimeGuardParams]:
    center = RegimeGuardParams(mode="confirmed_tail")
    return {
        "center": center,
        "volatility_80pct": replace(center, tail_volatility=0.36),
        "volatility_120pct": replace(center, tail_volatility=0.54),
        "exposure_80pct": replace(center, tail_exposure=0.56),
        "exposure_120pct": replace(center, tail_exposure=0.84),
        "decay_80pct": replace(center, tail_momentum_decay=-0.016),
        "decay_120pct": replace(center, tail_momentum_decay=-0.024),
        "weakness_80pct": replace(center, tail_absolute_weak=0.064),
        "weakness_120pct": replace(center, tail_absolute_weak=0.096),
    }


def conjunctive_exposure(
    *,
    risk_vote_reasons: tuple[str, ...],
    stale_holding_shock: bool,
    params: RegimeGuardParams,
) -> tuple[float, tuple[str, ...]]:
    """Phase-2 logical confirmation after the additive model was falsified.

    Single noisy warnings never change exposure.  Two different systemic
    measures must agree for zero risk; weak breadth must coincide with unstable
    leadership for half risk.  Portfolio drawdown is deliberately not an input.
    """
    reasons = set(risk_vote_reasons)
    systemic_confirmation = {
        "correlation_concentration",
        "downside_vol_acceleration",
    }.issubset(reasons)
    rotation_trap = {"weak_breadth", "unstable_leadership"}.issubset(reasons)
    if stale_holding_shock:
        return 0.0, ("stale_holding_shock",)
    if systemic_confirmation:
        return 0.0, ("confirmed_systemic_stress",)
    if rotation_trap:
        return params.yellow_exposure, ("confirmed_rotation_trap",)
    return 1.0, ()


def tail_decay_confirmation(
    *,
    volatility_20d: float,
    momentum_decay_5d: float,
    price_below_ma10: bool,
    momentum_score: float,
    params: RegimeGuardParams,
) -> bool:
    """Sparse V3-derived tail state; every condition must agree on T."""
    validate_params(params)
    return bool(
        volatility_20d > params.tail_volatility
        and momentum_decay_5d < params.tail_momentum_decay
        and price_below_ma10
        and momentum_score < params.tail_absolute_weak
    )


def confirmed_tail_exposure(
    *,
    holding_shock: bool,
    holding_changed: bool,
    tail_decay_confirmed: bool,
    params: RegimeGuardParams,
) -> tuple[float, tuple[str, ...]]:
    """Phase-3 sparse tail budget, including shock carryover after rotation."""
    validate_params(params)
    if holding_shock and not holding_changed:
        return 0.0, ("stale_holding_shock",)
    if holding_shock:
        return params.tail_exposure, ("shock_carryover_to_new_holding",)
    if tail_decay_confirmed:
        return params.tail_exposure, ("high_volatility_momentum_decay",)
    return 1.0, ()


def _close_prefix(
    data: dict[str, pd.DataFrame], code: str, index: int
) -> FloatArray:
    return cast(
        "FloatArray",
        np.asarray(data[code]["close"].values[: index + 1], dtype=np.float64),
    )


def compute_raw_factor_history(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute raw cross-asset state from information available through each T."""
    dates = rr.common_dates(data)
    index_maps = rr.build_index_maps(data, dates)
    codes = tuple(rq.ETF_POOL)
    leaders: list[str] = []
    rows: list[dict[str, Any]] = []
    for td in dates:
        idx_map = index_maps[td]
        returns_20: list[float] = []
        scores: list[tuple[str, float]] = []
        absolute_by_code: dict[str, int] = {}
        tail_state_by_code: dict[str, dict[str, float | bool]] = {}
        daily_blocks: list[FloatArray] = []
        valid = True
        for code in codes:
            if code not in idx_map:
                valid = False
                break
            close = _close_prefix(data, code, idx_map[code])
            if len(close) < 61:
                valid = False
                break
            return_10 = float(close[-1] / close[-11] - 1.0)
            return_20 = float(close[-1] / close[-21] - 1.0)
            return_60 = float(close[-1] / close[-61] - 1.0)
            ma20_now = float(np.mean(close[-20:]))
            ma20_prior = float(np.mean(close[-25:-5]))
            ma20_slope = ma20_now / ma20_prior - 1.0 if ma20_prior > 0.0 else 0.0
            absolute_by_code[code] = absolute_trend_votes(
                return_20d=return_20,
                return_60d=return_60,
                ma20_slope_5d=ma20_slope,
            )
            returns_20.append(return_20)
            momentum_score = 0.5 * return_10 + 0.5 * return_20
            prior_return_10 = float(close[-6] / close[-16] - 1.0)
            prior_return_20 = float(close[-6] / close[-26] - 1.0)
            prior_score = 0.5 * prior_return_10 + 0.5 * prior_return_20
            daily_returns = np.diff(close[-21:]) / close[-21:-1]
            tail_state_by_code[code] = {
                "volatility_20d": float(np.std(daily_returns) * np.sqrt(252)),
                "momentum_decay_5d": momentum_score - prior_score,
                "price_below_ma10": bool(close[-1] < np.mean(close[-10:])),
                "momentum_score": momentum_score,
            }
            scores.append((code, momentum_score))
            daily_blocks.append(np.diff(np.log(close[-61:])))
        if not valid:
            continue

        scores.sort(key=lambda item: item[1], reverse=True)
        leader = scores[0][0]
        leaders.append(leader)
        score_values: FloatArray = np.asarray(
            [score for _code, score in scores], dtype=np.float64
        )
        median_score = float(np.median(score_values))
        mad = float(np.median(np.abs(score_values - median_score)))
        confidence = float(
            (scores[0][1] - scores[1][1]) / max(mad, 1e-6)
        )
        recent_leaders = leaders[-10:]
        churn = sum(
            current != previous
            for previous, current in pairwise(recent_leaders)
        )

        return_matrix = np.column_stack(daily_blocks)
        correlation = cast(
            "FloatArray",
            np.asarray(
                np.corrcoef(return_matrix[-20:], rowvar=False),
                dtype=np.float64,
            ),
        )
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlation, 1.0)
        eigenvalues = np.linalg.eigvalsh(correlation)
        concentration = float(max(eigenvalues[-1], 0.0) / len(codes))
        equal_weight = np.mean(return_matrix, axis=1)
        downside_10 = math.sqrt(float(np.mean(np.minimum(equal_weight[-10:], 0.0) ** 2)))
        downside_60 = math.sqrt(float(np.mean(np.minimum(equal_weight, 0.0) ** 2)))
        acceleration = downside_10 / downside_60 if downside_60 > 0.0 else 1.0
        returns_20_array: FloatArray = np.asarray(
            returns_20, dtype=np.float64
        )
        rows.append({
            "trade_date": pd.Timestamp(td),
            "breadth_20d": float(np.mean(returns_20_array > 0.0)),
            "median_return_20d": float(np.median(returns_20_array)),
            "correlation_concentration": concentration,
            "downside_vol_acceleration": acceleration,
            "rank_churn_10d": float(churn),
            "leader": leader,
            "leader_confidence": confidence,
            "absolute_votes_by_code": absolute_by_code,
            "tail_state_by_code": tail_state_by_code,
        })
    return pd.DataFrame(rows)


def build_regime_inputs(
    raw: pd.DataFrame,
    curve: pd.DataFrame,
    params: RegimeGuardParams,
) -> pd.DataFrame:
    """Align prior-only thresholds and current V4 holding to the sleeve path."""
    validate_params(params)
    frame = raw.copy().reset_index(drop=True)
    for column, quantile, default in (
        ("correlation_concentration", params.stress_quantile, math.inf),
        ("downside_vol_acceleration", params.stress_quantile, math.inf),
        ("rank_churn_10d", params.stress_quantile, math.inf),
        ("leader_confidence", params.confidence_quantile, -math.inf),
    ):
        frame[f"{column}_threshold"] = expanding_thresholds(
            np.asarray(frame[column], dtype=np.float64),
            quantile=quantile,
            minimum_history=params.quantile_min_history,
            default=default,
        )
    factor_map = {
        pd.Timestamp(row["trade_date"]).normalize(): row
        for row in frame.to_dict("records")
    }
    aligned: list[dict[str, Any]] = []
    for row in curve.to_dict("records"):
        td = pd.Timestamp(row["trade_date"]).normalize()
        factor = factor_map[td]
        holding = str(row["holding"])
        absolute_by_code = factor["absolute_votes_by_code"]
        tail_by_code = factor["tail_state_by_code"]
        tail = tail_by_code.get(holding, {
            "volatility_20d": 0.0,
            "momentum_decay_5d": 0.0,
            "price_below_ma10": False,
            "momentum_score": math.inf,
        })
        count, reasons = risk_votes(
            breadth_20d=float(factor["breadth_20d"]),
            median_return_20d=float(factor["median_return_20d"]),
            correlation_concentration=float(factor["correlation_concentration"]),
            correlation_threshold=float(
                factor["correlation_concentration_threshold"]
            ),
            downside_vol_acceleration=float(factor["downside_vol_acceleration"]),
            downside_vol_threshold=float(
                factor["downside_vol_acceleration_threshold"]
            ),
            rank_churn_10d=float(factor["rank_churn_10d"]),
            churn_threshold=float(factor["rank_churn_10d_threshold"]),
            leader_confidence=float(factor["leader_confidence"]),
            confidence_threshold=float(factor["leader_confidence_threshold"]),
            params=params,
        )
        factor_payload: dict[str, Any] = {
            str(key): value
            for key, value in factor.items()
            if key not in {"absolute_votes_by_code", "tail_state_by_code"}
        }
        aligned.append({
            **factor_payload,
            "trade_date": td,
            "holding": holding,
            "absolute_votes": int(absolute_by_code.get(holding, 0)),
            "risk_votes": count,
            "risk_vote_reasons": reasons,
            "tail_volatility_20d": float(tail["volatility_20d"]),
            "tail_momentum_decay_5d": float(tail["momentum_decay_5d"]),
            "tail_price_below_ma10": bool(tail["price_below_ma10"]),
            "tail_momentum_score": float(tail["momentum_score"]),
            "tail_decay_confirmed": tail_decay_confirmation(
                volatility_20d=float(tail["volatility_20d"]),
                momentum_decay_5d=float(tail["momentum_decay_5d"]),
                price_below_ma10=bool(tail["price_below_ma10"]),
                momentum_score=float(tail["momentum_score"]),
                params=params,
            ),
        })
    return pd.DataFrame(aligned)


def _ablation_exposure(
    mode: GuardMode,
    row: pd.Series,
    params: RegimeGuardParams,
) -> tuple[float, tuple[str, ...]]:
    if mode == "absolute":
        return (
            (1.0, ())
            if int(row["absolute_votes"]) >= params.absolute_votes_required
            else (params.yellow_exposure, ("weak_absolute_trend",))
        )
    if mode == "breadth":
        stress = (
            float(row["breadth_20d"]) < params.breadth_floor
            or float(row["median_return_20d"]) < 0.0
        )
        return (
            (params.yellow_exposure, ("weak_breadth",))
            if stress else (1.0, ())
        )
    if mode == "systemic":
        reasons = tuple(
            reason for reason in row["risk_vote_reasons"]
            if reason in {"correlation_concentration", "downside_vol_acceleration"}
        )
        if len(reasons) == 2:
            return 0.0, reasons
        if reasons:
            return params.yellow_exposure, reasons
        return 1.0, ()
    if mode == "leadership":
        stress = "unstable_leadership" in row["risk_vote_reasons"]
        return (
            (params.yellow_exposure, ("unstable_leadership",))
            if stress else (1.0, ())
        )
    raise ValueError(f"unsupported ablation mode: {mode}")


def run_guard_overlay(
    baseline: dict[str, Any],
    safe_returns: FloatArray,
    held_1d: FloatArray,
    held_3d: FloatArray,
    holding_changes: np.ndarray[Any, np.dtype[np.bool_]],
    inputs: pd.DataFrame,
    params: RegimeGuardParams,
    *,
    cost_multiplier: float,
) -> dict[str, Any]:
    """Causal daily sleeve overlay; T factors set only the next exposure."""
    validate_params(params)
    base_curve = baseline["equity_curve"]
    base_returns = rb._curve_returns(base_curve)
    if not (
        len(base_returns)
        == len(safe_returns)
        == len(held_1d)
        == len(held_3d)
        == len(holding_changes)
        == len(inputs)
    ):
        raise ValueError("guard inputs are not aligned")

    wealth = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    exposure = 1.0
    risk_off_active = False
    recovery_streak = 0
    fee_per_reallocation = 2.0 * (rq.FEE + rq.SLIPPAGE) * cost_multiplier
    controller_cost = 0.0
    turnover = 0.0
    reason_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index, base_row in enumerate(base_curve.to_dict("records")):
        used_exposure = exposure
        risky_value = wealth * used_exposure * (1.0 + base_returns[index])
        safe_value = wealth * (1.0 - used_exposure) * (1.0 + safe_returns[index])
        wealth = risky_value + safe_value
        drifted_exposure = risky_value / wealth if wealth > 0.0 else 0.0
        peak = max(peak, wealth)
        portfolio_drawdown = wealth / peak - 1.0 if peak > 0.0 else 0.0
        signal = inputs.iloc[index]
        holding_shock = bool(
            held_1d[index] <= params.shock_1d
            or held_3d[index] <= params.shock_3d
        )
        stale_shock = bool(not holding_changes[index] and holding_shock)
        if params.mode == "full":
            next_exposure, risk_off_active, recovery_streak, reasons = regime_exposure(
                absolute_votes=int(signal["absolute_votes"]),
                votes=int(signal["risk_votes"]),
                stale_holding_shock=stale_shock,
                portfolio_drawdown=portfolio_drawdown,
                risk_off_active=risk_off_active,
                recovery_streak=recovery_streak,
                params=params,
            )
        elif params.mode == "conjunctive":
            next_exposure, reasons = conjunctive_exposure(
                risk_vote_reasons=tuple(signal["risk_vote_reasons"]),
                stale_holding_shock=stale_shock,
                params=params,
            )
        elif params.mode == "confirmed_tail":
            next_exposure, reasons = confirmed_tail_exposure(
                holding_shock=holding_shock,
                holding_changed=bool(holding_changes[index]),
                tail_decay_confirmed=bool(signal["tail_decay_confirmed"]),
                params=params,
            )
        else:
            next_exposure, reasons = _ablation_exposure(params.mode, signal, params)
        reason_counts.update(reasons)
        reallocation = abs(next_exposure - drifted_exposure)
        cost = wealth * reallocation * fee_per_reallocation
        wealth -= cost
        controller_cost += cost
        turnover += reallocation
        rows.append({
            "trade_date": pd.Timestamp(base_row["trade_date"]),
            "equity": wealth,
            "holding": base_row["holding"],
            "exposure_used": used_exposure,
            "next_exposure": next_exposure,
            "portfolio_drawdown": portfolio_drawdown,
            "absolute_votes": int(signal["absolute_votes"]),
            "risk_votes": int(signal["risk_votes"]),
            "risk_vote_reasons": list(signal["risk_vote_reasons"]),
            "decision_reasons": list(reasons),
            "breadth_20d": float(signal["breadth_20d"]),
            "median_return_20d": float(signal["median_return_20d"]),
            "correlation_concentration": float(signal["correlation_concentration"]),
            "downside_vol_acceleration": float(signal["downside_vol_acceleration"]),
            "rank_churn_10d": float(signal["rank_churn_10d"]),
            "leader_confidence": float(signal["leader_confidence"]),
            "held_1d_return": float(held_1d[index]),
            "held_3d_return": float(held_3d[index]),
            "holding_shock": holding_shock,
            "holding_changed": bool(holding_changes[index]),
            "stale_holding_shock": stale_shock,
            "tail_volatility_20d": float(signal["tail_volatility_20d"]),
            "tail_momentum_decay_5d": float(signal["tail_momentum_decay_5d"]),
            "tail_price_below_ma10": bool(signal["tail_price_below_ma10"]),
            "tail_momentum_score": float(signal["tail_momentum_score"]),
            "tail_decay_confirmed": bool(signal["tail_decay_confirmed"]),
            "risk_off_active": risk_off_active,
            "recovery_streak": recovery_streak,
            "controller_cost": cost,
        })
        exposure = next_exposure

    curve = pd.DataFrame(rows)
    metrics = rb._risk_metrics(curve)
    annual = rb._annual_rows(curve)
    metrics.update({
        "cost_multiplier": cost_multiplier,
        "average_exposure": float(curve["exposure_used"].mean()),
        "days_full_exposure": int(np.sum(curve["exposure_used"] >= 0.999)),
        "days_half_exposure": int(np.sum(
            np.isclose(curve["exposure_used"], params.yellow_exposure)
        )),
        "days_zero_exposure": int(np.sum(curve["exposure_used"] <= 0.001)),
        "days_reduced_exposure": int(np.sum(
            (curve["exposure_used"] > 0.001)
            & (curve["exposure_used"] < 0.999)
        )),
        "controller_turnover": turnover,
        "controller_cost": controller_cost,
        "negative_years": int(sum(row["return"] < 0.0 for row in annual)),
        "reason_counts": dict(reason_counts),
    })
    return {
        "params": asdict(params),
        "metrics": metrics,
        "annual": annual,
        "rolling_252": rb._rolling_252(curve),
        "equity_curve": curve,
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result.get("params", asdict(v4.V4_PARAMS)),
        "metrics": result["metrics"],
        "annual": result.get("annual", rb._annual_rows(result["equity_curve"])),
        "rolling_252": result.get(
            "rolling_252", rb._rolling_252(result["equity_curve"])
        ),
    }


def max_drawdown_episode(curve: pd.DataFrame) -> dict[str, Any]:
    equity = np.concatenate((
        np.asarray([INITIAL_CAPITAL], dtype=float),
        np.asarray(curve["equity"], dtype=float),
    ))
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(equity[: trough + 1]))
    peak_date = (
        "initial" if peak == 0
        else str(pd.Timestamp(curve.iloc[peak - 1]["trade_date"]).date())
    )
    trough_date = str(pd.Timestamp(curve.iloc[trough - 1]["trade_date"]).date())
    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_value": float(equity[peak]),
        "trough_value": float(equity[trough]),
        "max_drawdown": float(drawdown[trough]),
        "trading_days": trough - peak,
    }


def window_return(curve: pd.DataFrame, start: str, end: str) -> float:
    dates = pd.to_datetime(curve["trade_date"])
    start_row = curve.loc[dates == pd.Timestamp(start)]
    end_row = curve.loc[dates == pd.Timestamp(end)]
    if start_row.empty or end_row.empty:
        raise ValueError("window boundary missing from curve")
    return float(end_row.iloc[0]["equity"] / start_row.iloc[0]["equity"] - 1.0)


def bootstrap_guard(
    baseline: dict[str, Any],
    selected: dict[str, Any],
    *,
    bootstrap_samples: int,
    block_size: int,
    seed: int,
) -> dict[str, Any]:
    base = rb._curve_returns(baseline["equity_curve"])
    candidate = rb._curve_returns(selected["equity_curve"])
    rng = np.random.default_rng(seed)
    retention: list[float] = []
    improvements: list[float] = []
    passes = 0
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
        cagr_retention = candidate_cagr / base_cagr if base_cagr > 0.0 else 0.0
        improvement = 1.0 - candidate_mdd / base_mdd if base_mdd > 0.0 else 0.0
        retention.append(cagr_retention)
        improvements.append(improvement)
        passes += int(cagr_retention >= 0.85 and improvement >= 0.20)
    return {
        "block_size": block_size,
        "bootstrap_samples": bootstrap_samples,
        "probability_joint_85pct_cagr_20pct_mdd_rule": passes / bootstrap_samples,
        "cagr_retention_ci95": np.quantile(retention, (0.025, 0.975)).tolist(),
        "mdd_improvement_ci95": np.quantile(
            improvements, (0.025, 0.975)
        ).tolist(),
        "scope_warning": "resamples fixed-policy daily returns; no parameter reselection",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 regime guard research")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 500:
        raise ValueError("use at least 500 bootstrap samples")
    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V4-RG research expects disabled legacy downsizing")

    data = rq.load_data()
    raw = compute_raw_factor_history(data)
    baseline_by_cost = {
        cost: run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=cost)
        for cost in (0.0, 1.0, 2.0, 3.0)
    }
    baseline = baseline_by_cost[1.0]
    safe, held_1d, held_3d, _position_dd, holding_changes = rb.observable_inputs(
        data, baseline["equity_curve"]
    )
    variants: dict[str, dict[str, Any]] = {}
    for name, params in preregistered_variants().items():
        inputs = build_regime_inputs(raw, baseline["equity_curve"], params)
        variants[name] = run_guard_overlay(
            baseline,
            cast("FloatArray", safe),
            cast("FloatArray", held_1d),
            cast("FloatArray", held_3d),
            holding_changes,
            inputs,
            params,
            cost_multiplier=1.0,
        )
    revised_params = RegimeGuardParams(mode="conjunctive")
    revised_inputs = build_regime_inputs(
        raw, baseline["equity_curve"], revised_params
    )
    variants["v4_rg_conjunctive"] = run_guard_overlay(
        baseline,
        cast("FloatArray", safe),
        cast("FloatArray", held_1d),
        cast("FloatArray", held_3d),
        holding_changes,
        revised_inputs,
        revised_params,
        cost_multiplier=1.0,
    )
    tail_params = RegimeGuardParams(mode="confirmed_tail")
    tail_inputs = build_regime_inputs(raw, baseline["equity_curve"], tail_params)
    variants["v4_rg_confirmed_tail"] = run_guard_overlay(
        baseline,
        cast("FloatArray", safe),
        cast("FloatArray", held_1d),
        cast("FloatArray", held_3d),
        holding_changes,
        tail_inputs,
        tail_params,
        cost_multiplier=1.0,
    )
    selected_name = "v4_rg_confirmed_tail"
    selected = variants[selected_name]

    cost_pressure: dict[str, Any] = {}
    for cost, cost_baseline in baseline_by_cost.items():
        safe_c, held_1d_c, held_3d_c, _dd_c, changes_c = rb.observable_inputs(
            data, cost_baseline["equity_curve"]
        )
        params = RegimeGuardParams(mode="confirmed_tail")
        inputs_c = build_regime_inputs(raw, cost_baseline["equity_curve"], params)
        guarded = run_guard_overlay(
            cost_baseline,
            cast("FloatArray", safe_c),
            cast("FloatArray", held_1d_c),
            cast("FloatArray", held_3d_c),
            changes_c,
            inputs_c,
            params,
            cost_multiplier=cost,
        )
        cost_pressure[f"{cost:.0f}x"] = {
            "baseline_v4": compact(cost_baseline),
            selected_name: compact(guarded),
        }

    neighbors: list[dict[str, Any]] = []
    for name, params in parameter_neighborhood().items():
        inputs_n = build_regime_inputs(raw, baseline["equity_curve"], params)
        result = run_guard_overlay(
            baseline,
            cast("FloatArray", safe),
            cast("FloatArray", held_1d),
            cast("FloatArray", held_3d),
            holding_changes,
            inputs_n,
            params,
            cost_multiplier=1.0,
        )
        recent_n = mf.recent_cluster(baseline, result)
        metrics = result["metrics"]
        retention_n = metrics["cagr"] / baseline["metrics"]["cagr"]
        improvement_n = 1.0 - abs(metrics["max_drawdown"]) / abs(
            baseline["metrics"]["max_drawdown"]
        )
        neighbors.append({
            "name": name,
            "params": asdict(params),
            "final_value": metrics["final_value"],
            "cagr": metrics["cagr"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe"],
            "average_exposure": metrics["average_exposure"],
            "cagr_retention": retention_n,
            "mdd_improvement": improvement_n,
            "recent_cluster_loss_reduction": recent_n["loss_reduction"],
            "joint_passed": bool(
                retention_n >= 0.85
                and improvement_n >= 0.20
                and recent_n["loss_reduction"] >= 0.40
            ),
        })

    baseline_metrics = baseline["metrics"]
    selected_metrics = selected["metrics"]
    cagr_retention = selected_metrics["cagr"] / baseline_metrics["cagr"]
    mdd_improvement = 1.0 - abs(selected_metrics["max_drawdown"]) / abs(
        baseline_metrics["max_drawdown"]
    )
    recent = mf.recent_cluster(baseline, selected)
    baseline_episode = max_drawdown_episode(baseline["equity_curve"])
    selected_window_return = window_return(
        selected["equity_curve"],
        baseline_episode["peak_date"],
        baseline_episode["trough_date"],
    )
    baseline_window_return = window_return(
        baseline["equity_curve"],
        baseline_episode["peak_date"],
        baseline_episode["trough_date"],
    )
    historical_window_improvement = 1.0 - abs(selected_window_return) / abs(
        baseline_window_return
    )
    cost2 = cost_pressure["2x"][selected_name]["metrics"]
    acceptance_checks = {
        "cagr_retention_at_least_85pct": cagr_retention >= 0.85,
        "mdd_reduction_at_least_20pct": mdd_improvement >= 0.20,
        "baseline_mdd_window_loss_reduction_at_least_30pct": (
            historical_window_improvement >= 0.30
        ),
        "recent_cluster_loss_reduction_at_least_40pct": (
            recent["loss_reduction"] >= 0.40
        ),
        "no_additional_negative_year": (
            selected_metrics["negative_years"]
            <= sum(row["return"] < 0.0 for row in rb._annual_rows(
                baseline["equity_curve"]
            ))
        ),
        "positive_at_2x_cost": cost2["final_value"] > INITIAL_CAPITAL,
        "mdd_below_40pct_at_2x_cost": cost2["max_drawdown"] > -0.40,
    }
    bootstrap = {
        str(block): bootstrap_guard(
            baseline,
            selected,
            bootstrap_samples=args.bootstrap_samples,
            block_size=block,
            seed=20261200 + block,
        )
        for block in (5, 20, 60)
    }
    segments = {
        label: {
            "baseline_v4": rr.segment_metrics(baseline["equity_curve"], start, end),
            **{
                name: rr.segment_metrics(result["equity_curve"], start, end)
                for name, result in variants.items()
            },
        }
        for label, start, end in (
            ("retrospective_2020_2023", "2020-06-19", "2023-12-29"),
            ("retrospective_2024_current", "2024-01-01", "2026-08-11"),
        )
    }
    center_pass = bool(all(acceptance_checks.values()))
    neighborhood_pass_rate = sum(row["joint_passed"] for row in neighbors) / len(
        neighbors
    )
    classification = (
        "adaptive_phase3_shadow_candidate_requires_oos"
        if center_pass and neighborhood_pass_rate >= 0.80
        else (
            "center_pass_but_not_robust"
            if center_pass else "failed_joint_constraints"
        )
    )
    payload: dict[str, Any] = {
        "meta": {
            "strategy": "frozen canonical V4 sleeve plus V4-RG regime guard",
            "data_start": str(baseline["equity_curve"]["trade_date"].iloc[0].date()),
            "data_end": str(baseline["equity_curve"]["trade_date"].iloc[-1].date()),
            "observations": len(baseline["equity_curve"]),
            "design_date": "2026-08-12",
            "live_post_design_oos_observations": 0,
            "pre_registered_main_variants": list(preregistered_variants()),
            "adaptive_phase2_revision": selected_name,
            "adaptive_phase2_reason": (
                "the additive vote/drawdown state was falsified by 36.4% average "
                "de-risking, a 896-day drawdown, and three negative years; phase 2 "
                "removes drawdown feedback and requires logical concurrence between "
                "orthogonal signals"
            ),
            "adaptive_phase2_candidate_count": 1,
            "adaptive_phase3_revision": selected_name,
            "adaptive_phase3_reason": (
                "phase 2 reduced recent clustered loss but missed the 2022 rotation "
                "drawdown because its stale-shock rule stopped applying whenever V4 "
                "rotated; phase 3 carries shock risk into a new holding at 70% and "
                "adds only the previously studied high-volatility plus triple momentum-"
                "decay confirmation"
            ),
            "adaptive_phase3_candidate_count": 1,
            "historical_prior_warning": (
                "0.70 exposure, 0.45 annualized volatility, -0.02 five-day momentum "
                "decay, and 0.08 weak-momentum thresholds come from the old V3 tail-"
                "risk study; its results are not directly comparable to frozen V4"
            ),
            "no_return_weight_optimization": True,
            "no_lookahead": (
                "all T factors and prior-only expanding quantiles set T+1 exposure; "
                "current T is excluded from its own percentile threshold"
            ),
            "execution_warning": "daily close is a same-price proxy for 14:50",
            "frozen_sleeve_warning": (
                "the guard scales a frozen V4 sleeve; lower guarded equity is not fed "
                "back into canonical risk_overrides"
            ),
            "production_status": "research_only_no_production_change",
        },
        "theory": {
            "alpha_layer": "canonical V4 slow/fast consensus remains unchanged",
            "absolute_trend": "2 of: 20d return>0, 60d return>0, MA20 5d slope>0",
            "breadth": "risk vote if positive-20d breadth<1/3 or median 20d return<0",
            "systemic": (
                "risk votes for 20d correlation eigenvalue concentration and 10d/60d "
                "downside-vol acceleration above prior-only expanding 80th percentiles"
            ),
            "leadership": (
                "risk vote only when 10d rank churn is above its prior 80th percentile "
                "and robust top-vs-second confidence is below its prior 20th percentile"
            ),
            "shock": "unchanged V4 holding 1d<=-5% or 3d<=-10% sets next exposure to zero",
            "state_machine": (
                "0/1 votes=full unless absolute trend weak; 2 votes=50%; 3+=zero; "
                "10%/15% drawdown requires confirming risk votes; zero-risk recovery "
                "requires two consecutive healthy observations"
            ),
            "phase2_conjunctive_rule": (
                "zero only for stale held-asset shock or simultaneous correlation and "
                "downside-vol stress; half only for simultaneous weak breadth and "
                "unstable leadership; single warnings and portfolio drawdown do nothing"
            ),
            "phase3_confirmed_tail_rule": (
                "if the asset that earned T's sleeve return falls <=-5% in one day or "
                "<=-10% in three days, unchanged V4 holding sets T+1 risk to zero while "
                "a V4 rotation carries 70% risk into the new holding; otherwise 70% "
                "applies only when target vol20>45%, five-day momentum decay<-2pp, "
                "price<MA10, and 10/20d momentum score<8% all agree"
            ),
            "objective": (
                "retain >=85% V4 CAGR, reduce MDD >=20%, reduce the baseline MDD window "
                ">=30% and recent clustered loss >=40%, with no extra negative year and "
                "2x-cost viability"
            ),
            "primary_references": [
                "https://doi.org/10.3386/w22208",
                "https://doi.org/10.3386/w20439",
                "https://doi.org/10.3905/jpm.2017.44.1.015",
            ],
        },
        "baseline_v4": compact(baseline),
        "variants": {name: compact(result) for name, result in variants.items()},
        "acceptance": {
            "cagr_retention": cagr_retention,
            "mdd_improvement": mdd_improvement,
            "baseline_mdd_window_return": baseline_window_return,
            "selected_same_window_return": selected_window_return,
            "baseline_mdd_window_loss_reduction": historical_window_improvement,
            "checks": acceptance_checks,
            "passed": center_pass,
        },
        "baseline_max_drawdown_episode": baseline_episode,
        "selected_max_drawdown_episode": max_drawdown_episode(
            selected["equity_curve"]
        ),
        "recent_cluster_replay": recent,
        "documented_failure_replay": mf.replay_documented_failure(),
        "factor_snapshot_baseline_mdd": selected["equity_curve"].loc[
            selected["equity_curve"]["trade_date"].isin(pd.to_datetime([
                "2022-03-09", "2022-03-10", "2022-04-01", "2022-04-15",
                "2022-04-25", "2022-05-05", "2022-05-24",
            ]))
        ].to_dict("records"),
        "parameter_neighborhood": {
            "points": len(neighbors),
            "joint_passes": sum(row["joint_passed"] for row in neighbors),
            "joint_pass_rate": neighborhood_pass_rate,
            "rows": neighbors,
        },
        "paired_block_bootstrap": bootstrap,
        "cost_pressure": cost_pressure,
        "retrospective_segments": segments,
        "assessment": {
            "selected": selected_name,
            "classification": classification,
            "joint_constraints_passed": center_pass,
            "neighborhood_pass_rate": neighborhood_pass_rate,
            "production_status": "research_only_no_production_change",
            "interpretation": (
                "MDD is treated as an audit statistic, not a directly optimized target; "
                "zero untouched post-design observations prevent validation."
            ),
        },
    }

    print("\nV4-RG theory-first regime guard")
    print("variant               final       CAGR Sharpe     MDD avgExp zero")
    for name, result in [("baseline_v4", baseline), *variants.items()]:
        metrics = result["metrics"]
        print(
            f"{name:<21} {metrics['final_value']:>11,.0f} "
            f"{metrics['cagr']:>7.2%} {metrics['sharpe']:>6.3f} "
            f"{metrics['max_drawdown']:>7.2%} "
            f"{metrics.get('average_exposure', 1.0):>6.1%} "
            f"{metrics.get('days_zero_exposure', 0):>4}"
        )
    print(
        f"\nselected retention={cagr_retention:.1%} "
        f"MDD improvement={mdd_improvement:.1%} "
        f"2022-window improvement={historical_window_improvement:.1%} "
        f"recent-cluster improvement={recent['loss_reduction']:.1%} "
        f"assessment={classification}"
    )
    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
