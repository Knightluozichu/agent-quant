"""Production decision core for Qixing V4 full-pool fast/slow consensus.

This module is deliberately pure: callers supply data sliced through decision
date T, the V3-G eligible candidate set, confirmation state, and lock age.
Research replay and live execution import the same functions from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

try:
    from scripts import run_qixing_v3 as rq
except ModuleNotFoundError:
    import run_qixing_v3 as rq

STRATEGY_NAME = "V4"
STRATEGY_ID = "QIXING_V4_FULL_POOL_CONSENSUS_20260811"
STATE_SCHEMA_VERSION = 1
Mode = Literal["slow", "fast", "consensus", "or"]


@dataclass(frozen=True)
class AssetFactors:
    """Factors observable at decision time T for one asset."""

    code: str
    eligible: bool
    slow_momentum: float
    return_3d: float
    return_5d: float
    acceleration_5d: float
    trend_strength: float
    vol_adjusted_5d: float
    drawdown_5d: float


@dataclass(frozen=True)
class FullPoolParams:
    enabled: bool = True
    mode: Mode = "slow"
    slow_gap: float = 0.005
    fast_5d_gap: float = 0.015
    fast_3d_gap: float = 0.0075
    minimum_target_momentum: float = 0.0
    minimum_hold_days: int = 3
    confirmation_hits: int = 1
    confirmation_window: int = 2

    @classmethod
    def disabled(cls) -> FullPoolParams:
        return cls(enabled=False)


@dataclass(frozen=True)
class FullPoolDecision:
    triggered: bool
    target: str | None = None
    reasons: tuple[str, ...] = ()
    blocked_by: str = ""


V4_PARAMS = FullPoolParams(
    mode="consensus",
    slow_gap=0.0075,
    fast_5d_gap=0.0225,
    fast_3d_gap=0.01125,
    confirmation_hits=2,
)


def _config_payload(params: FullPoolParams = V4_PARAMS) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "params": asdict(params),
        "candidate_pool": list(rq.ETF_POOL),
        "defense": rq.DEFENSE,
        "rebalance_days": rq.REBALANCE_DAYS,
        "warmup": 130,
    }


def config_hash(params: FullPoolParams = V4_PARAMS) -> str:
    payload = json.dumps(_config_payload(params), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


CONFIG_HASH = config_hash()


def _period_return(close: np.ndarray, period: int) -> float:
    if len(close) <= period or close[-period - 1] <= 0:
        return 0.0
    return float(close[-1] / close[-period - 1] - 1.0)


def asset_factors(code: str, close: np.ndarray) -> AssetFactors:
    """Compute the archived V4 factor vector from a T-sliced close series."""
    r3 = _period_return(close, 3)
    r5 = _period_return(close, 5)
    r10 = _period_return(close, 10)
    r20 = _period_return(close, 20)
    slow = 0.5 * r10 + 0.5 * r20
    expected_5d = (1.0 + r20) ** 0.25 - 1.0 if r20 > -1.0 else r20 / 4.0
    acceleration = r5 - expected_5d

    ma5 = float(np.mean(close[-5:])) if len(close) >= 5 else float(close[-1])
    ma10 = float(np.mean(close[-10:])) if len(close) >= 10 else ma5
    price_vs_ma5 = float(close[-1] / ma5 - 1.0) if ma5 > 0 else 0.0
    ma_slope = float(ma5 / ma10 - 1.0) if ma10 > 0 else 0.0
    trend = 0.5 * price_vs_ma5 + 0.5 * ma_slope

    if len(close) >= 21:
        daily = np.diff(close[-21:]) / close[-21:-1]
        vol_5d = float(np.std(daily) * np.sqrt(5))
    else:
        vol_5d = 0.0
    vol_adjusted = r5 / vol_5d if vol_5d > 1e-12 else 0.0

    recent_peak = float(np.max(close[-6:])) if len(close) >= 6 else float(close[-1])
    drawdown = float(close[-1] / recent_peak - 1.0) if recent_peak > 0 else 0.0
    eligible = bool(len(close) >= 21 and rq.check_single_day_drop(close))
    return AssetFactors(
        code=code,
        eligible=eligible,
        slow_momentum=slow,
        return_3d=r3,
        return_5d=r5,
        acceleration_5d=acceleration,
        trend_strength=trend,
        vol_adjusted_5d=vol_adjusted,
        drawdown_5d=drawdown,
    )


def compute_factors(
    data: dict[str, pd.DataFrame],
    idx_map: dict[str, int],
    codes: Sequence[str],
) -> dict[str, AssetFactors]:
    """Compute factors using only rows at or before ``idx_map[code]``."""
    result: dict[str, AssetFactors] = {}
    for code in codes:
        if code not in data or code not in idx_map:
            continue
        close = data[code]["close"].values[: idx_map[code] + 1].astype(float)
        if len(close):
            result[code] = asset_factors(code, close)
    return result


def fast_score(factor: AssetFactors) -> float:
    return 0.5 * factor.return_3d + 0.5 * factor.return_5d


def decide_full_pool_handoff(
    *,
    holding: str | None,
    factors: dict[str, AssetFactors],
    params: FullPoolParams,
    signal_hits: int,
    days_since_rotation: int,
) -> FullPoolDecision:
    """Pure V4 decision using only supplied T-observable factors."""
    if not params.enabled:
        return FullPoolDecision(False, blocked_by="disabled")
    if not holding or holding not in factors:
        return FullPoolDecision(False, blocked_by="no_holding")
    if days_since_rotation < params.minimum_hold_days:
        return FullPoolDecision(False, blocked_by="minimum_hold")

    held = factors[holding]
    eligible = [
        factor
        for code, factor in factors.items()
        if code != holding
        and factor.eligible
        and factor.slow_momentum > params.minimum_target_momentum
    ]
    if not eligible:
        return FullPoolDecision(False, blocked_by="no_candidate")

    slow_leader = max(eligible, key=lambda item: item.slow_momentum)
    fast_eligible = [item for item in eligible if item.trend_strength > 0.0]
    fast_leader = max(fast_eligible, key=fast_score) if fast_eligible else None
    slow_ok = slow_leader.slow_momentum - held.slow_momentum >= params.slow_gap
    fast_ok = bool(
        fast_leader
        and fast_leader.return_5d - held.return_5d >= params.fast_5d_gap
        and fast_leader.return_3d - held.return_3d >= params.fast_3d_gap
    )

    target: str | None = None
    reasons: tuple[str, ...] = ()
    if params.mode == "slow" and slow_ok:
        target, reasons = slow_leader.code, ("slow",)
    elif params.mode == "fast" and fast_ok and fast_leader:
        target, reasons = fast_leader.code, ("fast",)
    elif (
        params.mode == "consensus"
        and slow_ok
        and fast_ok
        and fast_leader
        and slow_leader.code == fast_leader.code
    ):
        target, reasons = slow_leader.code, ("slow", "fast")
    elif params.mode == "or":
        if slow_ok:
            target, reasons = slow_leader.code, ("slow",)
            if fast_ok and fast_leader and fast_leader.code == target:
                reasons = ("slow", "fast")
        elif fast_ok and fast_leader:
            target, reasons = fast_leader.code, ("fast",)

    if target is None:
        return FullPoolDecision(False, blocked_by="no_signal")
    if signal_hits < params.confirmation_hits:
        return FullPoolDecision(False, blocked_by="confirmation")
    return FullPoolDecision(True, target=target, reasons=reasons)


def raw_candidate(
    holding: str | None,
    factors: dict[str, AssetFactors],
    params: FullPoolParams = V4_PARAMS,
) -> FullPoolDecision:
    """Evaluate today's raw signal before persistence and early-rotation lock."""
    return decide_full_pool_handoff(
        holding=holding,
        factors=factors,
        params=params,
        signal_hits=max(params.confirmation_hits, 1),
        days_since_rotation=10_000,
    )


def update_candidate_history(
    history: list[dict[str, Any]],
    *,
    trade_date: Any,
    raw_target: str | None,
    trading_dates: Sequence[Any],
    params: FullPoolParams = V4_PARAMS,
) -> tuple[list[dict[str, Any]], int]:
    """Idempotently record one raw target and return contiguous-window hits."""
    td = str(trade_date)
    dates = [str(item) for item in trading_dates]
    try:
        position = dates.index(td)
    except ValueError:
        position = len(dates) - 1
    window = max(params.confirmation_window, params.confirmation_hits, 1)
    valid_dates = dates[max(0, position - window + 1) : position + 1]
    normalized = {
        str(item.get("date")): item.get("target")
        for item in history
        if isinstance(item, dict) and item.get("date")
    }
    normalized[td] = raw_target
    result = [{"date": day, "target": normalized[day]} for day in valid_dates if day in normalized]
    hits = sum(item["target"] == raw_target for item in result) if raw_target else 0
    return result, hits


def trading_days_since(
    previous_date: str | None,
    current_date: Any,
    trading_dates: Sequence[Any],
) -> int:
    if not previous_date:
        return 10_000
    dates = [str(item) for item in trading_dates]
    try:
        return dates.index(str(current_date)) - dates.index(str(previous_date))
    except ValueError:
        return 10_000


def factors_payload(factors: dict[str, AssetFactors]) -> dict[str, dict[str, Any]]:
    return {code: asdict(factor) for code, factor in factors.items()}
