"""V4 + fixed portfolio holding stop research.

This is deliberately a research-only overlay.  It replays the canonical V4
path and adds exactly two pre-registered rules:

* current holding close-to-close return <= -5% over one trading day; or
* current holding close-to-close return <= -10% over three trading days.

When triggered, the target is the second-ranked eligible V3-G momentum
candidate (rank #2 overall, not rank #2 after removing the holding).  The
production V4 core is not modified by this file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_stop_team_research.json"
INITIAL_CAPITAL = 100_000.0
WARMUP = 130
STOP_1D = -0.05
STOP_3D = -0.10
STOP_MODES = (
    "fixed_shock_rank2",
    "v3g_confirmed_shock",
    "v4_first_confirmation_shock",
    "fixed_shock_cash",
    "v4_first_confirmation_cash",
)
ENTRY_GUARD_DAYS = 3
ENTRY_GUARD_ENTRY_LOSS = -0.10
ENTRY_GUARD_MODES = (
    "entry_guard_cash",
    "entry_guard_v4_or_cash",
    "entry_guard_best_or_cash",
    "entry_guard_selective",
)
MOMENTUM_FAILURE_MODES = (
    "stale_leader_cash",
    "stale_leader_best_or_cash",
)


@dataclass(frozen=True)
class EntryGuardDecision:
    """Decision from the research-only post-entry failure guard."""

    triggered: bool
    target: str | None = None
    exit_to_cash: bool = False
    reasons: tuple[str, ...] = ()


def _positive_relative_replacement(
    *,
    holding: str,
    candidates: list[tuple[str, float]],
    factors: dict[str, v4.AssetFactors],
) -> str | None:
    """Return the strongest qualified alternative without adding a tuned gap."""
    held = factors.get(holding)
    for code, _score in candidates:
        if code == holding:
            continue
        replacement = factors.get(code)
        if (
            held is not None
            and replacement is not None
            and replacement.eligible
            and replacement.slow_momentum > 0.0
            and replacement.trend_strength > 0.0
            and replacement.return_3d > held.return_3d
            and replacement.return_5d > held.return_5d
        ):
            return code
    return None


def entry_guard_decision(
    *,
    mode: str,
    holding: str | None,
    holding_age: int | None,
    entry_return: float,
    one_day: float,
    candidates: list[tuple[str, float]],
    factors: dict[str, v4.AssetFactors],
    first_v4_target: str | None,
    guard_days: int = ENTRY_GUARD_DAYS,
    day_threshold: float = STOP_1D,
    entry_loss_threshold: float = ENTRY_GUARD_ENTRY_LOSS,
) -> EntryGuardDecision:
    """Apply a low-degree-of-freedom guard only to a newly opened holding."""
    if mode not in ENTRY_GUARD_MODES:
        raise ValueError(f"unknown entry guard mode: {mode}")
    if (
        not holding
        or holding == rq.DEFENSE
        or holding_age is None
        or not 1 <= holding_age <= guard_days
    ):
        return EntryGuardDecision(False)

    reasons: list[str] = []
    if one_day <= day_threshold:
        reasons.append("entry_1d<=-5%")
    if entry_return <= entry_loss_threshold:
        reasons.append("entry_return<=-10%")
    if not reasons:
        return EntryGuardDecision(False)

    if mode == "entry_guard_selective":
        if entry_return <= entry_loss_threshold:
            return EntryGuardDecision(True, exit_to_cash=True, reasons=tuple(reasons))
        replacement = _positive_relative_replacement(
            holding=holding,
            candidates=candidates,
            factors=factors,
        )
        if one_day <= day_threshold and replacement:
            return EntryGuardDecision(
                True,
                target=replacement,
                reasons=(*reasons, "positive_relative_replacement"),
            )
        return EntryGuardDecision(False)

    if mode == "entry_guard_cash":
        return EntryGuardDecision(True, exit_to_cash=True, reasons=tuple(reasons))

    if mode == "entry_guard_v4_or_cash":
        if first_v4_target and first_v4_target != holding:
            return EntryGuardDecision(
                True,
                target=first_v4_target,
                reasons=(*reasons, "first_v4_consensus"),
            )
        return EntryGuardDecision(True, exit_to_cash=True, reasons=tuple(reasons))

    replacement = _positive_relative_replacement(
        holding=holding,
        candidates=candidates,
        factors=factors,
    )
    if replacement:
        return EntryGuardDecision(
            True,
            target=replacement,
            reasons=(*reasons, "positive_relative_replacement"),
        )
    return EntryGuardDecision(True, exit_to_cash=True, reasons=tuple(reasons))


def momentum_failure_decision(
    *,
    mode: str,
    holding: str | None,
    v4_would_exit: bool,
    one_day: float,
    three_day: float,
    candidates: list[tuple[str, float]],
    factors: dict[str, v4.AssetFactors],
    day_threshold: float = STOP_1D,
    three_day_threshold: float = STOP_3D,
) -> EntryGuardDecision:
    """Break a stale V4 leader only after an observable absolute shock.

    The gate is deliberately orthogonal to relative rank: it acts only when the
    canonical V4 state machine would otherwise retain the failed holding.  A
    qualified alternative needs no fitted gap; otherwise the rule exits to cash.
    """
    if mode not in MOMENTUM_FAILURE_MODES:
        raise ValueError(f"unknown momentum failure mode: {mode}")
    if not holding or holding == rq.DEFENSE or v4_would_exit:
        return EntryGuardDecision(False)

    reasons: list[str] = []
    if one_day <= day_threshold:
        reasons.append("failure_1d")
    if three_day <= three_day_threshold:
        reasons.append("failure_3d")
    if not reasons:
        return EntryGuardDecision(False)

    if mode == "stale_leader_best_or_cash":
        replacement = _positive_relative_replacement(
            holding=holding,
            candidates=candidates,
            factors=factors,
        )
        if replacement:
            return EntryGuardDecision(
                True,
                target=replacement,
                reasons=(*reasons, "qualified_alternative"),
            )
    return EntryGuardDecision(
        True,
        exit_to_cash=True,
        reasons=(*reasons, "no_qualified_alternative"),
    )


def staged_top_up_allowed(
    *,
    holding: str | None,
    holding_age: int | None,
    confirmation_days: int,
    selected_target: str | None,
    factors: dict[str, v4.AssetFactors],
    guard_triggered: bool,
) -> bool:
    """Return whether a research-only staged position may be topped up."""
    if confirmation_days < 1:
        raise ValueError("confirmation_days must be at least one")
    if (
        not holding
        or holding == rq.DEFENSE
        or holding_age is None
        or holding_age < confirmation_days
        or selected_target != holding
        or guard_triggered
    ):
        return False
    held = factors.get(holding)
    return bool(
        held is not None
        and held.eligible
        and held.slow_momentum > 0.0
        and held.trend_strength > 0.0
    )


def _holding_returns(
    data: dict[str, pd.DataFrame],
    code: str | None,
    idx_map: dict[str, int],
) -> tuple[float, float]:
    """Return the holding's 1d and 3d close-to-close returns at T."""
    if not code or code == rq.DEFENSE or code not in idx_map:
        return 0.0, 0.0
    close: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(
        data[code]["close"].values[: idx_map[code] + 1], dtype=np.float64
    )
    one_day = float(close[-1] / close[-2] - 1.0) if len(close) >= 2 else 0.0
    three_day = float(close[-1] / close[-4] - 1.0) if len(close) >= 4 else 0.0
    return one_day, three_day


def _annual_rows(curve: pd.DataFrame) -> list[dict[str, Any]]:
    """Calendar-year returns using the last prior equity as the base."""
    if curve.empty:
        return []
    rows: list[dict[str, Any]] = []
    indexed = curve.copy()
    indexed["year"] = indexed["trade_date"].dt.year
    previous_equity = INITIAL_CAPITAL
    for _year, group in indexed.groupby("year", sort=True):
        year = int(group["year"].iloc[0])
        start = float(group["equity"].iloc[0])
        end = float(group["equity"].iloc[-1])
        base = previous_equity if year != int(indexed["year"].iloc[0]) else INITIAL_CAPITAL
        rows.append(
            {
                "year": int(year),
                "start_value": base,
                "end_value": end,
                "return": end / base - 1.0 if base else 0.0,
                "within_year_start": start,
            }
        )
        previous_equity = end
    return rows


def _enrich_stop_events(
    events: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    trading_dates: list[Any],
    index_maps: dict[Any, dict[str, int]],
) -> list[dict[str, Any]]:
    """Add forward relative returns without feeding future data into the run."""
    position = {td: i for i, td in enumerate(trading_dates)}
    enriched: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        i = position[event["trade_date_raw"]]
        now_map = index_maps[event["trade_date_raw"]]
        for horizon in (1, 2, 3, 5, 10, 20):
            if i + horizon >= len(trading_dates):
                continue
            future_td = trading_dates[i + horizon]
            future_map = index_maps[future_td]
            old_now = rr.price_at(data, event["from"], now_map)
            old_future = rr.price_at(data, event["from"], future_map)
            old_return = old_future / old_now - 1.0 if old_now > 0 else 0.0
            if event.get("to"):
                new_now = rr.price_at(data, event["to"], now_map)
                new_future = rr.price_at(data, event["to"], future_map)
                new_return = new_future / new_now - 1.0 if new_now > 0 else 0.0
            else:
                cash_until = event.get("cash_until")
                if cash_until not in (None, "None") and pd.Timestamp(future_td) > pd.Timestamp(
                    cash_until
                ):
                    continue
                new_return = 0.0
            row[f"forward_old_{horizon}d"] = old_return
            row[f"forward_new_{horizon}d"] = new_return
            row[f"forward_relative_{horizon}d"] = new_return - old_return
        row.pop("trade_date_raw", None)
        enriched.append(row)
    return enriched


def _stop_signal(
    *,
    mode: str,
    holding: str | None,
    candidates: list[tuple[str, float]],
    factors: dict[str, v4.AssetFactors],
    one_day: float,
    three_day: float,
    first_v4_target: str | None,
) -> tuple[bool, str | None, list[str], list[str]]:
    """Return (rule_triggered, target, reasons, raw_shock_reasons)."""
    if mode not in STOP_MODES:
        raise ValueError(f"unknown stop mode: {mode}")
    cash_mode = mode.endswith("_cash")
    base_mode = mode.removesuffix("_cash") if cash_mode else mode
    if not holding or holding == rq.DEFENSE:
        return False, None, [], []

    shock_reasons: list[str] = []
    if one_day <= STOP_1D:
        shock_reasons.append("1d<=-5%")
    if three_day <= STOP_3D:
        shock_reasons.append("3d<=-10%")
    if not shock_reasons:
        return False, None, [], []

    if base_mode in ("fixed_shock_rank2", "fixed_shock"):
        target = candidates[1][0] if len(candidates) >= 2 else None
        return True, target, shock_reasons, shock_reasons

    held = factors.get(holding)
    if held is None:
        return False, None, [], shock_reasons

    if base_mode == "v3g_confirmed_shock":
        if held.return_5d > 0.0 or held.slow_momentum > 0.0:
            return False, None, [], shock_reasons
        if len(candidates) < 2:
            return True, None, [*shock_reasons, "v3g_fast_slow_failure"], shock_reasons
        target = candidates[1][0]
        replacement = factors.get(target)
        if (
            replacement is None
            or not replacement.eligible
            or replacement.slow_momentum <= 0.0
            or replacement.trend_strength <= 0.0
        ):
            return True, None, [*shock_reasons, "v3g_fast_slow_failure"], shock_reasons
        return True, target, [*shock_reasons, "v3g_fast_slow_failure"], shock_reasons

    # The first-confirmation variant uses the same V4 factors and target as
    # production, but accepts one confirmation only when the held asset has
    # suffered the pre-registered shock.
    if first_v4_target and first_v4_target != holding:
        return (
            True,
            first_v4_target,
            [*shock_reasons, "v4_first_confirmation"],
            shock_reasons,
        )
    return False, None, [], shock_reasons


def run_v4_with_stop(
    data: dict[str, pd.DataFrame],
    *,
    cost_multiplier: float = 1.0,
    stop_mode: str = "fixed_shock_rank2",
    entry_guard_mode: str | None = None,
    entry_guard_days: int = ENTRY_GUARD_DAYS,
    entry_guard_day_threshold: float = STOP_1D,
    entry_guard_entry_loss: float = ENTRY_GUARD_ENTRY_LOSS,
    staged_entry_fraction: float | None = None,
    staged_confirmation_days: int = 2,
    momentum_failure_mode: str | None = None,
    momentum_failure_day_threshold: float = STOP_1D,
    momentum_failure_three_day_threshold: float = STOP_3D,
) -> dict[str, Any]:
    """Replay V4 with optional stop, post-entry guard, and staged entry."""
    if staged_entry_fraction is not None and not 0.0 < staged_entry_fraction <= 1.0:
        raise ValueError("staged_entry_fraction must be in (0, 1]")
    if staged_confirmation_days < 1:
        raise ValueError("staged_confirmation_days must be at least one")
    dates = rr.common_dates(data)
    trading_dates = dates[WARMUP:]
    rebalance_set = set(trading_dates[:: rq.REBALANCE_DAYS])
    index_maps = rr.build_index_maps(data, dates)
    factor_codes = (*tuple(rq.ETF_POOL), rq.DEFENSE)

    state: dict[str, Any] = {
        "cash": INITIAL_CAPITAL,
        "holding": None,
        "shares": 0.0,
        "entry_price": 0.0,
        "entry_position": None,
        "staged_entry_active": False,
        "peak_equity": INITIAL_CAPITAL,
        "risk_exposure": 1.0,
        "cooldown_until": None,
        "h3_holding": None,
        "h3_peak": 0.0,
        "cash_until": None,
    }
    last_early_rotation = -10_000
    candidate_history: list[str | None] = []
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    stop_events: list[dict[str, Any]] = []
    risk_events: list[dict[str, Any]] = []
    realtime_events: list[dict[str, Any]] = []
    scheduled_lock_blocks = 0
    signal_days = 0
    raw_shock_count = 0
    raw_shock_1d_count = 0
    raw_shock_3d_count = 0
    stop_trigger_count = 0
    stop_switch_count = 0
    stop_noop_same_rank_count = 0
    stop_no_rank2_count = 0
    stop_cash_exit_count = 0
    entry_guard_trigger_count = 0
    entry_guard_switch_count = 0
    entry_guard_cash_exit_count = 0
    staged_entry_count = 0
    staged_top_up_count = 0

    for position, td in enumerate(trading_dates):
        cash_until = state.get("cash_until")
        if cash_until is not None and td >= cash_until:
            state["cash_until"] = None
        cash_lock_active = state.get("cash_until") is not None
        idx_map = index_maps[td]
        holding = state["holding"]
        target, candidates, _best_score, _weak = rq.select_target(data, idx_map, holding)
        candidates, dropped = rr.apply_server_realtime_filter(data, idx_map, candidates)
        if dropped:
            realtime_events.extend({"date": str(td), **event} for event in dropped)
            target = candidates[0][0] if candidates else rq.DEFENSE
        daily_selected_target = target

        factors = v4.compute_factors(data, idx_map, factor_codes)
        candidate_codes = {code for code, _score in candidates}
        factors = {
            code: (factor if code in candidate_codes else replace(factor, eligible=False))
            for code, factor in factors.items()
        }
        raw = v4.decide_full_pool_handoff(
            holding=holding,
            factors=factors,
            params=v4.V4_PARAMS,
            signal_hits=max(v4.V4_PARAMS.confirmation_hits, 1),
            days_since_rotation=10_000,
        )
        raw_target = raw.target if raw.triggered else None
        candidate_history.append(raw_target)
        window = max(v4.V4_PARAMS.confirmation_window, v4.V4_PARAMS.confirmation_hits, 1)
        signal_hits = sum(item == raw_target for item in candidate_history[-window:])
        if raw_target:
            signal_days += 1

        scheduled_rebalance = td in rebalance_set
        is_rebalance = scheduled_rebalance
        days_since_early = position - last_early_rotation
        held_factor = factors.get(holding) if holding else None
        if (
            scheduled_rebalance
            and target != holding
            and days_since_early < v4.V4_PARAMS.minimum_hold_days
            and held_factor is not None
            and held_factor.slow_momentum > 0.0
        ):
            target = holding
            scheduled_lock_blocks += 1

        v4_decision = v4.FullPoolDecision(False)
        first_v4_target: str | None = None
        if not scheduled_rebalance:
            v4_decision = v4.decide_full_pool_handoff(
                holding=holding,
                factors=factors,
                params=v4.V4_PARAMS,
                signal_hits=signal_hits,
                days_since_rotation=days_since_early,
            )
            if v4_decision.triggered and v4_decision.target:
                target = v4_decision.target
                is_rebalance = True
            first_v4 = v4.decide_full_pool_handoff(
                holding=holding,
                factors=factors,
                params=v4.V4_PARAMS,
                signal_hits=1,
                days_since_rotation=days_since_early,
            )
            if first_v4.triggered:
                first_v4_target = first_v4.target
        v4_target_before_stop = target

        one_day, three_day = _holding_returns(data, holding, idx_map)
        entry_position = state.get("entry_position")
        holding_age = (
            position - entry_position if holding and isinstance(entry_position, int) else None
        )
        entry_price = float(state.get("entry_price", 0.0))
        current_price = rr.price_at(data, holding, idx_map) if holding else 0.0
        entry_return = (
            current_price / entry_price - 1.0 if current_price > 0.0 and entry_price > 0.0 else 0.0
        )
        guard = EntryGuardDecision(False)
        if entry_guard_mode is not None:
            guard = entry_guard_decision(
                mode=entry_guard_mode,
                holding=holding,
                holding_age=holding_age,
                entry_return=entry_return,
                one_day=one_day,
                candidates=candidates,
                factors=factors,
                first_v4_target=first_v4_target,
                guard_days=entry_guard_days,
                day_threshold=entry_guard_day_threshold,
                entry_loss_threshold=entry_guard_entry_loss,
            )

        stop_reasons: list[str]
        shock_reasons: list[str]
        if stop_mode == "disabled":
            stop_triggered, stop_target, stop_reasons, shock_reasons = (
                False,
                None,
                [],
                [],
            )
        else:
            stop_triggered, stop_target, stop_reasons, shock_reasons = _stop_signal(
                mode=stop_mode,
                holding=holding,
                candidates=candidates,
                factors=factors,
                one_day=one_day,
                three_day=three_day,
                first_v4_target=first_v4_target,
            )
        decision_source = "holding_stop"
        stop_to_cash = stop_mode.endswith("_cash")
        failure = EntryGuardDecision(False)
        if momentum_failure_mode is not None:
            failure = momentum_failure_decision(
                mode=momentum_failure_mode,
                holding=holding,
                v4_would_exit=bool(is_rebalance and target != holding),
                one_day=one_day,
                three_day=three_day,
                candidates=candidates,
                factors=factors,
                day_threshold=momentum_failure_day_threshold,
                three_day_threshold=momentum_failure_three_day_threshold,
            )
        if failure.triggered:
            decision_source = "momentum_failure"
            stop_triggered = True
            stop_target = failure.target
            stop_reasons = list(failure.reasons)
            shock_reasons = list(failure.reasons)
            stop_to_cash = failure.exit_to_cash
        if guard.triggered:
            decision_source = "entry_guard"
            stop_triggered = True
            stop_target = guard.target
            stop_reasons = list(guard.reasons)
            shock_reasons = list(guard.reasons)
            stop_to_cash = guard.exit_to_cash
            entry_guard_trigger_count += 1
        raw_shock = bool(shock_reasons)
        raw_shock_count += int(raw_shock)
        raw_shock_1d_count += int(
            "1d<=-5%" in shock_reasons
            or "entry_1d<=-5%" in shock_reasons
            or "failure_1d" in shock_reasons
        )
        raw_shock_3d_count += int(
            "3d<=-10%" in shock_reasons
            or "entry_return<=-10%" in shock_reasons
            or "failure_3d" in shock_reasons
        )
        if stop_triggered:
            stop_trigger_count += 1
            if stop_to_cash:
                target = rq.DEFENSE
                is_rebalance = True
            elif stop_target:
                target = stop_target
                is_rebalance = True
                if stop_target == holding:
                    stop_noop_same_rank_count += 1
            else:
                stop_no_rank2_count += 1
        else:
            stop_to_cash = False

        # A cash-until-grid overlay suppresses both V3-G and V4 re-entry.
        # The lock is cleared at the next scheduled grid date above.
        if cash_lock_active:
            target = rq.DEFENSE
            is_rebalance = False

        current_value = float(state["cash"])
        if holding:
            current_value += float(state["shares"]) * rr.price_at(data, holding, idx_map)
        if current_value > float(state["peak_equity"]):
            state["peak_equity"] = current_value

        risk = ro.assess(
            target=target,
            holding=holding,
            state=state,
            data=data,
            td=td,
            idx_map=idx_map,
            is_rebalance=is_rebalance,
            common_dates=dates,
            spot_map=rr.spot_map_at(data, idx_map),
        )
        risk_events.extend(risk.events)
        target = risk.final_target or rq.DEFENSE
        if risk.action == ro.ACTION_EMERGENCY:
            is_rebalance = True

        top_up_due = bool(
            state.get("staged_entry_active")
            and target == holding
            and staged_top_up_allowed(
                holding=holding,
                holding_age=holding_age,
                confirmation_days=staged_confirmation_days,
                selected_target=daily_selected_target,
                factors=factors,
                guard_triggered=stop_triggered,
            )
        )

        old_holding = holding
        trade_executed = False
        if stop_triggered and stop_to_cash:
            cash = float(state["cash"])
            if holding:
                sell_price = rr.price_at(data, holding, idx_map)
                if sell_price > 0:
                    amount = (
                        float(state["shares"])
                        * sell_price
                        * (1.0 - (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                    )
                    cash += amount
                    trades.append(
                        {
                            "date": str(td),
                            "action": "sell",
                            "code": holding,
                            "price": sell_price,
                            "amount": amount,
                            "shares": float(state["shares"]),
                            "decision_source": "stop_to_cash",
                        }
                    )
                    state["holding"] = None
                    state["shares"] = 0.0
                    state["entry_price"] = 0.0
                    state["entry_position"] = None
                    state["staged_entry_active"] = False
                    stop_cash_exit_count += 1
                    if decision_source == "entry_guard":
                        entry_guard_cash_exit_count += 1
            state["cash"] = cash
            state["risk_exposure"] = 1.0
            next_grid = next(
                (
                    future_td
                    for future_td in trading_dates[position + 1 :]
                    if future_td in rebalance_set
                ),
                None,
            )
            state["cash_until"] = next_grid
            trade_executed = False
        elif is_rebalance and target != holding:
            cash = float(state["cash"])
            if holding:
                sell_price = rr.price_at(data, holding, idx_map)
                if sell_price > 0:
                    amount = (
                        float(state["shares"])
                        * sell_price
                        * (1.0 - (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                    )
                    cash += amount
                    trades.append(
                        {
                            "date": str(td),
                            "action": "sell",
                            "code": holding,
                            "price": sell_price,
                            "amount": amount,
                            "shares": float(state["shares"]),
                        }
                    )
                    state["holding"] = None
                    state["shares"] = 0.0
                    state["entry_price"] = 0.0
                    state["entry_position"] = None
                    state["staged_entry_active"] = False
            buy_price = rr.price_at(data, target, idx_map)
            if buy_price > 0:
                use_staged_entry = bool(
                    staged_entry_fraction is not None
                    and staged_entry_fraction < 1.0
                    and target != rq.DEFENSE
                )
                entry_fraction = staged_entry_fraction if use_staged_entry else 1.0
                shares = int(cash * risk.exposure * entry_fraction * 0.99 / buy_price / 100) * 100
                if shares > 0:
                    amount = shares * buy_price * (1.0 + (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                    cash -= amount
                    state["holding"] = target
                    state["shares"] = float(shares)
                    state["entry_price"] = buy_price
                    state["entry_position"] = position
                    state["staged_entry_active"] = use_staged_entry
                    staged_entry_count += int(use_staged_entry)
                    trades.append(
                        {
                            "date": str(td),
                            "action": "buy",
                            "code": target,
                            "price": buy_price,
                            "amount": amount,
                            "shares": float(shares),
                        }
                    )
                    trade_executed = target != old_holding
            state["cash"] = cash
            state["risk_exposure"] = 1.0
        elif top_up_due and holding:
            top_up_price = rr.price_at(data, holding, idx_map)
            cash = float(state["cash"])
            if top_up_price > 0.0:
                current_equity = cash + float(state["shares"]) * top_up_price
                desired_shares = (
                    int(current_equity * risk.exposure * 0.99 / top_up_price / 100) * 100
                )
                added_shares = max(
                    desired_shares - int(float(state["shares"])),
                    0,
                )
                amount = (
                    added_shares * top_up_price * (1.0 + (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                )
                if added_shares > 0 and amount <= cash:
                    cash -= amount
                    state["shares"] = float(state["shares"]) + added_shares
                    state["staged_entry_active"] = False
                    staged_top_up_count += 1
                    trades.append(
                        {
                            "date": str(td),
                            "action": "top_up",
                            "code": holding,
                            "price": top_up_price,
                            "amount": amount,
                            "shares": float(added_shares),
                        }
                    )
            state["cash"] = cash
            state["risk_exposure"] = 1.0
        else:
            state["risk_exposure"] = risk.exposure

        state["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
        if trade_executed and (stop_triggered or v4_decision.triggered):
            last_early_rotation = position

        stop_executed = bool(
            stop_target
            and trade_executed
            and old_holding != state["holding"]
            and state["holding"] == stop_target
        )
        stop_cash_executed = bool(
            stop_triggered and stop_to_cash and old_holding and state["holding"] is None
        )
        if stop_triggered:
            if stop_executed:
                stop_switch_count += int(stop_executed)
                if decision_source == "entry_guard":
                    entry_guard_switch_count += 1
            stop_events.append(
                {
                    "trade_date_raw": td,
                    "date": str(td),
                    "from": old_holding,
                    "to": state["holding"],
                    "intended_to": "CASH" if stop_to_cash else stop_target,
                    "risk_redirected": bool(stop_target and state["holding"] != stop_target),
                    "one_day_return": one_day,
                    "three_day_return": three_day,
                    "entry_return": entry_return,
                    "holding_age": holding_age,
                    "decision_source": decision_source,
                    "raw_shock_reasons": shock_reasons,
                    "reasons": stop_reasons,
                    "rank1": candidates[0][0] if candidates else rq.DEFENSE,
                    "rank2": stop_target,
                    "v4_target_before_stop": v4_target_before_stop,
                    "executed_switch": stop_executed,
                    "executed_cash_exit": stop_cash_executed,
                    "cash_until": str(state["cash_until"]) if stop_to_cash else None,
                }
            )

        holding = state["holding"]
        equity = float(state["cash"])
        if holding:
            equity += float(state["shares"]) * rr.price_at(data, holding, idx_map)
        equity_rows.append(
            {
                "trade_date": pd.Timestamp(td),
                "equity": equity,
                "holding": holding or rq.DEFENSE,
            }
        )

    curve = pd.DataFrame(equity_rows)
    enriched = _enrich_stop_events(stop_events, data, trading_dates, index_maps)
    metrics: dict[str, Any] = dict(rr.curve_metrics(curve, initial_capital=INITIAL_CAPITAL))
    metrics.update(
        {
            "cost_multiplier": cost_multiplier,
            "stop_mode": stop_mode,
            "trade_legs": len(trades),
            "raw_shocks": raw_shock_count,
            "raw_shock_1d": raw_shock_1d_count,
            "raw_shock_3d": raw_shock_3d_count,
            "stop_triggers": stop_trigger_count,
            "stop_switches": stop_switch_count,
            "stop_cash_exits": stop_cash_exit_count,
            "entry_guard_mode": entry_guard_mode,
            "entry_guard_triggers": entry_guard_trigger_count,
            "entry_guard_switches": entry_guard_switch_count,
            "entry_guard_cash_exits": entry_guard_cash_exit_count,
            "staged_entry_fraction": staged_entry_fraction,
            "staged_confirmation_days": staged_confirmation_days,
            "momentum_failure_mode": momentum_failure_mode,
            "momentum_failure_day_threshold": momentum_failure_day_threshold,
            "momentum_failure_three_day_threshold": (momentum_failure_three_day_threshold),
            "staged_entries": staged_entry_count,
            "staged_top_ups": staged_top_up_count,
            "stop_noop_same_rank2": stop_noop_same_rank_count,
            "stop_no_rank2": stop_no_rank2_count,
            "scheduled_lock_blocks": scheduled_lock_blocks,
            "signal_days": signal_days,
            "risk_events": len(risk_events),
        }
    )
    for horizon in (1, 2, 3, 5, 10, 20):
        values = [
            float(event[f"forward_relative_{horizon}d"])
            for event in enriched
            if f"forward_relative_{horizon}d" in event
            and (event.get("executed_switch") or event.get("executed_cash_exit"))
        ]
        metrics[f"stop_forward_relative_{horizon}d_avg"] = float(np.mean(values)) if values else 0.0
        metrics[f"stop_forward_relative_{horizon}d_count"] = len(values)
        metrics[f"stop_forward_relative_{horizon}d_win_rate"] = (
            float(np.mean(np.asarray(values) > 0)) if values else 0.0
        )
    return {
        "params": {
            "strategy": "V4",
            "stop_mode": stop_mode,
            "stop_1d": STOP_1D,
            "stop_3d": STOP_3D,
            "entry_guard_mode": entry_guard_mode,
            "entry_guard_days": entry_guard_days,
            "entry_guard_day_threshold": entry_guard_day_threshold,
            "entry_guard_entry_loss": entry_guard_entry_loss,
            "staged_entry_fraction": staged_entry_fraction,
            "staged_confirmation_days": staged_confirmation_days,
            "stop_target": (
                "cash_until_next_fixed_grid"
                if stop_mode.endswith("_cash") or momentum_failure_mode == "stale_leader_cash"
                else (
                    "qualified_alternative_or_cash_until_next_fixed_grid"
                    if momentum_failure_mode == "stale_leader_best_or_cash"
                    else "mode_defined_target"
                )
            ),
            "execution": "T-day close approximation, same as canonical V4 research",
        },
        "metrics": metrics,
        "equity_curve": curve,
        "annual": _annual_rows(curve),
        "trades": trades,
        "stop_events": enriched,
        "risk_event_log": risk_events,
        "realtime_event_log": realtime_events,
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
        "annual": result.get("annual", _annual_rows(result["equity_curve"])),
        "stop_events": result.get("stop_events", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 pre-registered stop research")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V4 research requires all downsize layers disabled")

    data = rq.load_data()
    baseline = run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=1.0)
    stops = {
        mode: run_v4_with_stop(
            data,
            cost_multiplier=1.0,
            stop_mode=mode,
        )
        for mode in STOP_MODES
    }
    cost_pressure = {
        f"{multiplier:.0f}x": {
            "baseline": run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=multiplier)[
                "metrics"
            ],
            **{
                mode: run_v4_with_stop(
                    data,
                    cost_multiplier=multiplier,
                    stop_mode=mode,
                )["metrics"]
                for mode in STOP_MODES
            },
        }
        for multiplier in (0.0, 1.0, 2.0, 3.0)
    }
    segments = {
        label: {
            "baseline": rr.segment_metrics(baseline["equity_curve"], start, end),
            **{
                mode: rr.segment_metrics(result["equity_curve"], start, end)
                for mode, result in stops.items()
            },
        }
        for label, start, end in (
            ("IS_2020_2023", "2020-06-19", "2023-12-29"),
            ("OOS_2024_2026", "2024-01-01", "2026-08-10"),
        )
    }
    payload = {
        "meta": {
            "strategy": "canonical server V4 + pre-registered holding stop overlays",
            "data_end": "2026-08-10",
            "initial_capital": INITIAL_CAPITAL,
            "v4_params": asdict(v4.V4_PARAMS),
            "stop_1d": STOP_1D,
            "stop_3d": STOP_3D,
            "pre_registered_variants": {
                "fixed_shock_rank2": ("shock <= -5%/ -10%, then V3-G filtered momentum rank #2"),
                "v3g_confirmed_shock": (
                    "shock plus held V3-G 5d momentum <=0 and slow momentum <=0; "
                    "rank #2 must have positive slow momentum and trend"
                ),
                "v4_first_confirmation_shock": (
                    "shock plus the first V4 fast/slow consensus target, "
                    "before the normal 2/2 confirmation"
                ),
                "fixed_shock_cash": (
                    "fixed shock exits to cash and suppresses re-entry until the next fixed grid"
                ),
                "v4_first_confirmation_cash": (
                    "shock plus first V4 consensus signal exits to cash; "
                    "re-entry waits for the next fixed grid"
                ),
            },
            "team_preferred_before_run": "v4_first_confirmation_shock",
            "no_parameter_scan": True,
            "no_lookahead": (
                "stop uses only current holding closes through T; "
                "forward relative returns are reporting only"
            ),
        },
        "baseline": compact(baseline),
        "variants": {mode: compact(result) for mode, result in stops.items()},
        "cost_pressure": cost_pressure,
        "segments": segments,
    }
    print(
        "\nvariant                       final       CAGR Sharpe     MDD "
        "legs stops switch cash rel5 rel10"
    )
    for label, result in [("V4", baseline), *list(stops.items())]:
        m = result["metrics"]
        print(
            f"{label:<28} {m['final_value']:>11,.0f} {m['cagr']:>7.1%} "
            f"{m['sharpe']:>6.2f} {m['max_drawdown']:>7.1%} "
            f"{m['trade_legs']:>4} {m.get('stop_triggers', 0):>5} "
            f"{m.get('stop_switches', 0):>8} "
            f"{m.get('stop_cash_exits', 0):>4} "
            f"{m.get('stop_forward_relative_5d_avg', 0.0):>5.1%} "
            f"{m.get('stop_forward_relative_10d_avg', 0.0):>6.1%}"
        )
    for mode, result in stops.items():
        print(f"\nannual {mode}")
        for row in result["annual"]:
            print(f"{row['year']}: {row['return']:>7.1%} end={row['end_value']:,.0f}")
    print("\nsegments")
    for label, rows in segments.items():
        print(label, end=" ")
        for mode, row in [("V4", rows["baseline"]), *[(m, rows[m]) for m in STOP_MODES]]:
            print(
                f"{mode}={row['final_value']:,.0f}/{row['max_drawdown']:.1%}",
                end=" ",
            )
        print()
    print("\ncost pressure")
    for label, rows in cost_pressure.items():
        print(label, end=" ")
        for mode, row in [("V4", rows["baseline"]), *[(m, rows[m]) for m in STOP_MODES]]:
            print(
                f"{mode}={row['final_value']:,.0f}/{row['max_drawdown']:.1%}",
                end=" ",
            )
        print()
    if args.save:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
