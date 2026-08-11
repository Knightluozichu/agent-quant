"""Gold/silver multifactor handoff research on the canonical server V3-G path.

The overlay is deliberately narrow: on non-grid days it may only rotate a
current gold position into silver, or a current silver position into gold.
Scheduled V3-G decisions remain untouched.  Every signal feature is sliced
through T; five-day forward returns are attached only after the replay.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3g_pair_multifactor.json"

INITIAL_CAPITAL = 100_000.0
WARMUP = 130
GOLD = "518880"
SILVER = "161226"
PAIR = (GOLD, SILVER)


PairAssetFactors = v4.AssetFactors


@dataclass(frozen=True)
class PairFactorParams:
    """Predeclared thresholds for the pair-only early-handoff overlay."""

    enabled: bool = True
    use_slow: bool = True
    use_fast: bool = False
    use_acceleration: bool = False
    use_drawdown: bool = False
    use_vol_adjusted: bool = False
    slow_gap: float = 0.005
    fast_gap: float = 0.015
    return_3d_gap: float = 0.0075
    acceleration_gap: float = 0.010
    drawdown_trigger: float = 0.030
    drawdown_fast_gap: float = 0.005
    vol_adjusted_gap: float = 0.50
    minimum_target_momentum: float = 0.0
    minimum_hold_days: int = 3
    confirmation_hits: int = 1
    confirmation_window: int = 2

    @classmethod
    def disabled(cls) -> PairFactorParams:
        return cls(enabled=False)

    @classmethod
    def slow_only(cls, **overrides: Any) -> PairFactorParams:
        return cls(**overrides)

    @classmethod
    def slow_fast(cls, **overrides: Any) -> PairFactorParams:
        return cls(use_fast=True, **overrides)

    @classmethod
    def multifactor(cls, **overrides: Any) -> PairFactorParams:
        return cls(
            use_fast=True,
            use_acceleration=True,
            use_drawdown=True,
            use_vol_adjusted=True,
            **overrides,
        )


@dataclass(frozen=True)
class PairHandoffDecision:
    triggered: bool
    target: str | None = None
    reasons: tuple[str, ...] = ()
    blocked_by: str = ""


def compute_pair_factors(
    data: dict[str, pd.DataFrame],
    idx_map: dict[str, int],
    codes: tuple[str, ...] = PAIR,
) -> dict[str, PairAssetFactors]:
    """Compute pair factors using only rows at or before ``idx_map[code]``."""
    return v4.compute_factors(data, idx_map, codes)


def decide_pair_handoff(
    *,
    holding: str | None,
    held: PairAssetFactors | None,
    peer: PairAssetFactors | None,
    params: PairFactorParams,
    signal_hits: int,
    days_since_rotation: int,
    pair: tuple[str, str] = PAIR,
) -> PairHandoffDecision:
    """Pure OR-ensemble decision; all inputs must be observable by T."""
    if not params.enabled:
        return PairHandoffDecision(False, blocked_by="disabled")
    if holding not in pair or held is None or peer is None or peer.code == holding:
        return PairHandoffDecision(False, blocked_by="outside_pair")
    if days_since_rotation < params.minimum_hold_days:
        return PairHandoffDecision(False, blocked_by="minimum_hold")
    if not peer.eligible:
        return PairHandoffDecision(False, blocked_by="gate")
    if peer.slow_momentum <= params.minimum_target_momentum:
        return PairHandoffDecision(False, blocked_by="absolute_momentum")

    slow_gap = peer.slow_momentum - held.slow_momentum
    fast_gap = peer.return_5d - held.return_5d
    r3_gap = peer.return_3d - held.return_3d
    acceleration_gap = peer.acceleration_5d - held.acceleration_5d
    vol_gap = peer.vol_adjusted_5d - held.vol_adjusted_5d
    trend_ok = peer.trend_strength > 0.0
    reasons: list[str] = []

    if params.use_slow and slow_gap >= params.slow_gap:
        reasons.append("slow")
    if (
        params.use_fast
        and fast_gap >= params.fast_gap
        and r3_gap >= params.return_3d_gap
        and trend_ok
    ):
        reasons.append("fast")
    if (
        params.use_acceleration
        and acceleration_gap >= params.acceleration_gap
        and fast_gap > 0.0
        and trend_ok
    ):
        reasons.append("acceleration")
    if (
        params.use_drawdown
        and held.drawdown_5d <= -params.drawdown_trigger
        and fast_gap >= params.drawdown_fast_gap
        and trend_ok
    ):
        reasons.append("drawdown")
    if (
        params.use_vol_adjusted
        and vol_gap >= params.vol_adjusted_gap
        and fast_gap >= params.drawdown_fast_gap
        and trend_ok
    ):
        reasons.append("vol_adjusted")

    if not reasons:
        return PairHandoffDecision(False, blocked_by="no_factor")
    if signal_hits < params.confirmation_hits:
        return PairHandoffDecision(False, blocked_by="confirmation")
    return PairHandoffDecision(True, target=peer.code, reasons=tuple(reasons))


def _candidate_signal(
    holding: str | None,
    factors: dict[str, PairAssetFactors],
    params: PairFactorParams,
    pair: tuple[str, str] = PAIR,
) -> PairHandoffDecision:
    if holding not in pair:
        return PairHandoffDecision(False, blocked_by="outside_pair")
    peer_code = pair[1] if holding == pair[0] else pair[0]
    return decide_pair_handoff(
        holding=holding,
        held=factors.get(holding),
        peer=factors.get(peer_code),
        params=params,
        signal_hits=max(params.confirmation_hits, 1),
        days_since_rotation=10_000,
        pair=pair,
    )


def _declared_pair(
    holding: str | None,
    pair_groups: tuple[tuple[str, str], ...],
) -> tuple[str, str] | None:
    return next((pair for pair in pair_groups if holding in pair), None)


def effective_signal_hits(
    *,
    observed_hits: int,
    required_hits: int,
    globally_confirmed: bool,
    allow_global_immediate: bool,
) -> int:
    """Let a full-pool Top1 confirmation bypass only the persistence check."""
    if globally_confirmed and allow_global_immediate:
        return max(observed_hits, required_hits)
    return observed_hits


def enrich_pair_rotation_events(
    events: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    trading_dates: list[Any],
    index_maps: dict[Any, dict[str, int]],
) -> list[dict[str, Any]]:
    """Separate intended pair performance from any server-risk redirection."""
    actual = rr.enrich_rotation_events(events, data, trading_dates, index_maps)
    intended_inputs = [
        {**event, "to": event.get("intended_to", event["to"])}
        for event in events
    ]
    intended = rr.enrich_rotation_events(
        intended_inputs, data, trading_dates, index_maps
    )
    date_position = {td: i for i, td in enumerate(trading_dates)}
    enriched: list[dict[str, Any]] = []
    for event, actual_row, intended_row in zip(events, actual, intended, strict=True):
        row = dict(actual_row)
        if "ex_post_relative_5d" in actual_row:
            row["ex_post_actual_relative_5d"] = actual_row["ex_post_relative_5d"]
        if "ex_post_new_return_5d" in intended_row:
            row["ex_post_intended_return_5d"] = intended_row["ex_post_new_return_5d"]
        if "ex_post_relative_5d" in intended_row:
            intended_relative = intended_row["ex_post_relative_5d"]
            row["ex_post_intended_relative_5d"] = intended_relative
            row["ex_post_relative_5d"] = intended_relative
        start_position = date_position[event["trade_date_raw"]]
        for horizon in (10, 20):
            if start_position + horizon >= len(trading_dates):
                continue
            start_date = trading_dates[start_position]
            future_date = trading_dates[start_position + horizon]
            start_map = index_maps[start_date]
            future_map = index_maps[future_date]
            old_now = rr.price_at(data, event["from"], start_map)
            old_future = rr.price_at(data, event["from"], future_map)
            intended_now = rr.price_at(data, event["intended_to"], start_map)
            intended_future = rr.price_at(data, event["intended_to"], future_map)
            actual_now = rr.price_at(data, event["to"], start_map)
            actual_future = rr.price_at(data, event["to"], future_map)
            old_return = old_future / old_now - 1.0 if old_now > 0 else 0.0
            intended_return = (
                intended_future / intended_now - 1.0 if intended_now > 0 else 0.0
            )
            actual_return = (
                actual_future / actual_now - 1.0 if actual_now > 0 else 0.0
            )
            row[f"ex_post_intended_relative_{horizon}d"] = (
                intended_return - old_return
            )
            row[f"ex_post_actual_relative_{horizon}d"] = actual_return - old_return
        enriched.append(row)
    return enriched


def run_pair_strategy(
    data: dict[str, pd.DataFrame],
    params: PairFactorParams,
    *,
    cost_multiplier: float = 1.0,
    pair_groups: tuple[tuple[str, str], ...] = (PAIR,),
    global_confirmation_immediate: bool = False,
) -> dict[str, Any]:
    """Replay server V3-G with an optional pair-only multifactor overlay."""
    dates = rr.common_dates(data)
    trading_dates = dates[WARMUP:]
    rebalance_set = set(trading_dates[:: rq.REBALANCE_DAYS])
    index_maps = rr.build_index_maps(data, dates)

    state: dict[str, Any] = {
        "cash": INITIAL_CAPITAL,
        "holding": None,
        "shares": 0.0,
        "entry_price": 0.0,
        "peak_equity": INITIAL_CAPITAL,
        "risk_exposure": 1.0,
        "cooldown_until": None,
        "h3_holding": None,
        "h3_peak": 0.0,
    }
    last_early_rotation = -10_000
    candidate_history: list[str | None] = []
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rotation_events: list[dict[str, Any]] = []
    risk_events: list[dict[str, Any]] = []
    realtime_events: list[dict[str, Any]] = []
    scheduled_lock_blocks = 0
    factor_signal_days = 0
    factor_codes = tuple(dict.fromkeys(code for pair in pair_groups for code in pair))

    for position, td in enumerate(trading_dates):
        idx_map = index_maps[td]
        holding = state["holding"]
        target, candidates, _best_score, _weak = rq.select_target(data, idx_map, holding)
        candidates, dropped = rr.apply_server_realtime_filter(data, idx_map, candidates)
        if dropped:
            realtime_events.extend({"date": str(td), **event} for event in dropped)
            target = candidates[0][0] if candidates else rq.DEFENSE

        factors = compute_pair_factors(data, idx_map, factor_codes)
        candidate_codes = {code for code, _score in candidates}
        factors = {
            code: (
                factor
                if code == rq.DEFENSE or code in candidate_codes
                else replace(factor, eligible=False)
            )
            for code, factor in factors.items()
        }
        active_pair = _declared_pair(holding, pair_groups)
        raw_signal = (
            _candidate_signal(holding, factors, params, active_pair)
            if active_pair is not None
            else PairHandoffDecision(False, blocked_by="outside_pair")
        )
        raw_target = raw_signal.target if raw_signal.triggered else None
        candidate_history.append(raw_target)
        window = max(params.confirmation_window, params.confirmation_hits, 1)
        signal_hits = sum(item == raw_target for item in candidate_history[-window:])
        globally_confirmed = bool(
            raw_target and candidates and candidates[0][0] == raw_target
        )
        decision_hits = effective_signal_hits(
            observed_hits=signal_hits,
            required_hits=params.confirmation_hits,
            globally_confirmed=globally_confirmed,
            allow_global_immediate=global_confirmation_immediate,
        )
        if raw_target:
            factor_signal_days += 1

        scheduled_rebalance = td in rebalance_set
        is_rebalance = scheduled_rebalance
        days_since_early = position - last_early_rotation
        held_factors = factors.get(holding) if holding else None

        if (
            params.enabled
            and scheduled_rebalance
            and target != holding
            and days_since_early < params.minimum_hold_days
            and held_factors is not None
            and held_factors.slow_momentum > 0.0
        ):
            target = holding
            scheduled_lock_blocks += 1

        decision = PairHandoffDecision(False)
        if params.enabled and not scheduled_rebalance and active_pair is not None:
            peer_code = active_pair[1] if holding == active_pair[0] else active_pair[0]
            decision = decide_pair_handoff(
                holding=holding,
                held=held_factors,
                peer=factors.get(peer_code),
                params=params,
                signal_hits=decision_hits,
                days_since_rotation=days_since_early,
                pair=active_pair,
            )
            if decision.triggered and decision.target:
                target = decision.target
                is_rebalance = True

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

        old_holding = holding
        trade_executed = False
        if is_rebalance and target != holding:
            cash = float(state["cash"])
            if holding:
                sell_price = rr.price_at(data, holding, idx_map)
                if sell_price > 0:
                    amount = float(state["shares"]) * sell_price * (
                        1.0 - (rq.FEE + rq.SLIPPAGE) * cost_multiplier
                    )
                    cash += amount
                    trades.append({
                        "date": str(td), "action": "sell", "code": holding,
                        "price": sell_price, "amount": amount,
                    })
                    state["holding"] = None
                    state["shares"] = 0.0
                    state["entry_price"] = 0.0
            buy_price = rr.price_at(data, target, idx_map)
            if buy_price > 0:
                shares = int(cash * risk.exposure * 0.99 / buy_price / 100) * 100
                if shares > 0:
                    amount = shares * buy_price * (
                        1.0 + (rq.FEE + rq.SLIPPAGE) * cost_multiplier
                    )
                    cash -= amount
                    state["holding"] = target
                    state["shares"] = float(shares)
                    state["entry_price"] = buy_price
                    trades.append({
                        "date": str(td), "action": "buy", "code": target,
                        "price": buy_price, "amount": amount,
                    })
                    trade_executed = target != old_holding
            state["cash"] = cash
            state["risk_exposure"] = 1.0
        else:
            state["risk_exposure"] = risk.exposure

        state["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None

        if decision.triggered and trade_executed:
            last_early_rotation = position
            held = factors.get(old_holding) if old_holding else None
            peer = factors.get(decision.target) if decision.target else None
            rotation_events.append({
                "trade_date_raw": td,
                "date": str(td),
                "from": old_holding,
                "to": state["holding"],
                "intended_to": decision.target,
                "risk_redirected": state["holding"] != decision.target,
                "reasons": list(decision.reasons),
                "signal_hits": signal_hits,
                "globally_confirmed": globally_confirmed,
                "slow_gap": (
                    peer.slow_momentum - held.slow_momentum if held and peer else 0.0
                ),
                "fast_gap": peer.return_5d - held.return_5d if held and peer else 0.0,
                "acceleration_gap": (
                    peer.acceleration_5d - held.acceleration_5d if held and peer else 0.0
                ),
                "vol_adjusted_gap": (
                    peer.vol_adjusted_5d - held.vol_adjusted_5d if held and peer else 0.0
                ),
                "holding_drawdown_5d": held.drawdown_5d if held else 0.0,
            })

        holding = state["holding"]
        equity = float(state["cash"])
        if holding:
            equity += float(state["shares"]) * rr.price_at(data, holding, idx_map)
        equity_rows.append({
            "trade_date": pd.Timestamp(td),
            "equity": equity,
            "holding": holding or rq.DEFENSE,
        })

    curve = pd.DataFrame(equity_rows)
    enriched = enrich_pair_rotation_events(
        rotation_events, data, trading_dates, index_maps
    )
    intended_relatives = {
        horizon: [
            float(event[f"ex_post_intended_relative_{horizon}d"])
            for event in enriched
            if f"ex_post_intended_relative_{horizon}d" in event
            and not event.get("risk_redirected", False)
        ]
        for horizon in (5, 10, 20)
    }
    rel5 = intended_relatives[5]
    reason_counts = Counter(
        reason for event in enriched for reason in event.get("reasons", [])
    )
    metrics = rr.curve_metrics(curve, initial_capital=INITIAL_CAPITAL)
    metrics.update({
        "cost_multiplier": cost_multiplier,
        "trade_legs": len(trades),
        "early_rotations": len(enriched),
        "global_confirmed_rotations": sum(
            bool(event.get("globally_confirmed")) for event in enriched
        ),
        "factor_signal_days": factor_signal_days,
        "scheduled_lock_blocks": scheduled_lock_blocks,
        "realtime_filter_events": len(realtime_events),
        "risk_events": len(risk_events),
        "reason_counts": dict(reason_counts),
        "ex_post_relative_5d_avg": float(np.mean(rel5)) if rel5 else 0.0,
        "ex_post_relative_5d_median": float(np.median(rel5)) if rel5 else 0.0,
        "ex_post_relative_5d_win_rate": (
            float(np.mean(np.asarray(rel5) > 0)) if rel5 else 0.0
        ),
    })
    for horizon in (10, 20):
        values = intended_relatives[horizon]
        metrics[f"ex_post_relative_{horizon}d_avg"] = (
            float(np.mean(values)) if values else 0.0
        )
        metrics[f"ex_post_relative_{horizon}d_win_rate"] = (
            float(np.mean(np.asarray(values) > 0)) if values else 0.0
        )
    return {
        "params": asdict(params),
        "metrics": metrics,
        "equity_curve": curve,
        "trades": trades,
        "rotation_events": enriched,
        "risk_event_log": risk_events,
        "realtime_event_log": realtime_events,
    }


def yearly_report(result: dict[str, Any]) -> dict[str, Any]:
    curve = result["equity_curve"].copy()
    curve["year"] = pd.to_datetime(curve["trade_date"]).dt.year
    previous = INITIAL_CAPITAL
    report: dict[str, Any] = {}
    for year, rows in curve.groupby("year", sort=True):
        end_value = float(rows["equity"].iloc[-1])
        anchored = np.concatenate(([previous], rows["equity"].astype(float).values))
        max_dd = float((anchored / np.maximum.accumulate(anchored) - 1.0).min())
        year_trades = [t for t in result["trades"] if pd.Timestamp(t["date"]).year == year]
        year_events = [
            e for e in result["rotation_events"] if pd.Timestamp(e["date"]).year == year
        ]
        rel5 = [
            float(e["ex_post_relative_5d"])
            for e in year_events
            if "ex_post_relative_5d" in e
            and not e.get("risk_redirected", False)
        ]
        report[str(year)] = {
            "start": str(pd.Timestamp(rows["trade_date"].iloc[0]).date()),
            "end": str(pd.Timestamp(rows["trade_date"].iloc[-1]).date()),
            "end_value": end_value,
            "return": end_value / previous - 1.0,
            "max_drawdown": max_dd,
            "trade_legs": len(year_trades),
            "early_rotations": len(year_events),
            "relative_5d_avg": float(np.mean(rel5)) if rel5 else 0.0,
            "relative_5d_win_rate": (
                float(np.mean(np.asarray(rel5) > 0)) if rel5 else 0.0
            ),
        }
        previous = end_value
    return report


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
        "yearly": yearly_report(result),
        "rotation_events": result["rotation_events"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-G gold/silver multifactor research")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("Current V3-G requires all downsize layers disabled")

    data = rq.load_data()
    base = PairFactorParams.multifactor()
    named_params = {
        "baseline": PairFactorParams.disabled(),
        "pair_slow": PairFactorParams.slow_only(),
        "pair_fast_only": PairFactorParams(use_slow=False, use_fast=True),
        "pair_acceleration_only": PairFactorParams(
            use_slow=False, use_acceleration=True
        ),
        "pair_drawdown_only": PairFactorParams(use_slow=False, use_drawdown=True),
        "pair_vol_adjusted_only": PairFactorParams(
            use_slow=False, use_vol_adjusted=True
        ),
        "pair_slow_fast": PairFactorParams.slow_fast(),
        "pair_slow_fast_acceleration": PairFactorParams(
            use_fast=True, use_acceleration=True
        ),
        "pair_multifactor": base,
        "pair_multifactor_confirm2": replace(base, confirmation_hits=2),
        "pair_multifactor_loose": replace(
            base,
            slow_gap=base.slow_gap * 0.75,
            fast_gap=base.fast_gap * 0.75,
            return_3d_gap=base.return_3d_gap * 0.75,
            acceleration_gap=base.acceleration_gap * 0.75,
            drawdown_trigger=base.drawdown_trigger * 0.75,
            drawdown_fast_gap=base.drawdown_fast_gap * 0.75,
            vol_adjusted_gap=base.vol_adjusted_gap * 0.75,
        ),
        "pair_multifactor_strict": replace(
            base,
            slow_gap=base.slow_gap * 1.25,
            fast_gap=base.fast_gap * 1.25,
            return_3d_gap=base.return_3d_gap * 1.25,
            acceleration_gap=base.acceleration_gap * 1.25,
            drawdown_trigger=base.drawdown_trigger * 1.25,
            drawdown_fast_gap=base.drawdown_fast_gap * 1.25,
            vol_adjusted_gap=base.vol_adjusted_gap * 1.25,
        ),
    }
    cache: dict[tuple[PairFactorParams, float], dict[str, Any]] = {}

    def evaluate(params: PairFactorParams, cost: float = 1.0) -> dict[str, Any]:
        key = (params, cost)
        if key not in cache:
            cache[key] = run_pair_strategy(data, params, cost_multiplier=cost)
        return cache[key]

    named = {name: evaluate(params) for name, params in named_params.items()}

    slow_sensitivity: dict[str, Any] = {}
    for gap in (0.0, 0.0025, 0.005, 0.0075, 0.010):
        for hits in (1, 2):
            params = PairFactorParams.slow_only(
                slow_gap=gap,
                confirmation_hits=hits,
            )
            key = f"gap{gap:.2%}_hits{hits}"
            slow_sensitivity[key] = compact(evaluate(params))

    legacy_params = rr.RotationParams(
        relative_gap=0.005,
        holding_drawdown=0.0,
        persistence_hits=1,
        fast_drawdown=None,
        minimum_hold_days=3,
        scope_assets=PAIR,
    )
    legacy = rr.run_strategy(data, legacy_params)

    costs: dict[str, Any] = {}
    for multiplier in (1.0, 2.0, 3.0):
        costs[f"{multiplier:.0f}x"] = {
            name: compact(evaluate(named_params[name], multiplier))
            for name in (
                "baseline", "pair_slow", "pair_multifactor",
                "pair_multifactor_confirm2",
            )
        }

    segments: dict[str, Any] = {}
    for label, start, end in (
        ("IS", "2020-06-19", "2023-12-29"),
        ("OOS", "2024-01-01", "2026-08-10"),
    ):
        segments[label] = {
            name: rr.segment_metrics(result["equity_curve"], start, end)
            for name, result in named.items()
        }

    print("\nvariant                         final    CAGR Sharpe     MDD legs early rel5d  win")
    for name, result in named.items():
        m = result["metrics"]
        print(
            f"{name:<31} {m['final_value']:>9,.0f} {m['cagr']:>6.1%} "
            f"{m['sharpe']:>6.2f} {m['max_drawdown']:>7.1%} "
            f"{m['trade_legs']:>4} {m['early_rotations']:>5} "
            f"{m['ex_post_relative_5d_avg']:>6.1%} "
            f"{m['ex_post_relative_5d_win_rate']:>5.0%}"
        )
    lm = legacy["metrics"]
    print(
        f"{'legacy_global_pair':<31} {lm['final_value']:>9,.0f} {lm['cagr']:>6.1%} "
        f"{lm['sharpe']:>6.2f} {lm['max_drawdown']:>7.1%} "
        f"{lm['trade_legs']:>4} {lm['early_rotations']:>5} "
        f"{lm['ex_post_relative_5d_avg']:>6.1%} "
        f"{lm['ex_post_relative_5d_win_rate']:>5.0%}"
    )

    print("\ncost pressure")
    for label, rows in costs.items():
        print(label, " ".join(
            f"{name}={row['metrics']['final_value']:,.0f}/{row['metrics']['max_drawdown']:.1%}"
            for name, row in rows.items()
        ))

    print("\nsegments")
    for label, rows in segments.items():
        print(label, " ".join(
            f"{name}={row['cagr']:.1%}/{row['max_drawdown']:.1%}"
            for name, row in rows.items()
        ))

    payload = {
        "meta": {
            "strategy": "canonical server V3-G; all downsize layers disabled",
            "execution": "T-day 14:50 close approximation",
            "initial_capital": INITIAL_CAPITAL,
            "pair": list(PAIR),
            "signal_inputs": "T and earlier only",
            "forward_event_metrics": "ex-post diagnostics only",
            "threshold_policy": "predeclared factor ablation plus uniform +/-25%",
        },
        "legacy_global_pair": rr.compact(legacy),
        "named_variants": {name: compact(result) for name, result in named.items()},
        "slow_sensitivity": slow_sensitivity,
        "cost_pressure": costs,
        "segments": segments,
    }
    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
