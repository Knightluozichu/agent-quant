"""V3-G relative-weakness early-rotation research.

Canonical baseline is the server path:

    run_qixing_v3.select_target -> risk_overrides.assess -> T-day 14:50 trade

The optional overlay permits a non-grid full rotation when the current holding
loses relative leadership *and* confirms absolute weakness.  It never reduces
position size.  All signal inputs are sliced through T; forward returns are
added only after the simulation as ex-post diagnostics.
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
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
except ModuleNotFoundError:
    import risk_overrides as ro
    import run_qixing_v3 as rq

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3g_relative_rotation.json"

INITIAL_CAPITAL = 100_000.0
WARMUP = 130


@dataclass(frozen=True)
class RotationParams:
    """Parameters for the proposed V3-G early-rotation overlay."""

    enabled: bool = True
    relative_gap: float = 0.02
    holding_drawdown: float = 0.04
    persistence_window: int = 3
    persistence_hits: int = 2
    fast_drawdown: float | None = 0.07
    minimum_hold_days: int = 3
    scope_assets: tuple[str, ...] = ()
    scope_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class RotationDecision:
    triggered: bool
    kind: str = ""
    reason: str = ""


def should_early_rotate(
    *,
    holding: str | None,
    leader: str | None,
    leader_score: float,
    holding_score: float,
    holding_drawdown: float,
    leader_count_last3: int,
    days_since_early_rotation: int,
    params: RotationParams,
) -> RotationDecision:
    """Pure decision rule; every input must be observable by T."""
    if not params.enabled:
        return RotationDecision(False, reason="disabled")
    if not holding or not leader or leader == holding:
        return RotationDecision(False, reason="no_rank_flip")
    if params.scope_assets and (
        holding not in params.scope_assets or leader not in params.scope_assets
    ):
        return RotationDecision(False, reason="outside_scope")
    if params.scope_groups and not any(
        holding in group and leader in group for group in params.scope_groups
    ):
        return RotationDecision(False, reason="outside_scope")
    if leader_score <= 0:
        return RotationDecision(False, reason="leader_not_positive")
    if days_since_early_rotation < params.minimum_hold_days and holding_score > 0:
        return RotationDecision(False, reason="minimum_hold")

    relative_gap = leader_score - holding_score
    if (
        params.fast_drawdown is not None
        and holding_drawdown <= -params.fast_drawdown
        and relative_gap > 0
    ):
        return RotationDecision(True, kind="fast", reason="severe_giveback")

    if relative_gap < params.relative_gap:
        return RotationDecision(False, reason="relative_gap")
    if holding_drawdown > -params.holding_drawdown:
        return RotationDecision(False, reason="drawdown")
    if leader_count_last3 < params.persistence_hits:
        return RotationDecision(False, reason="persistence")
    return RotationDecision(True, kind="standard", reason="relative_and_absolute")


def common_dates(data: dict[str, pd.DataFrame]) -> list[Any]:
    dates: set[Any] | None = None
    for code in [*list(rq.ETF_POOL), rq.DEFENSE]:
        if code not in data:
            continue
        current = set(data[code]["trade_date"].tolist())
        dates = current if dates is None else dates & current
    return sorted(dates or set())


def build_index_maps(data: dict[str, pd.DataFrame], dates: list[Any]) -> dict[Any, dict[str, int]]:
    by_code = {
        code: dict(zip(df["trade_date"].tolist(), range(len(df)), strict=False))
        for code, df in data.items()
    }
    maps: dict[Any, dict[str, int]] = {}
    for td in dates:
        maps[td] = {
            code: date_map[td]
            for code, date_map in by_code.items()
            if td in date_map and date_map[td] + 1 >= WARMUP
        }
    return maps


def close_through(data: dict[str, pd.DataFrame], code: str, idx_map: dict[str, int]) -> np.ndarray:
    return data[code]["close"].values[: idx_map[code] + 1].astype(float)


def price_at(data: dict[str, pd.DataFrame], code: str | None, idx_map: dict[str, int]) -> float:
    if not code or code not in idx_map:
        return 0.0
    return float(data[code].iloc[idx_map[code]]["close"])


def spot_map_at(
    data: dict[str, pd.DataFrame], idx_map: dict[str, int]
) -> dict[str, dict[str, float]]:
    spot: dict[str, dict[str, float]] = {}
    for code, idx in idx_map.items():
        price = float(data[code].iloc[idx]["close"])
        prev = float(data[code].iloc[idx - 1]["close"]) if idx > 0 else price
        spot[code] = {"price": price, "prev_close": prev}
    return spot


def apply_server_realtime_filter(
    data: dict[str, pd.DataFrame],
    idx_map: dict[str, int],
    candidates: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], list[dict[str, Any]]]:
    """Historical close approximation of live_signal.py's V3-G spot filter."""
    dropped: list[dict[str, Any]] = []
    for code, _score in candidates:
        close = close_through(data, code, idx_map)
        if len(close) < 2 or close[-2] <= 0:
            continue
        intraday = close[-1] / close[-2] - 1.0
        if intraday >= -0.03:
            continue
        if len(close) <= 61:
            dropped.append({"code": code, "intraday": intraday})
            continue
        ret60 = close[-1] / close[-61] - 1.0
        r10 = close[-1] / close[-11] - 1.0
        r20 = close[-1] / close[-21] - 1.0
        momentum = 0.5 * r10 + 0.5 * r20
        if ret60 >= 0.01 or momentum <= 0:
            dropped.append(
                {
                    "code": code,
                    "intraday": intraday,
                    "ret60": ret60,
                    "momentum": momentum,
                }
            )
    dropped_codes = {row["code"] for row in dropped}
    filtered = [(code, score) for code, score in candidates if code not in dropped_codes]
    return filtered, dropped


def holding_state(
    data: dict[str, pd.DataFrame],
    holding: str | None,
    idx_map: dict[str, int],
) -> tuple[float, float]:
    """Return raw V3-G momentum and drawdown from the prior five closes."""
    if not holding or holding == rq.DEFENSE or holding not in idx_map:
        return 0.0, 0.0
    close = close_through(data, holding, idx_map)
    score = float(rq.calc_momentum_score(close))
    prior = close[max(0, len(close) - 6) : -1]
    if len(prior) == 0:
        return score, 0.0
    peak_before = float(np.max(prior))
    drawdown = float(close[-1] / peak_before - 1.0) if peak_before > 0 else 0.0
    return score, drawdown


def curve_metrics(
    curve: pd.DataFrame,
    *,
    initial_capital: float | None = None,
) -> dict[str, float | str | int]:
    if curve.empty:
        return {"error": "empty"}
    equity = curve["equity"].astype(float)
    initial = float(initial_capital if initial_capital is not None else equity.iloc[0])
    total = float(equity.iloc[-1] / initial - 1.0)
    span_days = max((curve["trade_date"].iloc[-1] - curve["trade_date"].iloc[0]).days, 1)
    cagr = float((1.0 + total) ** (365.25 / span_days) - 1.0)
    rets = equity.pct_change().dropna()
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "start": str(curve["trade_date"].iloc[0].date()),
        "end": str(curve["trade_date"].iloc[-1].date()),
        "observations": len(curve),
        "initial_value": initial,
        "final_value": float(equity.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def segment_metrics(curve: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    segment = curve[(curve["trade_date"] >= start_ts) & (curve["trade_date"] <= end_ts)].copy()
    if segment.empty:
        return {"error": "empty"}
    indexed = segment.copy()
    indexed["equity"] = segment["equity"] / float(segment["equity"].iloc[0]) * INITIAL_CAPITAL
    return curve_metrics(indexed, initial_capital=INITIAL_CAPITAL)


def enrich_rotation_events(
    events: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    trading_dates: list[Any],
    index_maps: dict[Any, dict[str, int]],
) -> list[dict[str, Any]]:
    """Add explicitly ex-post five-day relative returns after simulation."""
    pos = {td: i for i, td in enumerate(trading_dates)}
    enriched: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        td = event["trade_date_raw"]
        i = pos[td]
        if i + 5 < len(trading_dates):
            future_td = trading_dates[i + 5]
            now_map = index_maps[td]
            future_map = index_maps[future_td]
            old_now = price_at(data, event["from"], now_map)
            old_future = price_at(data, event["from"], future_map)
            new_now = price_at(data, event["to"], now_map)
            new_future = price_at(data, event["to"], future_map)
            old_fwd = old_future / old_now - 1.0 if old_now > 0 else 0.0
            new_fwd = new_future / new_now - 1.0 if new_now > 0 else 0.0
            row.update(
                {
                    "ex_post_date_5d": str(future_td),
                    "ex_post_old_return_5d": old_fwd,
                    "ex_post_new_return_5d": new_fwd,
                    "ex_post_relative_5d": new_fwd - old_fwd,
                }
            )
        row.pop("trade_date_raw", None)
        enriched.append(row)
    return enriched


def run_strategy(
    data: dict[str, pd.DataFrame],
    params: RotationParams,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Replay the server V3-G path and optionally enable early rotations."""
    dates = common_dates(data)
    trading_dates = dates[WARMUP:]
    rebalance_set = set(trading_dates[:: rq.REBALANCE_DAYS])
    index_maps = build_index_maps(data, dates)

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
    leader_history: list[str] = []
    last_early_rotation = -10_000
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rotation_events: list[dict[str, Any]] = []
    risk_events: list[dict[str, Any]] = []
    realtime_events: list[dict[str, Any]] = []
    minimum_hold_blocks = 0

    for position, td in enumerate(trading_dates):
        idx_map = index_maps[td]
        holding = state["holding"]
        target, candidates, _best_score, _weak = rq.select_target(data, idx_map, holding)
        candidates, dropped = apply_server_realtime_filter(data, idx_map, candidates)
        if dropped:
            for event in dropped:
                realtime_events.append({"date": str(td), **event})
            target = candidates[0][0] if candidates else rq.DEFENSE

        leader = candidates[0][0] if candidates else rq.DEFENSE
        leader_score = float(candidates[0][1]) if candidates else 0.0
        leader_history.append(leader)
        recent_leaders = leader_history[-params.persistence_window :]
        persistence = sum(item == leader for item in recent_leaders)
        held_score, held_drawdown = holding_state(data, holding, idx_map)
        days_since_early = position - last_early_rotation

        scheduled_rebalance = td in rebalance_set
        is_rebalance = scheduled_rebalance
        early_decision = RotationDecision(False)

        # The three-day lock also protects a nearby scheduled grid from
        # immediately reversing a just-completed early rotation.
        if (
            params.enabled
            and scheduled_rebalance
            and target != holding
            and days_since_early < params.minimum_hold_days
            and held_score > 0
        ):
            target = holding
            minimum_hold_blocks += 1

        if params.enabled and not scheduled_rebalance:
            early_decision = should_early_rotate(
                holding=holding,
                leader=leader,
                leader_score=leader_score,
                holding_score=held_score,
                holding_drawdown=held_drawdown,
                leader_count_last3=persistence,
                days_since_early_rotation=days_since_early,
                params=params,
            )
            if early_decision.triggered:
                target = leader
                is_rebalance = True

        current_value = float(state["cash"])
        if holding:
            current_value += float(state["shares"]) * price_at(data, holding, idx_map)
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
            spot_map=spot_map_at(data, idx_map),
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
                sell_price = price_at(data, holding, idx_map)
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
                        }
                    )
                    state["holding"] = None
                    state["shares"] = 0.0
                    state["entry_price"] = 0.0
            buy_price = price_at(data, target, idx_map)
            if buy_price > 0:
                shares = int(cash * risk.exposure * 0.99 / buy_price / 100) * 100
                if shares > 0:
                    amount = shares * buy_price * (1.0 + (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                    cash -= amount
                    state["holding"] = target
                    state["shares"] = float(shares)
                    state["entry_price"] = buy_price
                    trades.append(
                        {
                            "date": str(td),
                            "action": "buy",
                            "code": target,
                            "price": buy_price,
                            "amount": amount,
                        }
                    )
                    trade_executed = target != old_holding
            state["cash"] = cash
            state["risk_exposure"] = 1.0
        else:
            state["risk_exposure"] = risk.exposure

        state["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None

        if early_decision.triggered and trade_executed:
            last_early_rotation = position
            rotation_events.append(
                {
                    "trade_date_raw": td,
                    "date": str(td),
                    "kind": early_decision.kind,
                    "from": old_holding,
                    "to": state["holding"],
                    "intended_to": leader,
                    "risk_redirected": state["holding"] != leader,
                    "leader_score": leader_score,
                    "holding_score": held_score,
                    "relative_gap": leader_score - held_score,
                    "holding_drawdown_5d": held_drawdown,
                    "leader_hits": persistence,
                }
            )

        holding = state["holding"]
        equity = float(state["cash"])
        if holding:
            equity += float(state["shares"]) * price_at(data, holding, idx_map)
        equity_rows.append(
            {
                "trade_date": pd.Timestamp(td),
                "equity": equity,
                "holding": holding or rq.DEFENSE,
            }
        )

    curve = pd.DataFrame(equity_rows)
    enriched = enrich_rotation_events(rotation_events, data, trading_dates, index_maps)
    relative_5d = [
        float(event["ex_post_relative_5d"]) for event in enriched if "ex_post_relative_5d" in event
    ]
    metrics = curve_metrics(curve, initial_capital=INITIAL_CAPITAL)
    metrics.update(
        {
            "cost_multiplier": cost_multiplier,
            "trade_legs": len(trades),
            "early_rotations": len(enriched),
            "standard_rotations": sum(event["kind"] == "standard" for event in enriched),
            "fast_rotations": sum(event["kind"] == "fast" for event in enriched),
            "minimum_hold_blocks": minimum_hold_blocks,
            "realtime_filter_events": len(realtime_events),
            "risk_events": len(risk_events),
            "ex_post_relative_5d_avg": float(np.mean(relative_5d)) if relative_5d else 0.0,
            "ex_post_relative_5d_win_rate": (
                float(np.mean(np.asarray(relative_5d) > 0)) if relative_5d else 0.0
            ),
        }
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


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
        "rotation_events": result["rotation_events"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-G relative rotation research")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("Current V3-G requires all downsize layers to be disabled")

    data = rq.load_data()
    cache: dict[tuple[RotationParams, float], dict[str, Any]] = {}

    def evaluate(params: RotationParams, cost: float = 1.0) -> dict[str, Any]:
        key = (params, cost)
        if key not in cache:
            cache[key] = run_strategy(data, params, cost_multiplier=cost)
        return cache[key]

    baseline_params = RotationParams(enabled=False)
    proposal_params = RotationParams()
    named_params = {
        "baseline": baseline_params,
        "rank_flip_only": RotationParams(
            relative_gap=0.0,
            holding_drawdown=0.0,
            persistence_hits=1,
            fast_drawdown=None,
            minimum_hold_days=0,
        ),
        "loose": RotationParams(relative_gap=0.01, holding_drawdown=0.03, fast_drawdown=0.06),
        "proposal": proposal_params,
        "strict": RotationParams(relative_gap=0.03, holding_drawdown=0.05, fast_drawdown=0.08),
        "proposal_no_fast": replace(proposal_params, fast_drawdown=None),
        "precious_pair_rank_flip": RotationParams(
            relative_gap=0.0,
            holding_drawdown=0.0,
            persistence_hits=1,
            fast_drawdown=None,
            scope_assets=("518880", "161226"),
        ),
        "precious_pair_loose": RotationParams(
            relative_gap=0.01,
            holding_drawdown=0.03,
            fast_drawdown=0.06,
            scope_assets=("518880", "161226"),
        ),
        "precious_pair_proposal": replace(proposal_params, scope_assets=("518880", "161226")),
        "precious_pair_strict": RotationParams(
            relative_gap=0.03,
            holding_drawdown=0.05,
            fast_drawdown=0.08,
            scope_assets=("518880", "161226"),
        ),
    }
    named = {name: evaluate(params) for name, params in named_params.items()}

    grid: dict[str, Any] = {}
    for gap in (0.01, 0.02, 0.03):
        for drawdown in (0.03, 0.04, 0.05):
            params = replace(
                proposal_params,
                relative_gap=gap,
                holding_drawdown=drawdown,
            )
            result = evaluate(params)
            grid[f"gap{gap:.0%}_dd{drawdown:.0%}"] = compact(result)

    precious_grid: dict[str, Any] = {}
    for gap in (0.0, 0.005, 0.01):
        for hits in (1, 2):
            for hold_days in (0, 3, 5):
                params = RotationParams(
                    relative_gap=gap,
                    holding_drawdown=0.0,
                    persistence_hits=hits,
                    fast_drawdown=None,
                    minimum_hold_days=hold_days,
                    scope_assets=("518880", "161226"),
                )
                key = f"gap{gap:.1%}_hits{hits}_hold{hold_days}"
                precious_grid[key] = compact(evaluate(params))

    costs: dict[str, Any] = {}
    for multiplier in (1.0, 2.0, 3.0):
        costs[f"{multiplier:.0f}x"] = {
            "baseline": compact(evaluate(baseline_params, multiplier)),
            "proposal": compact(evaluate(proposal_params, multiplier)),
            "precious_pair_rank_flip": compact(
                evaluate(named_params["precious_pair_rank_flip"], multiplier)
            ),
        }

    segments: dict[str, Any] = {}
    for label, start, end in (
        ("IS", "2020-06-19", "2023-12-29"),
        ("OOS", "2024-01-01", "2026-08-10"),
    ):
        segments[label] = {
            "baseline": segment_metrics(named["baseline"]["equity_curve"], start, end),
            "proposal": segment_metrics(named["proposal"]["equity_curve"], start, end),
            "precious_pair_rank_flip": segment_metrics(
                named["precious_pair_rank_flip"]["equity_curve"], start, end
            ),
        }

    print("\nvariant              final       CAGR  Sharpe     MDD trades early rel5d")
    for name, result in named.items():
        m = result["metrics"]
        print(
            f"{name:<20} {m['final_value']:>10,.0f} {m['cagr']:>7.1%} "
            f"{m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%} "
            f"{m['trade_legs']:>6} {m['early_rotations']:>5} "
            f"{m['ex_post_relative_5d_avg']:>6.1%}"
        )

    print("\ncost pressure")
    for label, rows in costs.items():
        b = rows["baseline"]["metrics"]
        p = rows["proposal"]["metrics"]
        pair = rows["precious_pair_rank_flip"]["metrics"]
        print(
            f"{label}: baseline={b['final_value']:,.0f}/{b['max_drawdown']:.1%} "
            f"proposal={p['final_value']:,.0f}/{p['max_drawdown']:.1%} "
            f"pair={pair['final_value']:,.0f}/{pair['max_drawdown']:.1%}"
        )

    print("\nsegments")
    for label, rows in segments.items():
        b, p = rows["baseline"], rows["proposal"]
        pair = rows["precious_pair_rank_flip"]
        print(
            f"{label}: baseline CAGR={b['cagr']:.1%} MDD={b['max_drawdown']:.1%}; "
            f"proposal CAGR={p['cagr']:.1%} MDD={p['max_drawdown']:.1%}; "
            f"pair CAGR={pair['cagr']:.1%} MDD={pair['max_drawdown']:.1%}"
        )

    payload = {
        "meta": {
            "strategy": "server V3-G; all downsize layers disabled",
            "execution": "T-day 14:50 close approximation",
            "initial_capital": INITIAL_CAPITAL,
            "signal_inputs": "T and earlier only",
            "forward_event_metrics": "ex-post diagnostics only",
        },
        "named_variants": {name: compact(result) for name, result in named.items()},
        "grid": grid,
        "precious_pair_grid": precious_grid,
        "cost_pressure": costs,
        "segments": segments,
    }
    if args.save:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
