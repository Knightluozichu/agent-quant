"""Full-pool fast/slow non-grid handoff research for server V3-G.

Slow momentum is V3-G's native 0.5*r10 + 0.5*r20 score.  Fast momentum uses
3-day and 5-day relative returns with a positive slow trend requirement.  The
overlay changes only non-grid decisions; scheduled V3-G selection and the
server risk path remain canonical.  Every signal input is observable by T.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_pair_multifactor import (
        compact as compact_pair_result,
    )
    from scripts.exp_v3g_pair_multifactor import (
        compute_pair_factors,
        enrich_pair_rotation_events,
    )
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_pair_multifactor import (
        compact as compact_pair_result,
    )
    from exp_v3g_pair_multifactor import (
        compute_pair_factors,
        enrich_pair_rotation_events,
    )

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3g_full_pool_fast_slow.json"
INITIAL_CAPITAL = 100_000.0
WARMUP = 130
FullPoolParams = v4.FullPoolParams
FullPoolDecision = v4.FullPoolDecision
decide_full_pool_handoff = v4.decide_full_pool_handoff


def run_full_pool_strategy(
    data: dict[str, pd.DataFrame],
    params: FullPoolParams,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Replay canonical V3-G with an optional full-pool daily overlay."""
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
    signal_days = 0

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
                if code in candidate_codes
                else replace(factor, eligible=False)
            )
            for code, factor in factors.items()
        }
        raw = decide_full_pool_handoff(
            holding=holding,
            factors=factors,
            params=params,
            signal_hits=max(params.confirmation_hits, 1),
            days_since_rotation=10_000,
        )
        raw_target = raw.target if raw.triggered else None
        candidate_history.append(raw_target)
        window = max(params.confirmation_window, params.confirmation_hits, 1)
        signal_hits = sum(item == raw_target for item in candidate_history[-window:])
        if raw_target:
            signal_days += 1

        scheduled_rebalance = td in rebalance_set
        is_rebalance = scheduled_rebalance
        days_since_early = position - last_early_rotation
        held_factor = factors.get(holding) if holding else None
        if (
            params.enabled
            and scheduled_rebalance
            and target != holding
            and days_since_early < params.minimum_hold_days
            and held_factor is not None
            and held_factor.slow_momentum > 0.0
        ):
            target = holding
            scheduled_lock_blocks += 1

        decision = FullPoolDecision(False)
        if params.enabled and not scheduled_rebalance:
            decision = decide_full_pool_handoff(
                holding=holding,
                factors=factors,
                params=params,
                signal_hits=signal_hits,
                days_since_rotation=days_since_early,
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
            intended = factors.get(decision.target) if decision.target else None
            rotation_events.append({
                "trade_date_raw": td,
                "date": str(td),
                "from": old_holding,
                "to": state["holding"],
                "intended_to": decision.target,
                "risk_redirected": state["holding"] != decision.target,
                "reasons": list(decision.reasons),
                "signal_hits": signal_hits,
                "slow_gap": (
                    intended.slow_momentum - held.slow_momentum
                    if held and intended else 0.0
                ),
                "fast_3d_gap": (
                    intended.return_3d - held.return_3d if held and intended else 0.0
                ),
                "fast_5d_gap": (
                    intended.return_5d - held.return_5d if held and intended else 0.0
                ),
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
    relative = {
        horizon: [
            float(event[f"ex_post_intended_relative_{horizon}d"])
            for event in enriched
            if f"ex_post_intended_relative_{horizon}d" in event
            and not event.get("risk_redirected", False)
        ]
        for horizon in (5, 10, 20)
    }
    reason_counts = Counter(
        reason for event in enriched for reason in event.get("reasons", [])
    )
    metrics = rr.curve_metrics(curve, initial_capital=INITIAL_CAPITAL)
    metrics.update({
        "cost_multiplier": cost_multiplier,
        "trade_legs": len(trades),
        "early_rotations": len(enriched),
        "signal_days": signal_days,
        "scheduled_lock_blocks": scheduled_lock_blocks,
        "realtime_filter_events": len(realtime_events),
        "risk_events": len(risk_events),
        "reason_counts": dict(reason_counts),
    })
    for horizon, values in relative.items():
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


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return compact_pair_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-G full-pool fast/slow research")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("Current V3-G requires all downsize layers disabled")

    data = rq.load_data()
    named_params = {
        "baseline": FullPoolParams.disabled(),
        "slow": FullPoolParams(mode="slow"),
        "slow_confirm2": FullPoolParams(mode="slow", confirmation_hits=2),
        "slow_gap1_confirm2": FullPoolParams(
            mode="slow", slow_gap=0.01, confirmation_hits=2
        ),
        "fast": FullPoolParams(mode="fast"),
        "fast_confirm2": FullPoolParams(mode="fast", confirmation_hits=2),
        "consensus": FullPoolParams(mode="consensus"),
        "consensus_confirm2": FullPoolParams(
            mode="consensus", confirmation_hits=2
        ),
        "consensus_strict": FullPoolParams(
            mode="consensus",
            slow_gap=0.0075,
            fast_5d_gap=0.0225,
            fast_3d_gap=0.01125,
            confirmation_hits=2,
        ),
        "consensus_very_strict": FullPoolParams(
            mode="consensus",
            slow_gap=0.01,
            fast_5d_gap=0.03,
            fast_3d_gap=0.015,
            confirmation_hits=2,
        ),
        "or": FullPoolParams(mode="or"),
    }
    cache: dict[tuple[FullPoolParams, float], dict[str, Any]] = {}

    def evaluate(params: FullPoolParams, cost: float = 1.0) -> dict[str, Any]:
        key = (params, cost)
        if key not in cache:
            cache[key] = run_full_pool_strategy(
                data, params, cost_multiplier=cost
            )
        return cache[key]

    named = {name: evaluate(params) for name, params in named_params.items()}
    costs = {
        f"{multiplier:.0f}x": {
            name: compact(evaluate(params, multiplier))
            for name, params in named_params.items()
        }
        for multiplier in (1.0, 2.0, 3.0)
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

    slow_sensitivity: dict[str, Any] = {}
    for gap in (0.005, 0.01, 0.02, 0.03, 0.05):
        for hits in (1, 2):
            params = FullPoolParams(
                mode="slow", slow_gap=gap, confirmation_hits=hits
            )
            slow_sensitivity[f"gap{gap:.1%}_hits{hits}"] = compact(
                evaluate(params)
            )

    print("\nvariant             final    CAGR Sharpe     MDD legs early rel5  rel10")
    for name, result in named.items():
        metrics = result["metrics"]
        print(
            f"{name:<18} {metrics['final_value']:>9,.0f} "
            f"{metrics['cagr']:>6.1%} {metrics['sharpe']:>6.2f} "
            f"{metrics['max_drawdown']:>7.1%} {metrics['trade_legs']:>4} "
            f"{metrics['early_rotations']:>5} "
            f"{metrics['ex_post_relative_5d_avg']:>5.1%} "
            f"{metrics['ex_post_relative_10d_avg']:>6.1%}"
        )

    print("\ncost pressure")
    for label, rows in costs.items():
        print(label, " ".join(
            f"{name}={row['metrics']['final_value']:,.0f}/{row['metrics']['max_drawdown']:.1%}"
            for name, row in rows.items()
        ))

    payload = {
        "meta": {
            "strategy": "canonical server V3-G; all downsize layers disabled",
            "execution": "T-day 14:50 close approximation",
            "initial_capital": INITIAL_CAPITAL,
            "slow": "0.5*r10+0.5*r20",
            "fast": "3d gap >=0.75pp and 5d gap >=1.5pp; target trend positive",
            "signal_inputs": "T and earlier only",
        },
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
