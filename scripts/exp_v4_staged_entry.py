"""Research-only replay of a two-stage entry overlay for canonical V4.

The overlay changes position sizing, not V4 ranking or handoff logic.  A new
risk-asset position opens at 50% or 60% of the normal exposure.  It can be
topped up after two trading days only while the holding remains the daily
V3-G target and its eligibility, slow momentum, and short trend stay positive.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from scripts.exp_v4_stop import run_v4_with_stop
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v4_stop import run_v4_with_stop


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_staged_entry_research.json"
INITIAL_CAPITAL = 100_000.0
CONFIRMATION_DAYS = 2
VARIANTS: dict[str, tuple[float, str | None]] = {
    "stage50": (0.50, None),
    "stage60": (0.60, None),
    "stage50_selective": (0.50, "entry_guard_selective"),
    "stage60_selective": (0.60, "entry_guard_selective"),
}


def _trade_tail(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure realized round trips, including any top-up cash flow."""
    entry_code: str | None = None
    entry_cost = 0.0
    rows: list[dict[str, Any]] = []
    for trade in trades:
        action = str(trade.get("action"))
        code = str(trade.get("code"))
        amount = float(trade.get("amount", 0.0))
        if action == "buy":
            entry_code = code
            entry_cost = amount
        elif action == "top_up" and entry_code == code:
            entry_cost += amount
        elif action == "sell" and entry_code == code and entry_cost > 0.0:
            rows.append(
                {
                    "exit_date": str(trade.get("date")),
                    "code": code,
                    "net_return": amount / entry_cost - 1.0,
                }
            )
            entry_code = None
            entry_cost = 0.0

    values = np.asarray([row["net_return"] for row in rows], dtype=float)
    return {
        "round_trips": len(rows),
        "worst_trade_return": float(np.min(values)) if len(values) else 0.0,
        "p05_trade_return": float(np.quantile(values, 0.05)) if len(values) else 0.0,
        "losses_below_10pct": int(np.sum(values <= -0.10)) if len(values) else 0,
        "trade_returns": rows,
    }


def _evaluate(
    data: dict[str, Any],
    *,
    fraction: float | None,
    guard_mode: str | None,
    cost_multiplier: float,
) -> dict[str, Any]:
    result = run_v4_with_stop(
        data,
        cost_multiplier=cost_multiplier,
        stop_mode="disabled",
        entry_guard_mode=guard_mode,
        staged_entry_fraction=fraction,
        staged_confirmation_days=CONFIRMATION_DAYS,
    )
    tail = _trade_tail(result["trades"])
    result["trade_tail"] = tail
    result["metrics"].update({key: value for key, value in tail.items() if key != "trade_returns"})
    return result


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
        "annual": result["annual"],
        "trade_tail": result["trade_tail"],
        "entry_guard_events": [
            event
            for event in result["stop_events"]
            if event.get("decision_source") == "entry_guard"
        ],
    }


def _assert_baseline_parity(canonical: dict[str, Any], replay: dict[str, Any]) -> None:
    left = canonical["equity_curve"]
    right = replay["equity_curve"]
    if left["trade_date"].tolist() != right["trade_date"].tolist():
        raise RuntimeError("staged-entry baseline date drift")
    if not np.allclose(left["equity"], right["equity"]):
        raise RuntimeError("staged-entry baseline equity drift")
    if left["holding"].tolist() != right["holding"].tolist():
        raise RuntimeError("staged-entry baseline holding drift")


def _stress_scenarios() -> dict[str, Any]:
    """Deterministic arithmetic for the user's silver/gold path."""
    silver_terminal = (1.0 - 0.05) * (1.0 - 0.08) * (1.0 - 0.03)
    gold_after_switch = (1.0 + 0.02) * (1.0 + 0.03)
    silver_through_day2 = (1.0 - 0.05) * (1.0 - 0.08)
    rows: dict[str, Any] = {}
    for fraction in (0.50, 0.60):
        after_day1 = 1.0 - fraction * 0.05
        switch_terminal = after_day1 * ((1.0 - fraction) + fraction * gold_after_switch)
        rows[f"stage{int(fraction * 100)}"] = {
            "hold_silver_without_guard": ((1.0 - fraction) + fraction * silver_terminal - 1.0),
            "qualified_gold_switch_day1": switch_terminal - 1.0,
            "no_qualified_replacement_cash_day2": (
                (1.0 - fraction) + fraction * silver_through_day2 - 1.0
            ),
        }
    return {
        "path": {
            "silver_daily": [-0.05, -0.08, -0.03],
            "gold_daily": [0.01, 0.02, 0.03],
        },
        "full_position_v4_hold": silver_terminal - 1.0,
        "variants": rows,
        "assumption": (
            "a qualified day-1 switch reopens the replacement at the same staged "
            "fraction; an unqualified path exits after cumulative loss breaches -10%"
        ),
        "scope": "mechanism stress only, not historical performance evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 two-stage entry research")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V4 research requires all downsize layers disabled")

    data = rq.load_data()
    cache: dict[tuple[float | None, str | None, float], dict[str, Any]] = {}

    def evaluate(
        fraction: float | None,
        guard_mode: str | None,
        cost: float = 1.0,
    ) -> dict[str, Any]:
        key = (fraction, guard_mode, cost)
        if key not in cache:
            cache[key] = _evaluate(
                data,
                fraction=fraction,
                guard_mode=guard_mode,
                cost_multiplier=cost,
            )
        return cache[key]

    baseline = evaluate(None, None)
    canonical = run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=1.0)
    _assert_baseline_parity(canonical, baseline)
    variants = {
        label: evaluate(fraction, guard_mode) for label, (fraction, guard_mode) in VARIANTS.items()
    }
    dates = rr.common_dates(data)
    data_end = str(dates[-1])

    segments = {
        label: {
            "baseline": rr.segment_metrics(baseline["equity_curve"], start, end),
            **{
                name: rr.segment_metrics(result["equity_curve"], start, end)
                for name, result in variants.items()
            },
        }
        for label, start, end in (
            ("IS_2020_2023", "2020-06-19", "2023-12-29"),
            ("OOS_2024_2026", "2024-01-01", data_end),
        )
    }
    cost_pressure: dict[str, Any] = {}
    for cost in (0.0, 1.0, 2.0, 3.0):
        replay_at_cost = evaluate(None, None, cost)
        canonical_at_cost = run_full_pool_strategy(
            data,
            v4.V4_PARAMS,
            cost_multiplier=cost,
        )
        _assert_baseline_parity(canonical_at_cost, replay_at_cost)
        cost_pressure[f"{cost:.0f}x"] = {
            "baseline": replay_at_cost["metrics"],
            **{
                label: evaluate(fraction, guard_mode, cost)["metrics"]
                for label, (fraction, guard_mode) in VARIANTS.items()
            },
        }

    payload = {
        "meta": {
            "strategy": "canonical server V4 + research-only staged entry",
            "data_end": data_end,
            "initial_capital": INITIAL_CAPITAL,
            "v4_params": asdict(v4.V4_PARAMS),
            "confirmation_days": CONFIRMATION_DAYS,
            "pre_registered_variants": {
                label: {
                    "initial_fraction": fraction,
                    "entry_guard": guard_mode,
                }
                for label, (fraction, guard_mode) in VARIANTS.items()
            },
            "top_up_rule": (
                "from age 2 onward: still the daily V3-G target, eligible, "
                "slow momentum > 0, short trend > 0, and no guard trigger"
            ),
            "no_lookahead": "all decisions use prices and factors through T only",
            "no_parameter_scan": True,
            "baseline_parity": "canonical V4 verified at 0x/1x/2x/3x costs",
        },
        "baseline": _compact(baseline),
        "variants": {label: _compact(result) for label, result in variants.items()},
        "segments": segments,
        "cost_pressure": cost_pressure,
        "stress_scenarios": _stress_scenarios(),
    }

    print(
        "\nvariant              final       CAGR Sharpe     MDD legs "
        "staged topup guard worst    p05"
    )
    for label, result in [("V4", baseline), *variants.items()]:
        metrics = result["metrics"]
        print(
            f"{label:<20} {metrics['final_value']:>11,.0f} "
            f"{metrics['cagr']:>7.1%} {metrics['sharpe']:>6.2f} "
            f"{metrics['max_drawdown']:>7.1%} {metrics['trade_legs']:>4} "
            f"{metrics.get('staged_entries', 0):>6} "
            f"{metrics.get('staged_top_ups', 0):>5} "
            f"{metrics.get('entry_guard_triggers', 0):>5} "
            f"{metrics['worst_trade_return']:>6.1%} "
            f"{metrics['p05_trade_return']:>6.1%}"
        )

    print("\nsegments")
    for label, rows in segments.items():
        print(label, end=" ")
        for name, row in [
            ("V4", rows["baseline"]),
            *[(variant, rows[variant]) for variant in VARIANTS],
        ]:
            print(
                f"{name}={row['final_value']:,.0f}/{row['max_drawdown']:.1%}",
                end=" ",
            )
        print()

    print("\ncost pressure")
    for label, rows in cost_pressure.items():
        print(label, end=" ")
        for name, row in [
            ("V4", rows["baseline"]),
            *[(variant, rows[variant]) for variant in VARIANTS],
        ]:
            print(
                f"{name}={row['final_value']:,.0f}/{row['max_drawdown']:.1%}",
                end=" ",
            )
        print()

    print("\nuser stress")
    print(json.dumps(payload["stress_scenarios"], ensure_ascii=False, indent=2))

    if args.save:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
