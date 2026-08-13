"""Pre-registered V4 post-entry failure guard research.

The production V4 core is unchanged.  This experiment tests only the first
three trading days after a new entry and reuses the already registered shock
levels: one-day return <= -5% or return since entry <= -10%.
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
    from scripts.exp_v4_stop import (
        ENTRY_GUARD_DAYS,
        ENTRY_GUARD_ENTRY_LOSS,
        ENTRY_GUARD_MODES,
        STOP_1D,
        entry_guard_decision,
        run_v4_with_stop,
    )
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v4_stop import (
        ENTRY_GUARD_DAYS,
        ENTRY_GUARD_ENTRY_LOSS,
        ENTRY_GUARD_MODES,
        STOP_1D,
        entry_guard_decision,
        run_v4_with_stop,
    )


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_entry_guard_research.json"
INITIAL_CAPITAL = 100_000.0


def _factor(
    code: str,
    *,
    slow: float,
    ret3: float,
    ret5: float,
    trend: float,
) -> v4.AssetFactors:
    return v4.AssetFactors(
        code=code,
        eligible=True,
        slow_momentum=slow,
        return_3d=ret3,
        return_5d=ret5,
        acceleration_5d=0.0,
        trend_strength=trend,
        vol_adjusted_5d=0.0,
        drawdown_5d=0.0,
    )


def _stress_scenarios() -> dict[str, Any]:
    """Replay the user-specified shock path through the pure guard decision."""
    factors = {
        "silver": _factor("silver", slow=0.14, ret3=-0.05, ret5=-0.05, trend=-0.01),
        "gold": _factor("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=0.01),
    }
    day1 = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=1,
        entry_return=-0.05,
        one_day=-0.05,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors=factors,
        first_v4_target=None,
    )
    no_replacement = dict(factors)
    no_replacement["gold"] = _factor(
        "gold", slow=0.06, ret3=0.01, ret5=0.02, trend=-0.01
    )
    day1_without_replacement = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=1,
        entry_return=-0.05,
        one_day=-0.05,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors=no_replacement,
        first_v4_target=None,
    )
    day2_without_replacement = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=2,
        entry_return=0.95 * 0.92 - 1.0,
        one_day=-0.08,
        candidates=[("silver", 0.08), ("gold", 0.07)],
        factors=no_replacement,
        first_v4_target=None,
    )
    baseline_terminal = 0.95 * 0.92 * 0.97 - 1.0
    switch_terminal = 0.95 * 1.02 * 1.03 - 1.0
    cash_day2_terminal = 0.95 * 0.92 - 1.0
    return {
        "path": {
            "holding_daily": [-0.05, -0.08, -0.03],
            "alternative_daily": [0.01, 0.02, 0.03],
        },
        "baseline_terminal_return": baseline_terminal,
        "qualified_replacement": {
            "day1_triggered": day1.triggered,
            "day1_target": day1.target,
            "terminal_return_after_day1_switch": switch_terminal,
            "improvement_vs_baseline": switch_terminal - baseline_terminal,
        },
        "no_qualified_replacement": {
            "day1_triggered": day1_without_replacement.triggered,
            "day2_triggered": day2_without_replacement.triggered,
            "day2_exit_to_cash": day2_without_replacement.exit_to_cash,
            "terminal_return_after_day2_cash": cash_day2_terminal,
            "improvement_vs_baseline": cash_day2_terminal - baseline_terminal,
        },
        "scope": "deterministic mechanism stress, not historical performance evidence",
    }


def _entry_path_diagnostics(
    trades: list[dict[str, Any]], data: dict[str, Any]
) -> dict[str, Any]:
    """Describe first-three-day entry paths for reporting only."""
    entries: list[dict[str, Any]] = []
    open_entry: dict[str, Any] | None = None
    for trade in trades:
        if trade.get("action") == "buy":
            open_entry = {
                "code": str(trade["code"]),
                "date": str(trade["date"]),
                "price": float(trade["price"]),
                "exit_date": None,
            }
            entries.append(open_entry)
        elif (
            trade.get("action") == "sell"
            and open_entry
            and str(trade.get("code")) == open_entry["code"]
        ):
            open_entry["exit_date"] = str(trade["date"])
            open_entry = None

    dates = rr.common_dates(data)
    date_strings = [str(day) for day in dates]
    positions = {day: index for index, day in enumerate(date_strings)}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        start = positions.get(entry["date"])
        code = entry["code"]
        if start is None or code not in data:
            continue
        prices = {
            str(day): float(close)
            for day, close in zip(
                data[code]["trade_date"].tolist(),
                data[code]["close"].tolist(),
                strict=False,
            )
        }
        for age in range(1, ENTRY_GUARD_DAYS + 1):
            if start + age >= len(date_strings):
                continue
            day = date_strings[start + age]
            if entry["exit_date"] is not None and day > entry["exit_date"]:
                break
            price = prices.get(day)
            if price is None or entry["price"] <= 0.0:
                continue
            rows.append({
                "entry_date": entry["date"],
                "date": day,
                "code": code,
                "age": age,
                "return_since_entry": price / entry["price"] - 1.0,
            })

    by_age: dict[str, Any] = {}
    for age in range(1, ENTRY_GUARD_DAYS + 1):
        values = np.asarray(
            [row["return_since_entry"] for row in rows if row["age"] == age],
            dtype=float,
        )
        by_age[str(age)] = {
            "observations": len(values),
            "minimum": float(np.min(values)) if len(values) else 0.0,
            "p05": float(np.quantile(values, 0.05)) if len(values) else 0.0,
            "at_or_below_5pct": int(np.sum(values <= -0.05)) if len(values) else 0,
            "at_or_below_10pct": int(np.sum(values <= -0.10)) if len(values) else 0,
        }
    return {"entries": len(entries), "by_age": by_age, "observations": rows}


def _trade_tail(trades: list[dict[str, Any]], cost_multiplier: float) -> dict[str, Any]:
    """Summarize realized round-trip returns without using future decision data."""
    entry: tuple[str, float] | None = None
    rows: list[dict[str, Any]] = []
    unit_cost = (rq.FEE + rq.SLIPPAGE) * cost_multiplier
    for trade in trades:
        action = trade.get("action")
        code = str(trade.get("code"))
        price = float(trade.get("price", 0.0))
        if action == "buy" and price > 0.0:
            entry = (code, price)
        elif action == "sell" and entry and entry[0] == code and price > 0.0:
            gross_return = price / entry[1] - 1.0
            net_return = price * (1.0 - unit_cost) / (
                entry[1] * (1.0 + unit_cost)
            ) - 1.0
            rows.append({
                "exit_date": str(trade.get("date")),
                "code": code,
                "gross_return": gross_return,
                "net_return": net_return,
            })
            entry = None

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
    mode: str | None,
    cost_multiplier: float,
    *,
    guard_days: int = ENTRY_GUARD_DAYS,
    day_threshold: float = STOP_1D,
    entry_loss_threshold: float = ENTRY_GUARD_ENTRY_LOSS,
) -> dict[str, Any]:
    result = run_v4_with_stop(
        data,
        cost_multiplier=cost_multiplier,
        stop_mode="disabled",
        entry_guard_mode=mode,
        entry_guard_days=guard_days,
        entry_guard_day_threshold=day_threshold,
        entry_guard_entry_loss=entry_loss_threshold,
    )
    result["trade_tail"] = _trade_tail(result["trades"], cost_multiplier)
    result["metrics"].update({
        key: value
        for key, value in result["trade_tail"].items()
        if key != "trade_returns"
    })
    return result


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
        "annual": result["annual"],
        "stop_events": result["stop_events"],
        "trade_tail": result["trade_tail"],
    }


def _assert_baseline_parity(
    canonical: dict[str, Any], replay: dict[str, Any]
) -> None:
    left = canonical["equity_curve"]
    right = replay["equity_curve"]
    if left["trade_date"].tolist() != right["trade_date"].tolist():
        raise RuntimeError("entry-guard baseline date drift")
    if not np.allclose(left["equity"], right["equity"]):
        raise RuntimeError("entry-guard baseline equity drift")
    if left["holding"].tolist() != right["holding"].tolist():
        raise RuntimeError("entry-guard baseline holding drift")


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 post-entry failure guard research")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("V4 research requires all downsize layers disabled")

    data = rq.load_data()
    cache: dict[tuple[str | None, float, int, float, float], dict[str, Any]] = {}

    def evaluate(
        mode: str | None,
        cost: float = 1.0,
        *,
        guard_days: int = ENTRY_GUARD_DAYS,
        day_threshold: float = STOP_1D,
        entry_loss_threshold: float = ENTRY_GUARD_ENTRY_LOSS,
    ) -> dict[str, Any]:
        key = (mode, cost, guard_days, day_threshold, entry_loss_threshold)
        if key not in cache:
            cache[key] = _evaluate(
                data,
                mode,
                cost,
                guard_days=guard_days,
                day_threshold=day_threshold,
                entry_loss_threshold=entry_loss_threshold,
            )
        return cache[key]

    baseline = evaluate(None)
    canonical = run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=1.0)
    _assert_baseline_parity(canonical, baseline)
    variants = {mode: evaluate(mode) for mode in ENTRY_GUARD_MODES}
    data_end = str(rr.common_dates(data)[-1])

    cost_pressure = {
        f"{multiplier:.0f}x": {
            "baseline": evaluate(None, multiplier)["metrics"],
            **{
                mode: evaluate(mode, multiplier)["metrics"]
                for mode in ENTRY_GUARD_MODES
            },
        }
        for multiplier in (0.0, 1.0, 2.0, 3.0)
    }
    segments = {
        label: {
            "baseline": rr.segment_metrics(baseline["equity_curve"], start, end),
            **{
                mode: rr.segment_metrics(result["equity_curve"], start, end)
                for mode, result in variants.items()
            },
        }
        for label, start, end in (
            ("IS_2020_2023", "2020-06-19", "2023-12-29"),
            ("OOS_2024_2026", "2024-01-01", "2026-08-10"),
        )
    }
    sensitivity_specs: dict[str, tuple[int, float, float]] = {
        "days_2": (2, STOP_1D, ENTRY_GUARD_ENTRY_LOSS),
        "days_4": (4, STOP_1D, ENTRY_GUARD_ENTRY_LOSS),
        "day_threshold_-4pct": (ENTRY_GUARD_DAYS, -0.04, ENTRY_GUARD_ENTRY_LOSS),
        "day_threshold_-6pct": (ENTRY_GUARD_DAYS, -0.06, ENTRY_GUARD_ENTRY_LOSS),
        "entry_loss_-8pct": (ENTRY_GUARD_DAYS, STOP_1D, -0.08),
        "entry_loss_-12pct": (ENTRY_GUARD_DAYS, STOP_1D, -0.12),
    }
    sensitivity = {
        label: _compact(
            evaluate(
                "entry_guard_selective",
                guard_days=spec[0],
                day_threshold=spec[1],
                entry_loss_threshold=spec[2],
            )
        )
        for label, spec in sensitivity_specs.items()
    }
    stress_scenarios = _stress_scenarios()
    entry_path_diagnostics = _entry_path_diagnostics(baseline["trades"], data)

    payload = {
        "meta": {
            "strategy": "canonical server V4 + post-entry failure guard",
            "data_end": data_end,
            "initial_capital": INITIAL_CAPITAL,
            "v4_params": asdict(v4.V4_PARAMS),
            "guard_days": ENTRY_GUARD_DAYS,
            "shock_1d": STOP_1D,
            "entry_loss": ENTRY_GUARD_ENTRY_LOSS,
            "pre_registered_variants": {
                "entry_guard_cash": (
                    "new holding shock exits to cash until the next fixed grid"
                ),
                "entry_guard_v4_or_cash": (
                    "new holding shock switches on first full V4 consensus; "
                    "otherwise exits to cash until the next fixed grid"
                ),
                "entry_guard_best_or_cash": (
                    "new holding shock switches to the strongest eligible positive-trend "
                    "alternative that beats the holding on both 3d and 5d return; "
                    "otherwise exits to cash"
                ),
                "entry_guard_selective": (
                    "a one-day shock switches only when a positive relative replacement "
                    "exists; otherwise it holds, while a <= -10% post-entry loss exits "
                    "to cash until the next fixed grid"
                ),
            },
            "no_parameter_scan": True,
            "no_lookahead": (
                "guard uses entry price, holding age, and closes/factors through T only; "
                "forward returns are reporting only"
            ),
        },
        "baseline": _compact(baseline),
        "variants": {mode: _compact(result) for mode, result in variants.items()},
        "cost_pressure": cost_pressure,
        "segments": segments,
        "sensitivity": sensitivity,
        "stress_scenarios": stress_scenarios,
        "entry_path_diagnostics": entry_path_diagnostics,
    }

    print(
        "\nvariant                       final       CAGR Sharpe     MDD "
        "legs guard switch cash worst    p05"
    )
    for label, result in [("V4", baseline), *list(variants.items())]:
        m = result["metrics"]
        print(
            f"{label:<28} {m['final_value']:>11,.0f} {m['cagr']:>7.1%} "
            f"{m['sharpe']:>6.2f} {m['max_drawdown']:>7.1%} "
            f"{m['trade_legs']:>4} {m.get('entry_guard_triggers', 0):>5} "
            f"{m.get('entry_guard_switches', 0):>6} "
            f"{m.get('entry_guard_cash_exits', 0):>4} "
            f"{m['worst_trade_return']:>6.1%} {m['p05_trade_return']:>6.1%}"
        )

    print("\nsegments")
    for label, rows in segments.items():
        print(label, end=" ")
        for mode, row in [
            ("V4", rows["baseline"]),
            *[(name, rows[name]) for name in ENTRY_GUARD_MODES],
        ]:
            print(f"{mode}={row['final_value']:,.0f}/{row['max_drawdown']:.1%}", end=" ")
        print()

    print("\ncost pressure")
    for label, rows in cost_pressure.items():
        print(label, end=" ")
        for mode, row in [
            ("V4", rows["baseline"]),
            *[(name, rows[name]) for name in ENTRY_GUARD_MODES],
        ]:
            print(f"{mode}={row['final_value']:,.0f}/{row['max_drawdown']:.1%}", end=" ")
        print()

    print("\nselective sensitivity")
    for label, result in sensitivity.items():
        m = result["metrics"]
        print(
            f"{label:<24} final={m['final_value']:>11,.0f} "
            f"Sharpe={m['sharpe']:.2f} MDD={m['max_drawdown']:.1%} "
            f"guard={m.get('entry_guard_triggers', 0)}"
        )

    print("\nentry path diagnostics")
    for age, row in entry_path_diagnostics["by_age"].items():
        print(
            f"day {age}: n={row['observations']} min={row['minimum']:.1%} "
            f"p05={row['p05']:.1%} <=-5%={row['at_or_below_5pct']} "
            f"<=-10%={row['at_or_below_10pct']}"
        )
    print("\nuser stress")
    print(json.dumps(stress_scenarios, ensure_ascii=False, indent=2))

    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
