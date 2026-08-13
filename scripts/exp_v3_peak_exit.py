"""V3 peak-exit overlay experiment (14:50 same-day execution).

This is a research-only runner.  It keeps the production ``select_target``
logic and the same-day/live-mirror execution semantics, then adds an optional
daily high-water-mark exit:

    activate after peak profit >= activation
    exit to the defense ETF after drawdown from the holding peak >= trail
    re-entry is allowed only on the next scheduled rebalance date

The signal uses only data through T and executes at T close, which is the
project's daily approximation of the 14:30 snapshot / 14:50 execution flow.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts import run_qixing_v3 as rq
except ModuleNotFoundError:
    import run_qixing_v3 as rq

PROJECT_ROOT = Path(__file__).parent.parent

INITIAL_CAPITAL = 100_000.0
WARMUP = 130
REBALANCE_DAYS = rq.REBALANCE_DAYS


def common_dates(data: dict) -> list:
    dates: set | None = None
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        current = set(data[code]["trade_date"].tolist())
        dates = current if dates is None else dates & current
    if rq.DEFENSE in data:
        dates = (dates or set()) & set(data[rq.DEFENSE]["trade_date"].tolist())
    return sorted(dates or set())


def price_on(data: dict, code: str, td) -> float:
    rows = data[code][data[code]["trade_date"] == td]
    return float(rows.iloc[0]["close"]) if not rows.empty else 0.0


def close_through(data: dict, code: str, td) -> np.ndarray:
    rows = data[code][data[code]["trade_date"] <= td]
    return rows["close"].astype(float).to_numpy()


def tradable(data: dict, code: str, td) -> bool:
    if code not in data:
        return False
    rows = data[code][data[code]["trade_date"] == td]
    if rows.empty or float(rows.iloc[0]["close"]) <= 0:
        return False
    hist = data[code][data[code]["trade_date"] < td]
    if not hist.empty:
        prev = float(hist.iloc[-1]["close"])
        current = float(rows.iloc[0]["close"])
        if prev > 0 and abs(current / prev - 1) >= 0.099:
            return False
    return True


def target_on_rebalance(
    data: dict,
    td,
    holding: str | None,
    live_mirror: bool,
) -> tuple[str, list[tuple[str, float]]]:
    idx_map = {}
    for code in [*list(rq.ETF_POOL), rq.DEFENSE]:
        if code not in data:
            continue
        mask = data[code]["trade_date"] <= td
        if mask.sum() >= WARMUP:
            idx_map[code] = int(mask.sum()) - 1

    target, candidates, _best_score, _a_share_weak = rq.select_target(data, idx_map, holding)
    if live_mirror and candidates:
        dropped: set[str] = set()
        for code, _score in candidates:
            close = close_through(data, code, td)
            if len(close) >= 2 and close[-2] > 0 and (close[-1] - close[-2]) / close[-2] < -0.03:
                dropped.add(code)
        if dropped:
            candidates = [(code, score) for code, score in candidates if code not in dropped]
            target = candidates[0][0] if candidates else rq.DEFENSE
    return target, candidates


def trade_to(
    data: dict,
    td,
    cash: float,
    holding: str | None,
    shares: int,
    target: str,
    cost_multiplier: float,
    trade_log: list[dict],
    exposure: float = 1.0,
) -> tuple[float, str | None, int, float, float]:
    """Execute the same-day all-in trade model used by run_qixing_v3_same_day."""
    fee = rq.FEE * cost_multiplier
    slippage = rq.SLIPPAGE * cost_multiplier
    old_holding = holding
    old_shares = shares
    if holding and holding in data:
        if not tradable(data, holding, td):
            trade_log.append(
                {
                    "date": str(td),
                    "action": "sell",
                    "code": holding,
                    "status": "cancelled",
                    "reason": "not_tradable",
                }
            )
            return cash, holding, shares, 0.0, 0.0
        sell_price = price_on(data, holding, td)
        amount = shares * sell_price * (1 - fee - slippage)
        cash += amount
        trade_log.append(
            {
                "date": str(td),
                "action": "sell",
                "code": holding,
                "shares": shares,
                "price": sell_price,
                "amount": amount,
                "status": "executed",
            }
        )
        holding = None
        shares = 0

    if holding is None and target in data and tradable(data, target, td):
        buy_price = price_on(data, target, td)
        buy_shares = int(cash * max(0.0, min(exposure, 1.0)) * 0.99 / buy_price / 100) * 100
        if buy_shares > 0:
            cost = buy_shares * buy_price * (1 + fee + slippage)
            cash -= cost
            holding = target
            shares = buy_shares
            trade_log.append(
                {
                    "date": str(td),
                    "action": "buy",
                    "code": target,
                    "shares": buy_shares,
                    "price": buy_price,
                    "amount": cost,
                    "status": "executed",
                }
            )
        else:
            trade_log.append(
                {
                    "date": str(td),
                    "action": "buy",
                    "code": target,
                    "status": "cancelled",
                    "reason": "zero_shares",
                }
            )

    entry_price = price_on(data, holding, td) if holding and holding != old_holding else 0.0
    if holding == old_holding and old_shares == shares:
        entry_price = 0.0
    return cash, holding, shares, entry_price, float(old_shares)


def summarize(
    equity_curve: pd.DataFrame,
    trade_log: list[dict],
    exit_events: list[dict],
    initial_capital: float,
) -> dict:
    equity = equity_curve["equity"].astype(float)
    total = float(equity.iloc[-1] / initial_capital - 1)
    span_days = max(
        (equity_curve["trade_date"].iloc[-1] - equity_curve["trade_date"].iloc[0]).days,
        1,
    )
    cagr = float((1 + total) ** (365.25 / span_days) - 1)
    rets = equity.pct_change().dropna()
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0
    high_water = equity.cummax()
    drawdown = equity / high_water - 1
    max_dd = float(drawdown.min())
    below_10 = float((drawdown <= -0.10).mean())
    below_20 = float((drawdown <= -0.20).mean())
    sells = [x for x in trade_log if x.get("action") == "sell" and x.get("status") == "executed"]
    buys = [x for x in trade_log if x.get("action") == "buy" and x.get("status") == "executed"]
    turnover = sum(float(x.get("amount", 0.0)) for x in sells + buys)
    return {
        "start": str(equity_curve["trade_date"].iloc[0].date()),
        "end": str(equity_curve["trade_date"].iloc[-1].date()),
        "initial_capital": initial_capital,
        "final_value": float(equity.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "time_below_10pct_dd": below_10,
        "time_below_20pct_dd": below_20,
        "executed_buys": len(buys),
        "executed_sells": len(sells),
        "turnover": turnover,
        "trail_exit_count": len(exit_events),
        "trail_exit_avg_dd": (
            float(np.mean([x["dd_from_peak"] for x in exit_events])) if exit_events else 0.0
        ),
        "trail_exit_events": exit_events,
    }


def run_strategy(
    data: dict,
    *,
    activation: float = 0.05,
    trail: float | None = None,
    vol_target: float | None = None,
    cost_multiplier: float = 1.0,
    live_mirror: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    dates = common_dates(data)
    full_trading_dates = dates[WARMUP:]
    start = pd.Timestamp(start_date).date() if start_date else None
    end = pd.Timestamp(end_date).date() if end_date else None
    trading_dates = [
        td
        for td in full_trading_dates
        if (start is None or td >= start) and (end is None or td <= end)
    ]
    rebalance_dates = full_trading_dates[::REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)
    next_rebalance = dict(pairwise(rebalance_dates))

    cash = INITIAL_CAPITAL
    holding: str | None = None
    shares = 0
    entry_price = 0.0
    holding_peak = 0.0
    cooldown_until = None
    equity_rows: list[dict] = []
    trade_log: list[dict] = []
    exit_events: list[dict] = []

    for td in trading_dates:
        exited_today = False

        # Daily 14:50 overlay.  Peak is the high watermark strictly before T.
        if trail is not None and holding and holding != rq.DEFENSE and entry_price > 0:
            current = price_on(data, holding, td)
            peak_before = holding_peak if holding_peak > 0 else entry_price
            peak_gain = peak_before / entry_price - 1
            dd_from_peak = current / peak_before - 1 if peak_before > 0 else 0.0
            if peak_gain >= activation and dd_from_peak <= -trail:
                old_holding = holding
                cash, holding, shares, new_entry, _ = trade_to(
                    data,
                    td,
                    cash,
                    holding,
                    shares,
                    rq.DEFENSE,
                    cost_multiplier,
                    trade_log,
                    exposure=1.0,
                )
                if holding == rq.DEFENSE:
                    entry_price = new_entry or price_on(data, holding, td)
                    holding_peak = entry_price
                    cooldown_until = next_rebalance.get(td)
                    exit_events.append(
                        {
                            "date": str(td),
                            "code": old_holding,
                            "peak_gain": peak_gain,
                            "dd_from_peak": dd_from_peak,
                            "trail": trail,
                            "activation": activation,
                        }
                    )
                    exited_today = True
                else:
                    # If the close is not tradable, retain the old peak state.
                    holding_peak = max(holding_peak, current)

        # Re-enter/rotate only on the scheduled grid after a protective exit.
        if (
            td in rebalance_set
            and (cooldown_until is None or td >= cooldown_until)
            and not exited_today
        ):
            cooldown_until = None
            target, _candidates = target_on_rebalance(data, td, holding, live_mirror)
            if target != holding:
                exposure = 1.0
                if vol_target is not None and target in rq.ETF_POOL:
                    close = close_through(data, target, td)
                    if len(close) >= 21:
                        daily = np.diff(close[-21:]) / close[-21:-1]
                        vol20 = float(daily.std() * np.sqrt(252))
                        if vol20 > 0:
                            exposure = max(0.35, min(1.0, vol_target / vol20))
                cash, new_holding, new_shares, new_entry, _ = trade_to(
                    data,
                    td,
                    cash,
                    holding,
                    shares,
                    target,
                    cost_multiplier,
                    trade_log,
                    exposure=exposure,
                )
                if new_holding != holding:
                    holding = new_holding
                    shares = new_shares
                    if holding:
                        entry_price = new_entry or price_on(data, holding, td)
                        holding_peak = entry_price
            elif holding and holding != rq.DEFENSE:
                # No trade: the peak/entry state remains unchanged.
                pass

        if holding and holding in data:
            current = price_on(data, holding, td)
            if holding != rq.DEFENSE:
                holding_peak = max(holding_peak, current)
            equity = cash + shares * current
        else:
            equity = cash
        equity_rows.append(
            {
                "trade_date": pd.Timestamp(td),
                "equity": equity,
                "holding": holding or rq.DEFENSE,
            }
        )

    curve = pd.DataFrame(equity_rows)
    return summarize(curve, trade_log, exit_events, INITIAL_CAPITAL), curve


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 peak-exit overlay research")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save JSON report under data/v9_results",
    )
    args = parser.parse_args()

    data = rq.load_data()
    dates = common_dates(data)
    print(
        f"common_dates={len(dates)} start={dates[0]} end={dates[-1]} backtest_start={dates[WARMUP]}"
    )

    baseline_result = rq.run_qixing_v3_same_day(
        data, INITIAL_CAPITAL, cost_multiplier=1.0, live_mirror=True
    )
    base_curve = baseline_result["equity_curve"].copy()
    base_metrics = {
        "start": str(pd.Timestamp(base_curve["trade_date"].iloc[0]).date()),
        "end": str(pd.Timestamp(base_curve["trade_date"].iloc[-1]).date()),
        "initial_capital": INITIAL_CAPITAL,
        "final_value": float(base_curve["equity"].iloc[-1]),
        "total_return": float(baseline_result["total_return"]),
        "cagr": float(baseline_result["ann_return"]),
        "sharpe": float(baseline_result["sharpe"]),
        "max_drawdown": float(baseline_result["max_drawdown"]),
        "executed_buys": sum(
            1
            for x in baseline_result["trade_log"]
            if x.get("action") == "buy" and x.get("status") == "executed"
        ),
        "executed_sells": sum(
            1
            for x in baseline_result["trade_log"]
            if x.get("action") == "sell" and x.get("status") == "executed"
        ),
    }

    reports = {"baseline": base_metrics}
    for trail in (0.04, 0.06, 0.08):
        metrics, _curve = run_strategy(data, activation=0.05, trail=trail)
        reports[f"peak_exit_{trail:.0%}"] = metrics
    metrics, _curve = run_strategy(data, activation=0.05, trail=None, vol_target=0.40)
    reports["vol_target_40%"] = metrics
    metrics, _curve = run_strategy(data, activation=0.05, trail=0.06, vol_target=0.40)
    reports["vol_target_40%_plus_peak_exit_6%"] = metrics

    vol_grid = {}
    for vol_target in (0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        metrics, _curve = run_strategy(data, vol_target=vol_target)
        vol_grid[f"{vol_target:.0%}"] = metrics

    cost_pressure = {}
    for multiplier in (1.0, 2.0, 3.0):
        base = rq.run_qixing_v3_same_day(
            data,
            INITIAL_CAPITAL,
            cost_multiplier=multiplier,
            live_mirror=True,
        )
        vol30, _curve = run_strategy(
            data,
            vol_target=0.30,
            cost_multiplier=multiplier,
        )
        cost_pressure[f"{multiplier:.0f}x"] = {
            "baseline_final": float(base["equity_curve"]["equity"].iloc[-1]),
            "baseline_sharpe": float(base["sharpe"]),
            "baseline_max_drawdown": float(base["max_drawdown"]),
            "vol30_final": float(vol30["final_value"]),
            "vol30_sharpe": float(vol30["sharpe"]),
            "vol30_max_drawdown": float(vol30["max_drawdown"]),
        }

    segments = {}
    for label, start, end in (
        ("IS", "2020-06-19", "2023-12-29"),
        ("OOS", "2024-01-01", "2026-08-10"),
    ):
        segments[label] = {}
        for vol_target in (0.30, 0.35, 0.40):
            metrics, _curve = run_strategy(
                data,
                vol_target=vol_target,
                start_date=start,
                end_date=end,
            )
            segments[label][f"vol{vol_target:.0%}"] = metrics

    print("\nvariant                         final       CAGR  Sharpe     MDD  trail exits")
    for name, m in reports.items():
        print(
            f"{name:<28} {m['final_value']:>10,.0f} "
            f"{m['cagr']:>7.1%} {m['sharpe']:>7.2f} "
            f"{m['max_drawdown']:>7.1%} {m.get('trail_exit_count', 0):>11}"
        )

    if args.save:
        out = PROJECT_ROOT / "data" / "v9_results" / "v3_peak_exit.json"
        payload = {
            "meta": {
                "common_date_count": len(dates),
                "common_start": str(dates[0]),
                "common_end": str(dates[-1]),
                "backtest_start": str(dates[WARMUP]),
                "initial_capital": INITIAL_CAPITAL,
                "execution": "14:50 same-day close approximation",
            },
            "variants": reports,
            "vol_target_grid": vol_grid,
            "cost_pressure": cost_pressure,
            "segments": segments,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
