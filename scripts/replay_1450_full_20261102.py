"""Full strict 14:50 replay for V3-G/V4/V5 on purchased 5-minute data.

Signal and fill both use the 14:50 price. Prior days use full daily bars;
the current day uses a partial bar through 14:50 only.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import live_signal as live
import numpy as np
import pandas as pd
import qixing_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/etf_5min_daily"
OUT = ROOT / "artifacts/replay_1450_full_20261102"
CODES = [
    "518880.SH",
    "159985.SZ",
    "501018.SH",
    "161226.SZ",
    "513100.SH",
    "159915.SZ",
    "511220.SH",
    "511880.SH",
]
INITIAL = 10_000.0
BUY_COST = 0.0015
SELL_COST = 0.0020


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for code in CODES:
        df = (
            pd.read_parquet(DATA / f"{code}.parquet")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        df["trade_date"] = pd.to_datetime(df.trade_date).dt.date
        out[code.split(".")[0]] = df
    return out


def metrics(curve: list[dict]) -> dict:
    frame = pd.DataFrame(curve)
    eq = frame.equity.to_numpy(dtype=float)
    path = np.r_[INITIAL, eq]
    years = (
        pd.to_datetime(frame.trade_date.iloc[-1]) - pd.to_datetime(frame.trade_date.iloc[0])
    ).days / 365.25
    rets = np.diff(path) / path[:-1]
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    monthly = (
        frame.set_index(pd.to_datetime(frame.trade_date))
        .equity.resample("ME")
        .last()
        .pct_change()
        .dropna()
    )
    return {
        "final_equity": round(float(eq[-1]), 2),
        "total_return_pct": round(float(eq[-1] / INITIAL - 1) * 100, 2),
        "cagr_pct": round(float((eq[-1] / INITIAL) ** (1 / years) - 1) * 100, 2) if years else 0.0,
        "sharpe": round(float(rets.mean() / vol * np.sqrt(252)), 2) if vol else 0.0,
        "max_drawdown_pct": round(float((path / np.maximum.accumulate(path) - 1).min()) * 100, 2),
        "years": round(years, 2),
        "months": len(monthly),
        "months_ge_10pct": int((monthly >= 0.10).sum()),
        "months_ge_20pct": int((monthly >= 0.20).sum()),
        "best_month_pct": round(float(monthly.max()) * 100, 2) if len(monthly) else 0.0,
        "worst_month_pct": round(float(monthly.min()) * 100, 2) if len(monthly) else 0.0,
        "trades": int(frame.trades.iloc[-1]) if "trades" in frame else 0,
        "holding": frame.holding.iloc[-1] if isinstance(frame.holding.iloc[-1], str) else None,
        "start": str(frame.trade_date.iloc[0]),
        "end": str(frame.trade_date.iloc[-1]),
    }


def run(
    mode: str,
    data: dict[str, pd.DataFrame],
    dates: list[date],
    grid: set[date],
    cost_mult: float = 1.0,
) -> dict:
    cash = INITIAL
    holding = None
    shares = 0
    entry_price = 0.0
    state = live.default_state(INITIAL)
    state.update({"cash": cash, "holding": None, "shares": 0, "entry_price": 0.0})
    curve = []
    trades = 0
    codes = list(data)
    for i, td in enumerate(dates):
        today = {}
        for code in codes:
            row = data[code][data[code].trade_date == td]
            if row.empty or not np.isfinite(row.px1450.iloc[0]):
                today = {}
                break
            today[code] = row.iloc[0]
        if not today:
            continue
        hist = {}
        idx_map = {}
        for code in codes:
            base = data[code][data[code].trade_date < td].copy()
            r = today[code]
            base = pd.concat(
                [
                    base,
                    pd.DataFrame(
                        [
                            {
                                "trade_date": td,
                                "open": float(r.open1450),
                                "high": float(r.high1450),
                                "low": float(r.low1450),
                                "close": float(r.px1450),
                                "volume": float(r.vol1450),
                                "symbol": code,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            hist[code] = base
            idx_map[code] = len(base) - 1
        target, candidates, _best, _weak = live.select_target(hist, idx_map, holding)
        scheduled = td in grid
        if mode == "V3-G":
            final_target = target
            rebalance = scheduled and target != holding
        else:
            state["cash"] = cash
            state["holding"] = holding
            state["shares"] = shares
            state["entry_price"] = entry_price
            overlay = live.evaluate_v4_overlay(
                data=hist,
                idx_map=idx_map,
                trading_dates=dates[: i + 1],
                td=td,
                holding=holding,
                base_target=target,
                candidates=candidates,
                scheduled_rebalance=scheduled,
                state=state,
                mode="V4",
            )
            risk = (
                v5.assess_v5(
                    target=overlay["target"],
                    holding=holding,
                    state=state,
                    data=hist,
                    td=td,
                    idx_map=idx_map,
                    is_rebalance=overlay["is_rebalance"],
                    common_dates=dates[: i + 1],
                    spot_map=None,
                )
                if mode == "V5"
                else live.risk_assess(
                    target=overlay["target"],
                    holding=holding,
                    state=state,
                    data=hist,
                    td=td,
                    idx_map=idx_map,
                    is_rebalance=overlay["is_rebalance"],
                    common_dates=dates[: i + 1],
                    spot_map=None,
                )
            )
            final_target = risk.final_target or live.DEFENSE
            rebalance = overlay["is_rebalance"] or risk.action == live.ACTION_EMERGENCY
        price = float(today[final_target].px1450) if final_target in today else 0.0
        if rebalance and final_target != holding and price > 0:
            if holding and shares:
                cash += shares * float(today[holding].px1450) * (1 - SELL_COST * cost_mult)
            shares = int(cash * 0.99 / price / 100) * 100
            if shares:
                cash -= shares * price * (1 + BUY_COST * cost_mult)
                holding = final_target
                entry_price = price
                trades += 1
            else:
                holding = None
                shares = 0
        equity = cash + shares * float(today[holding].px1450) if holding else cash
        state["peak_equity"] = max(float(state.get("peak_equity", INITIAL)), equity)
        curve.append({"trade_date": td, "equity": equity, "holding": holding, "trades": trades})
    result = metrics(curve)
    result["mode"] = mode
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    all_dates = sorted(set.union(*(set(data[c].trade_date) for c in data)))
    dates = [d for d in all_dates if d >= date(2020, 6, 19)]
    anchor = all_dates.index(date(2020, 6, 19))
    grid = set(all_dates[anchor::5])
    results = {
        f"{mode}_{mult}x": run(mode, deepcopy(data), dates, grid, cost_mult=mult)
        for mode in ("V3-G", "V4", "V5")
        for mult in (1.0, 2.0, 3.0)
    }
    payload = {
        "note": "Strict 14:50 full replay; signal and fill use 14:50; 2020-06-19..2026-10-26.",
        "data": "purchased 5-minute parquet, adjusted by adj_factor",
        "results": results,
    }
    (OUT / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
