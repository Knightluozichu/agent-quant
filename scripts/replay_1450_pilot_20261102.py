"""Strict 14:50 pilot for V3-G/V4/V5 using Sina 5-minute bars.

Only 2026-08-27..2026-11-02 is available; this is a pilot, not full history.
Signal and fill both use the 14:50 price (open of the 14:55 5-min bar).
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
DAILY = ROOT / "artifacts/tencent_v4_v5_20261102/inputs"
MIN = ROOT / "data/stock_cache"
OUT = ROOT / "artifacts/replay_1450_20261102"
CODES = ["518880", "159985", "501018", "161226", "513100", "159915", "511220", "511880"]
INITIAL = 10_000.0
BUY_COST = 0.0015
SELL_COST = 0.0020


def load_daily() -> dict[str, pd.DataFrame]:
    data = {}
    for code in CODES:
        df = (
            pd.read_parquet(DAILY / f"{code}.parquet")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        df["trade_date"] = pd.to_datetime(df.trade_date).dt.date
        data[code] = df
    return data


def load_1450() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    bars: dict[str, pd.DataFrame] = {}
    for code in CODES:
        df = pd.read_parquet(MIN / f"min_etf_{code}.parquet")
        df["dt"] = pd.to_datetime(df.dt)
        df["date"] = df.dt.dt.date
        df["time"] = df.dt.dt.strftime("%H:%M:%S")
        bars[code] = df
        for d, g in df.groupby("date"):
            g = g.sort_values("time")
            at = g[g.time == "14:55:00"]
            if at.empty:
                at = g[g.time <= "14:55:00"]
            if at.empty:
                continue
            bar = at.iloc[-1]
            price = float(bar.open) if bar.time == "14:55:00" else float(bar.close)
            vol = float(g[g.time <= bar.time].volume.astype(float).sum())
            rows.append({"trade_date": d, "code": code, "price": price, "volume": vol})
    frame = pd.DataFrame(rows)
    return frame, bars


def metrics(curve: list[dict]) -> dict:
    frame = pd.DataFrame(curve)
    eq = frame.equity.to_numpy(dtype=float)
    path = np.r_[INITIAL, eq]
    mdd = float((path / np.maximum.accumulate(path) - 1).min()) if len(path) else 0.0
    return {
        "final_equity": round(float(eq[-1]), 2),
        "total_return_pct": round(float(eq[-1] / INITIAL - 1) * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "days": len(frame),
        "trades": int(frame.trades.iloc[-1]) if "trades" in frame else 0,
    }


def run(
    mode: str, data: dict[str, pd.DataFrame], px: pd.DataFrame, dates: list[date], grid: set[date]
) -> dict:
    cash = INITIAL
    holding = None
    shares = 0
    entry_price = 0.0
    state = live.default_state(INITIAL)
    state.update({"cash": cash, "holding": None, "shares": 0, "entry_price": 0.0})
    curve = []
    trades = 0
    for td in dates:
        today = px[px.trade_date == td].set_index("code")
        if len(today) < len(CODES):
            continue
        hist = {}
        idx_map = {}
        for code in CODES:
            base = data[code][data[code].trade_date < td].copy()
            row = {
                "trade_date": td,
                "open": float(today.loc[code, "price"]),
                "close": float(today.loc[code, "price"]),
                "high": float(today.loc[code, "price"]),
                "low": float(today.loc[code, "price"]),
                "volume": float(today.loc[code, "volume"]),
                "symbol": code,
            }
            base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
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
                trading_dates=dates,
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
                    common_dates=dates,
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
                    common_dates=dates,
                    spot_map=None,
                )
            )
            final_target = risk.final_target or live.DEFENSE
            rebalance = overlay["is_rebalance"] or risk.action == live.ACTION_EMERGENCY
        price = float(today.loc[final_target, "price"])
        if rebalance and final_target != holding and price > 0:
            if holding and shares:
                cash += shares * float(today.loc[holding, "price"]) * (1 - SELL_COST)
            shares = int(cash * 0.99 / price / 100) * 100
            if shares:
                cash -= shares * price * (1 + BUY_COST)
                holding = final_target
                entry_price = price
                trades += 1
            else:
                holding = None
                shares = 0
        equity = cash + shares * float(today.loc[holding, "price"]) if holding else cash
        state["peak_equity"] = max(float(state.get("peak_equity", INITIAL)), equity)
        curve.append({"trade_date": td, "equity": equity, "holding": holding, "trades": trades})
    result = metrics(curve)
    result["mode"] = mode
    result["start"] = str(curve[0]["trade_date"])
    result["end"] = str(curve[-1]["trade_date"])
    result["holding"] = holding
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_daily()
    px, _bars = load_1450()
    dates = sorted(px.trade_date.unique())
    all_dates = sorted(set.union(*(set(data[c].trade_date) for c in CODES)) | set(dates))
    anchor = all_dates.index(date(2020, 6, 19))
    grid = set(all_dates[anchor::5])
    results = {mode: run(mode, deepcopy(data), px, dates, grid) for mode in ("V3-G", "V4", "V5")}
    payload = {
        "note": "Strict 14:50 pilot; 2026-08-27..2026-11-02; not full history.",
        "fill": "open of 14:55 5-min bar = 14:50 price",
        "cost": {"buy": BUY_COST, "sell": SELL_COST},
        "results": results,
    }
    (OUT / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
