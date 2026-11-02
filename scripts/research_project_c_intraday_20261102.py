"""Project C4 pilot: 5-minute intraday T+0 overlay on a base position.

Only recent 5-minute history is available (about two months), so this is a
pilot, not a 6-year backtest. No broker, no orders.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/stock_cache"
OUT = ROOT / "artifacts/project_c_20261102"
SYMBOLS = {"sz159915": "创业板ETF", "sh510300": "沪深300ETF", "sz000001": "平安银行"}
INITIAL = 10_000.0
BASE_CAPITAL = 5_000.0
OVERLAY = 5_000.0
COST = 0.0015
ENTRY = "10:00:00"
EXIT = "14:55:00"


def load_minute(symbol: str) -> pd.DataFrame:
    path = CACHE / f"min_{symbol}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
    df = df.rename(columns={"day": "dt"})
    df["dt"] = pd.to_datetime(df.dt)
    df = df.dropna(subset=["close"]).sort_values("dt").reset_index(drop=True)
    df.to_parquet(path, index=False)
    time.sleep(0.2)
    return df


def metrics(curve: pd.DataFrame) -> dict:
    equity = curve.equity.to_numpy(dtype=float)
    path = np.r_[INITIAL, equity]
    days = (curve.dt.iloc[-1] - curve.dt.iloc[0]).days
    years = max(days / 365.25, 1 / 365.25)
    monthly = curve.set_index("dt").equity.resample("ME").last().pct_change().dropna()
    return {
        "final_equity": round(float(equity[-1]), 2),
        "total_return_pct": round(float(equity[-1] / INITIAL - 1) * 100, 2),
        "cagr_pct": round(float((equity[-1] / INITIAL) ** (1 / years) - 1) * 100, 1),
        "max_drawdown_pct": round(float((path / np.maximum.accumulate(path) - 1).min()) * 100, 2),
        "days": len(curve),
        "months": len(monthly),
        "monthly_returns_pct": [round(float(v) * 100, 2) for v in monthly],
        "overlay_pnl": round(float(curve.overlay_pnl.iloc[-1]), 2),
    }


def run_one(symbol: str, rule: str) -> dict:
    df = load_minute(symbol)
    df["date"] = df.dt.dt.date
    df["time"] = df.dt.dt.strftime("%H:%M:%S")
    rows = []
    base_shares = 0
    cash = INITIAL
    overlay_pnl = 0.0
    base_bought = False
    for date, day in df.groupby("date"):
        day = day.set_index("time")
        if ENTRY not in day.index or EXIT not in day.index:
            continue
        start = day.iloc[0]
        entry_px = float(day.loc[ENTRY, "close"])
        exit_px = float(day.loc[EXIT, "close"])
        if not base_bought:
            base_shares = int(BASE_CAPITAL / entry_px / 100) * 100
            cash -= base_shares * entry_px * (1 + COST)
            base_bought = True
        first_move = entry_px / float(start.open) - 1
        direction = 0
        if rule == "momentum":
            if first_move > 0.002:
                direction = 1
            elif first_move < -0.002:
                direction = -1
        elif rule == "fade":
            if first_move > 0.002:
                direction = -1
            elif first_move < -0.002:
                direction = 1
        if direction:
            trade_ret = exit_px / entry_px - 1
            overlay_pnl += direction * trade_ret * OVERLAY - COST * 2 * OVERLAY
        close_px = float(day.iloc[-1].close)
        equity = cash + base_shares * close_px + overlay_pnl
        rows.append(
            {
                "dt": day.iloc[-1].name if False else pd.Timestamp(f"{date} 15:00:00"),
                "equity": equity,
                "overlay_pnl": overlay_pnl,
            }
        )
    curve = pd.DataFrame(rows)
    return {
        "symbol": symbol,
        "name": SYMBOLS[symbol],
        "rule": rule,
        "metrics": metrics(curve),
        "curve": curve,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for symbol in SYMBOLS:
        for rule in ("momentum", "fade"):
            result = run_one(symbol, rule)
            result["curve"].to_csv(OUT / f"c4_{symbol}_{rule}.csv", index=False)
            results.append({k: v for k, v in result.items() if k != "curve"})
    payload = {
        "project": "Project C4 pilot",
        "candidate": "5-minute intraday T+0 overlay on a 50% base position",
        "data_window": "recent ~2 months; not a 6-year backtest",
        "constraints": {
            "capital": INITIAL,
            "base": BASE_CAPITAL,
            "overlay": OVERLAY,
            "leverage": "none",
        },
        "results": results,
    }
    (OUT / "c4_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
