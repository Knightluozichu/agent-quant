"""Simplified Chanlun third-buy backtest, 2015-2026, 10k, A-share sample.

This is a testable simplification (fractal/stroke/pivot/third-buy), not the
canonical discretionary Chanlun. Research only; no broker.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/chanlun_cache"
OUT = ROOT / "artifacts/chanlun_20261102"
INITIAL = 10_000.0
BUY_COST = 0.0015
SELL_COST = 0.0020
MAX_POS = 4
MAX_HOLD = 30
STOP = -0.07


@dataclass
class Bar:
    date: object
    high: float
    low: float
    close: float


@dataclass
class Pivot:
    end_idx: int
    zg: float
    zd: float


def merge_bars(df: pd.DataFrame) -> list[Bar]:
    merged: list[Bar] = []
    for _, row in df.iterrows():
        bar = Bar(row.trade_date, float(row.high), float(row.low), float(row.close))
        if len(merged) < 2:
            merged.append(bar)
            continue
        prev = merged[-1]
        inclusive = (prev.high >= bar.high and prev.low <= bar.low) or (
            bar.high >= prev.high and bar.low <= prev.low
        )
        if not inclusive:
            merged.append(bar)
            continue
        trend = 1 if prev.high > merged[-2].high else -1
        if trend == 1:
            high, low = max(prev.high, bar.high), max(prev.low, bar.low)
        else:
            high, low = min(prev.high, bar.high), min(prev.low, bar.low)
        merged[-1] = Bar(bar.date, high, low, bar.close)
    return merged


def build_pivots(merged: list[Bar]) -> list[Pivot]:
    fractals: list[tuple[int, str, float]] = []
    for i in range(1, len(merged) - 1):
        a, b, c = merged[i - 1], merged[i], merged[i + 1]
        if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
            fractals.append((i, "top", b.high))
        elif b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
            fractals.append((i, "bottom", b.low))
    strokes: list[tuple[int, int, float, float]] = []
    last: tuple[int, str, float] | None = None
    for idx, kind, price in fractals:
        if last is None:
            last = (idx, kind, price)
            continue
        if kind == last[1]:
            if (kind == "top" and price > last[2]) or (kind == "bottom" and price < last[2]):
                last = (idx, kind, price)
            continue
        if idx - last[0] < 4:
            continue
        strokes.append((last[0], idx, last[2], price))
        last = (idx, kind, price)
    pivots: list[Pivot] = []
    for i in range(len(strokes) - 2):
        trio = strokes[i : i + 3]
        highs = [max(s[2], s[3]) for s in trio]
        lows = [min(s[2], s[3]) for s in trio]
        zg, zd = min(highs), max(lows)
        if zg > zd:
            pivots.append(Pivot(trio[-1][1], zg, zd))
    return pivots


def metrics(curve: pd.DataFrame) -> dict:
    equity = curve.equity.to_numpy(dtype=float)
    path = np.r_[INITIAL, equity]
    years = (
        pd.to_datetime(curve.trade_date.iloc[-1]) - pd.to_datetime(curve.trade_date.iloc[0])
    ).days / 365.25
    returns = np.diff(path) / path[:-1]
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    monthly = (
        curve.set_index(pd.to_datetime(curve.trade_date))
        .equity.resample("ME")
        .last()
        .pct_change()
        .dropna()
    )
    return {
        "final_equity": round(float(equity[-1]), 2),
        "total_return": round(float(equity[-1] / INITIAL - 1) * 100, 1),
        "cagr": round(float((equity[-1] / INITIAL) ** (1 / years) - 1) * 100, 1) if years else 0.0,
        "sharpe": round(float(returns.mean() / vol * np.sqrt(252)), 2) if vol else 0.0,
        "max_drawdown": round(float((path / np.maximum.accumulate(path) - 1).min()) * 100, 1),
        "years": round(years, 1),
        "months": len(monthly),
        "months_ge_10pct": int((monthly >= 0.10).sum()),
        "months_ge_20pct": int((monthly >= 0.20).sum()),
        "best_month": round(float(monthly.max()) * 100, 1) if len(monthly) else 0.0,
        "worst_month": round(float(monthly.min()) * 100, 1) if len(monthly) else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stocks: dict[str, pd.DataFrame] = {}
    pivots: dict[str, list[Pivot]] = {}
    pivot_dates: dict[str, list[object]] = {}
    for path in sorted(CACHE.glob("*.parquet")):
        df = pd.read_parquet(path).sort_values("trade_date").reset_index(drop=True)
        if len(df) < 500:
            continue
        merged = merge_bars(df)
        p = build_pivots(merged)
        if not p:
            continue
        stocks[path.stem] = df
        pivots[path.stem] = p
        pivot_dates[path.stem] = [merged[x.end_idx].date for x in p]

    dates = sorted(set.union(*(set(df.trade_date) for df in stocks.values())))
    maps = {c: {d: i for i, d in enumerate(df.trade_date)} for c, df in stocks.items()}
    cash = INITIAL
    positions: dict[str, dict] = {}
    pending_buys: list[tuple[str, float]] = []
    pending_sells: set[str] = set()
    curve = []
    peak = INITIAL
    stops = 0
    trades = 0

    for i, td in enumerate(dates):
        # sells at open
        for code in list(pending_sells):
            if code not in positions:
                pending_sells.discard(code)
                continue
            idx = maps[code].get(td)
            if idx is None:
                continue
            open_px = float(stocks[code].iloc[idx].open)
            if open_px <= 0:
                continue
            cash += positions[code]["shares"] * open_px * (1 - SELL_COST)
            del positions[code]
            pending_sells.discard(code)
        # buys at open
        for code, zd in list(pending_buys):
            if code in positions:
                pending_buys.remove((code, zd))
                continue
            idx = maps[code].get(td)
            if idx is None:
                continue
            open_px = float(stocks[code].iloc[idx].open)
            if open_px <= 0:
                pending_buys.remove((code, zd))
                continue
            slots = MAX_POS - len(positions)
            if slots <= 0:
                break
            budget = cash / slots
            shares = int(budget / open_px / 100) * 100
            if shares <= 0:
                pending_buys.remove((code, zd))
                continue
            cash -= shares * open_px * (1 + BUY_COST)
            positions[code] = {"shares": shares, "entry": open_px, "entry_i": i, "zd": zd}
            pending_buys.remove((code, zd))
            trades += 1

        # close equity
        equity = cash + sum(
            pos["shares"] * float(stocks[code].iloc[maps[code][td]].close)
            for code, pos in positions.items()
            if td in maps[code]
        )
        peak = max(peak, equity)
        if equity / peak - 1 <= -0.10 and not pending_sells:
            pending_sells = set(positions)
            stops += 1
            peak = equity

        # exit checks
        for code, pos in list(positions.items()):
            idx = maps[code].get(td)
            if idx is None or pos.get("exit"):
                continue
            close = float(stocks[code].iloc[idx].close)
            if (
                close <= pos["entry"] * (1 + STOP)
                or close < pos["zd"]
                or i - pos["entry_i"] >= MAX_HOLD
            ):
                pending_sells.add(code)
                pos["exit"] = True

        # third-buy signal
        for code, df in stocks.items():
            if code in positions or any(x[0] == code for x in pending_buys):
                continue
            idx = maps[code].get(td)
            if idx is None or idx < 30:
                continue
            pdates = pivot_dates[code]
            pos = bisect_left(pdates, td)
            if pos == 0:
                continue
            pivot = pivots[code][pos - 1]
            close = float(df.iloc[idx].close)
            low = float(df.iloc[idx].low)
            recent = df.iloc[max(0, idx - 10) : idx]
            breakout = (recent.close > pivot.zg).any()
            retest = close > pivot.zg and low >= pivot.zg * 0.99 and low <= pivot.zg * 1.02
            if breakout and retest:
                pending_buys.append((code, pivot.zd))
        curve.append(
            {
                "trade_date": td,
                "equity": equity,
                "cash": cash,
                "positions": ",".join(sorted(positions)) or None,
            }
        )

    frame = pd.DataFrame(curve)
    result = metrics(frame)
    result.update({"stocks": len(stocks), "trades": trades, "stops": stops, "days": len(frame)})
    payload = {
        "project": "Simplified Chanlun",
        "rule": (
            "third buy: breakout above pivot ZG, retest ZG, buy next open; exit close<ZD/-7%/30d"
        ),
        "constraints": {"capital": INITIAL, "max_positions": MAX_POS, "leverage": "none"},
        "result": result,
        "bias": (
            "300-stock current-listed sample; survivorship bias; "
            "simplified rules, not canonical Chanlun."
        ),
    }
    (OUT / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(OUT / "curve.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
