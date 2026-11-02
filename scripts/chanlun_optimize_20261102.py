"""Pre-registered Chanlun third-buy variants: trend/volume/confirmation/risk."""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/chanlun_cache"
OUT = ROOT / "artifacts/chanlun_20261102"
INITIAL = 10_000.0
BUY_COST = 0.0015
SELL_COST = 0.0020


def load() -> tuple[dict[str, pd.DataFrame], dict[str, list], dict[str, list]]:
    from chanlun_backtest_20261102 import build_pivots, merge_bars

    stocks: dict[str, pd.DataFrame] = {}
    pivots: dict[str, list] = {}
    pivot_dates: dict[str, list] = {}
    for path in sorted(CACHE.glob("*.parquet")):
        df = pd.read_parquet(path).sort_values("trade_date").reset_index(drop=True)
        if len(df) < 500:
            continue
        close = df.close.astype(float)
        df["ma60"] = close.rolling(60).mean()
        df["vol20"] = df.volume.rolling(20).mean()
        tr = pd.concat(
            [
                df.high - df.low,
                (df.high - close.shift()).abs(),
                (df.low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        merged = merge_bars(df)
        p = build_pivots(merged)
        if not p:
            continue
        stocks[path.stem] = df
        pivots[path.stem] = p
        pivot_dates[path.stem] = [merged[x.end_idx].date for x in p]
    return stocks, pivots, pivot_dates


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
        "months_ge_10pct": int((monthly >= 0.10).sum()),
        "months_ge_20pct": int((monthly >= 0.20).sum()),
        "best_month": round(float(monthly.max()) * 100, 1) if len(monthly) else 0.0,
        "worst_month": round(float(monthly.min()) * 100, 1) if len(monthly) else 0.0,
    }


def run(
    stocks,
    pivots,
    pivot_dates,
    *,
    trend=False,
    volume=False,
    confirm=False,
    atr_stop=False,
    max_pos=4,
) -> dict:
    dates = sorted(set.union(*(set(df.trade_date) for df in stocks.values())))
    maps = {c: {d: i for i, d in enumerate(df.trade_date)} for c, df in stocks.items()}
    cash = INITIAL
    positions: dict[str, dict] = {}
    pending_buys: list[tuple[str, float, float]] = []
    pending_sells: set[str] = set()
    curve = []
    peak = INITIAL
    trades = 0
    stops = 0

    for i, td in enumerate(dates):
        for code in list(pending_sells):
            if code not in positions:
                pending_sells.discard(code)
                continue
            idx = maps[code].get(td)
            if idx is None:
                continue
            open_px = float(stocks[code].iloc[idx].open)
            if open_px > 0:
                cash += positions[code]["shares"] * open_px * (1 - SELL_COST)
                del positions[code]
                pending_sells.discard(code)

        for code, zd, stop_px in list(pending_buys):
            if code in positions:
                pending_buys.remove((code, zd, stop_px))
                continue
            idx = maps[code].get(td)
            if idx is None:
                continue
            open_px = float(stocks[code].iloc[idx].open)
            if open_px <= 0:
                pending_buys.remove((code, zd, stop_px))
                continue
            slots = max_pos - len(positions)
            if slots <= 0:
                break
            budget = cash / slots
            shares = int(budget / open_px / 100) * 100
            if shares <= 0:
                pending_buys.remove((code, zd, stop_px))
                continue
            cash -= shares * open_px * (1 + BUY_COST)
            positions[code] = {
                "shares": shares,
                "entry": open_px,
                "entry_i": i,
                "zd": zd,
                "stop": stop_px,
            }
            pending_buys.remove((code, zd, stop_px))
            trades += 1

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

        for code, pos in list(positions.items()):
            idx = maps[code].get(td)
            if idx is None or pos.get("exit"):
                continue
            close = float(stocks[code].iloc[idx].close)
            if close <= pos["stop"] or close < pos["zd"] or i - pos["entry_i"] >= 30:
                pending_sells.add(code)
                pos["exit"] = True

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
            if trend and (not np.isfinite(df.iloc[idx].ma60) or close <= df.iloc[idx].ma60):
                continue
            recent = df.iloc[max(0, idx - 10) : idx]
            breakout = (recent.close > pivot.zg).any()
            retest = close > pivot.zg and low >= pivot.zg * 0.99 and low <= pivot.zg * 1.02
            if confirm and not (idx >= 1 and df.iloc[idx - 1].close > pivot.zg):
                continue
            if volume and (
                not np.isfinite(df.iloc[idx].vol20)
                or df.iloc[idx].volume < df.iloc[idx].vol20 * 1.2
            ):
                continue
            if breakout and retest:
                stop_px = (
                    float(df.iloc[idx].close - 2 * df.iloc[idx].atr14)
                    if atr_stop
                    else float(close * 0.93)
                )
                pending_buys.append((code, pivot.zd, stop_px))
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
    result.update({"trades": trades, "stops": stops})
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stocks, pivots, pivot_dates = load()
    variants = {
        "baseline": {},
        "trend": {"trend": True},
        "trend_volume": {"trend": True, "volume": True},
        "trend_confirm": {"trend": True, "confirm": True},
        "trend_atr_stop": {"trend": True, "atr_stop": True},
        "trend_volume_atr": {"trend": True, "volume": True, "atr_stop": True},
    }
    results = {
        name: run(stocks, pivots, pivot_dates, **kwargs) for name, kwargs in variants.items()
    }
    payload = {"stocks": len(stocks), "variants": results}
    (OUT / "optimization.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
