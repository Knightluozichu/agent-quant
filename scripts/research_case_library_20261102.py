"""Find 600322-style early washout reversal cases in the A-share sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/stock_cache"
OUT = ROOT / "artifacts/project_c_20261102"
MAX_MCAP = 10_000_000_000.0
MAX_HOLD = 30
STOP = -0.07
COST = 0.0035


def load_stocks() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(CACHE.glob("*.parquet")):
        if path.stem in {"sample_200"}:
            continue
        df = pd.read_parquet(path)
        if "trade_date" not in df.columns or len(df) < 200 or "outstanding_share" not in df.columns:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df.close.astype(float)
        high = df.high.astype(float)
        low = df.low.astype(float)
        df["ma5"] = close.rolling(5).mean()
        df["ma10"] = close.rolling(10).mean()
        df["ma20"] = close.rolling(20).mean()
        df["high60"] = high.rolling(60).max()
        df["low60"] = low.rolling(60).min()
        df["vol20"] = df.volume.rolling(20).mean()
        df["ret5"] = close / close.shift(5) - 1
        df["mcap"] = close * df.outstanding_share.astype(float)
        data[path.stem] = df
    return data


def market_regime() -> pd.Series:
    path = ROOT / "data/cross_asset/159915.parquet"
    df = pd.read_parquet(path)[["trade_date", "close"]].sort_values("trade_date")
    df["ma20"] = df.close.rolling(20).mean()
    df["above"] = df.close > df.ma20
    return pd.Series(df.above.to_numpy(), index=df.trade_date)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_stocks()
    regime = market_regime()
    events = []
    for code, df in data.items():
        if len(df) < 100:
            continue
        dates = list(df.trade_date)
        for i in range(61, len(df) - 1):
            row = df.iloc[i]
            if not np.isfinite(row.ma20) or not np.isfinite(row.high60) or row.high60 <= 0:
                continue
            if row.mcap >= MAX_MCAP or row.close <= 0:
                continue
            dd = row.close / row.high60 - 1
            if dd > -0.15 or dd < -0.45:
                continue
            if row.close <= row.ma10 or row.ma5 <= row.ma10:
                continue
            if row.close > row.ma20 * 1.01:
                continue
            if not np.isfinite(row.vol20) or row.vol20 <= 0:
                continue
            volx = row.volume / row.vol20
            if volx < 0.4 or volx > 1.2:
                continue
            # MA5 crossed above MA10 within the last 3 sessions.
            crossed = any(
                df.iloc[k].ma5 > df.iloc[k].ma10 and df.iloc[k - 1].ma5 <= df.iloc[k - 1].ma10
                for k in range(max(1, i - 2), i + 1)
            )
            if not crossed:
                continue
            entry = float(row.close)
            exit_i = None
            reason = "max_hold"
            for j in range(i + 1, min(i + 1 + MAX_HOLD, len(df))):
                r = df.iloc[j]
                if r.close <= entry * (1 + STOP):
                    exit_i, reason = j, "stop"
                    break
                if r.ma5 < r.ma10 and df.iloc[j - 1].ma5 >= df.iloc[j - 1].ma10:
                    exit_i, reason = j, "dead_cross"
                    break
            if exit_i is None:
                exit_i = min(i + MAX_HOLD, len(df) - 1)
            ret = float(df.iloc[exit_i].close / entry - 1) - COST
            events.append(
                {
                    "code": code,
                    "entry_date": str(dates[i]),
                    "exit_date": str(dates[exit_i]),
                    "entry": round(entry, 3),
                    "exit": round(float(df.iloc[exit_i].close), 3),
                    "return_pct": round(ret * 100, 2),
                    "hold_days": exit_i - i,
                    "reason": reason,
                    "mcap_yi": round(float(row.mcap / 1e8), 1),
                    "dd60_pct": round(dd * 100, 1),
                    "volx": round(volx, 2),
                    "price": round(entry, 2),
                    "ret5_pct": round(float(row.ret5) * 100, 1) if np.isfinite(row.ret5) else None,
                    "ma20_gap_pct": round((row.close / row.ma20 - 1) * 100, 1),
                    "market_above_ma20": bool(regime.get(dates[i], False)),
                }
            )
    frame = pd.DataFrame(events)
    frame.to_csv(OUT / "case_library_events.csv", index=False)
    if frame.empty:
        print("no events")
        return
    wins = frame.return_pct > 0
    summary = {
        "events": len(frame),
        "win_rate": round(float(wins.mean()), 3),
        "mean_return_pct": round(float(frame.return_pct.mean()), 2),
        "median_return_pct": round(float(frame.return_pct.median()), 2),
        "p90_return_pct": round(float(frame.return_pct.quantile(0.9)), 2),
        "worst_pct": round(float(frame.return_pct.min()), 2),
        "best_pct": round(float(frame.return_pct.max()), 2),
        "dead_cross_exit_pct": round(float((frame.reason == "dead_cross").mean()) * 100, 1),
        "market_above_win_rate": round(float(wins[frame.market_above_ma20].mean()), 3),
        "market_below_win_rate": round(float(wins[~frame.market_above_ma20].mean()), 3),
        "top20": frame.sort_values("return_pct", ascending=False).head(20).to_dict("records"),
        "worst10": frame.sort_values("return_pct").head(10).to_dict("records"),
    }
    # 600322 specific event, if present.
    target = frame[(frame.code == "sh600322") & (frame.entry_date == "2022-11-09")]
    summary["target_600322"] = target.to_dict("records")
    (OUT / "case_library_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
