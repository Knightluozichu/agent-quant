"""C11 breadth study: exact 600322 template on a 500-stock sample + sectors."""

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
MAX_MCAP = 3_000_000_000.0
MAX_HOLD = 30
STOP = -0.07
COST = 0.0035


def sector_map() -> dict[str, str]:
    path = CACHE / "sector_map.csv"
    if path.exists():
        df = pd.read_csv(path)
        return dict(zip(df.symbol, df.sector, strict=True))
    spot = ak.stock_sector_spot()
    mapping: dict[str, str] = {}
    for _, row in spot.iterrows():
        try:
            detail = ak.stock_sector_detail(sector=row["label"])
            for symbol in detail["symbol"].astype(str):
                mapping[symbol] = row["板块"]
        except Exception as exc:
            print(f"  skip sector {row['label']}: {exc}")
            continue
        time.sleep(0.05)
    df = pd.DataFrame([{"symbol": k, "sector": v} for k, v in mapping.items()])
    df.to_csv(path, index=False)
    return mapping


def load() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(CACHE.glob("*.parquet")):
        if path.stem in {"sample_200", "sample_500"}:
            continue
        df = pd.read_parquet(path)
        if "trade_date" not in df.columns or len(df) < 200 or "outstanding_share" not in df.columns:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df.close.astype(float)
        high = df.high.astype(float)
        df["ma5"] = close.rolling(5).mean()
        df["ma10"] = close.rolling(10).mean()
        df["ma20"] = close.rolling(20).mean()
        df["prior5_high"] = high.shift(1).rolling(5).max()
        df["high60"] = high.rolling(60).max()
        df["vol20"] = df.volume.rolling(20).mean()
        df["mcap"] = close * df.outstanding_share.astype(float)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["dif"] = ema12 - ema26
        df["dea"] = df.dif.ewm(span=9, adjust=False).mean()
        df["hist"] = df.dif - df.dea
        df["prev_hist"] = df["hist"].shift(1)
        data[path.stem] = df
    return data


def market_regime() -> pd.Series:
    df = pd.read_parquet(ROOT / "data/cross_asset/159915.parquet")[
        ["trade_date", "close"]
    ].sort_values("trade_date")
    df["ma20"] = df.close.rolling(20).mean()
    return pd.Series((df.close > df.ma20).to_numpy(), index=df.trade_date)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    sectors = sector_map()
    regime = market_regime()
    events = []
    for code, df in data.items():
        for i in range(61, len(df) - 1):
            row = df.iloc[i]
            if (
                not np.isfinite(row.high60)
                or row.high60 <= 0
                or row.mcap >= MAX_MCAP
                or row.close >= 3
            ):
                continue
            dd = row.close / row.high60 - 1
            if dd > -0.20 or dd < -0.30:
                continue
            if row.close <= row.ma10 or row.ma5 <= row.ma10 or row.close >= row.ma20:
                continue
            if row.close <= row.prior5_high:
                continue
            if (
                not np.isfinite(row.vol20)
                or row.volume < row.vol20 * 0.35
                or row.volume > row.vol20 * 0.80
            ):
                continue
            if (
                not np.isfinite(row["hist"])
                or not np.isfinite(row["prev_hist"])
                or row["hist"] <= row["prev_hist"]
            ):
                continue
            entry = float(row.close)
            exit_i = None
            reason = "max_hold"
            for j in range(i + 1, min(i + 1 + MAX_HOLD, len(df))):
                r = df.iloc[j]
                if r.close <= entry * (1 + STOP):
                    exit_i, reason = j, "stop"
                    break
                if r.dif < r.dea and df.iloc[j - 1].dif >= df.iloc[j - 1].dea:
                    exit_i, reason = j, "macd_dead"
                    break
            if exit_i is None:
                exit_i = min(i + MAX_HOLD, len(df) - 1)
            events.append(
                {
                    "code": code,
                    "sector": sectors.get(code, "未知"),
                    "entry_date": str(df.iloc[i].trade_date),
                    "exit_date": str(df.iloc[exit_i].trade_date),
                    "return_pct": round((float(df.iloc[exit_i].close / entry - 1) - COST) * 100, 2),
                    "hold_days": exit_i - i,
                    "reason": reason,
                    "dd60_pct": round(dd * 100, 1),
                    "mcap_yi": round(float(row.mcap / 1e8), 1),
                    "price": round(entry, 2),
                    "market_above": bool(regime.get(df.iloc[i].trade_date, False)),
                }
            )
    frame = pd.DataFrame(events)
    frame.to_csv(OUT / "c11_events_500.csv", index=False)
    if frame.empty:
        print("no events")
        return
    frame["year"] = frame.entry_date.str[:4]
    summary = {
        "stocks": len(data),
        "events": len(frame),
        "events_per_year": frame.groupby("year").size().to_dict(),
        "win_rate": round(float((frame.return_pct > 0).mean()), 3),
        "mean_pct": round(float(frame.return_pct.mean()), 2),
        "median_pct": round(float(frame.return_pct.median()), 2),
        "best_pct": round(float(frame.return_pct.max()), 2),
        "worst_pct": round(float(frame.return_pct.min()), 2),
        "by_sector": frame.groupby("sector")
        .agg(
            n=("return_pct", "size"),
            win=("return_pct", lambda x: round(float((x > 0).mean()), 3)),
            mean=("return_pct", lambda x: round(float(x.mean()), 2)),
        )
        .sort_values("n", ascending=False)
        .head(15)
        .to_dict("index"),
        "target_600322": frame[frame.code == "sh600322"].to_dict("records"),
    }
    (OUT / "c11_breadth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
