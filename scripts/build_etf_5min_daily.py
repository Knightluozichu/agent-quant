"""Build adjusted daily + 14:50 bars from the purchased 5-minute parquet files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/etf_5min"
OUT = ROOT / "data/etf_5min_daily"
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for code in CODES:
        df = pd.read_parquet(SRC / f"{code}.parquet").reset_index()
        df["trade_date"] = pd.to_datetime(df.trade_date).dt.date
        df["trade_time"] = pd.to_datetime(df.trade_time)
        df["hm"] = df.trade_time.dt.strftime("%H:%M:%S")
        df = df[(df.close > 0) & (df.hm >= "09:30:00") & (df.hm <= "15:00:00")].copy()
        df["adj_factor"] = pd.to_numeric(df.adj_factor, errors="coerce").fillna(1.0)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce") * df.adj_factor
        df["vol"] = pd.to_numeric(df.vol, errors="coerce").fillna(0.0)
        daily = (
            df.groupby("trade_date")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("vol", "sum"),
                n=("close", "size"),
            )
            .reset_index()
        )
        # 14:50 price = open of the 14:55 bar; fall back to last bar <= 14:50.
        rows = []
        for d, g in df.groupby("trade_date"):
            g = g.sort_values("hm")
            at = g[g.hm == "14:55:00"]
            if at.empty:
                at = g[g.hm <= "14:50:00"]
            if at.empty:
                continue
            bar = at.iloc[-1]
            up = g[g.hm <= bar.hm]
            rows.append(
                {
                    "trade_date": d,
                    "open1450": float(up.open.iloc[0]),
                    "high1450": float(up.high.max()),
                    "low1450": float(up.low.min()),
                    "px1450": float(bar.open),
                    "vol1450": float(up.vol.sum()),
                }
            )
        p1450 = pd.DataFrame(rows)
        merged = daily.merge(p1450, on="trade_date", how="left")
        merged.to_parquet(OUT / f"{code}.parquet", index=False)
        report[code] = {
            "rows": len(merged),
            "start": str(merged.trade_date.min()),
            "end": str(merged.trade_date.max()),
            "missing_1450": int(merged.px1450.isna().sum()),
            "days_2020_2026": int(
                (
                    (merged.trade_date >= pd.Timestamp("2020-01-01").date())
                    & (merged.trade_date <= pd.Timestamp("2026-10-31").date())
                ).sum()
            ),
            "adj_factor_last": float(df.adj_factor.iloc[-1]),
        }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
