"""C11 sector-strength filter: does industry momentum/breadth improve the template?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/stock_cache"
OUT = ROOT / "artifacts/project_c_20261102"


def load() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    sectors = dict(pd.read_csv(CACHE / "sector_map.csv").values)
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(CACHE.glob("*.parquet")):
        if path.stem in {"sample_200", "sample_500"}:
            continue
        df = pd.read_parquet(path)
        if "trade_date" not in df.columns or len(df) < 200:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df.close.astype(float)
        df["ma20"] = close.rolling(20).mean()
        data[path.stem] = df[["trade_date", "close", "ma20"]]
    return data, sectors


def main() -> None:
    data, sectors = load()
    events = pd.read_csv(OUT / "c11_events_500.csv")
    events["entry_date"] = pd.to_datetime(events.entry_date).dt.date
    # Build sector daily equal-weight close-to-close returns and breadth.
    frames = []
    for code, df in data.items():
        sector = sectors.get(code, "未知")
        d = df.copy()
        d["sector"] = sector
        d["ret"] = d.close.pct_change()
        d["above"] = (d.close > d.ma20).astype(float)
        frames.append(d[["trade_date", "sector", "ret", "above"]])
    all_df = pd.concat(frames, ignore_index=True)
    sector_daily = (
        all_df.groupby(["trade_date", "sector"])
        .agg(ret=("ret", "mean"), breadth=("above", "mean"), n=("above", "size"))
        .reset_index()
    )
    # 20-day sector momentum and breadth at event date.
    sector_daily = sector_daily.sort_values(["sector", "trade_date"])
    sector_daily["sector_ret20"] = sector_daily.groupby("sector")["ret"].transform(
        lambda s: (1 + s.fillna(0)).rolling(20).apply(np.prod, raw=True) - 1
    )
    sector_daily["sector_breadth20"] = sector_daily.groupby("sector")["breadth"].transform(
        lambda s: s.rolling(20).mean()
    )
    key = sector_daily.set_index(["trade_date", "sector"])
    rows = []
    for _, event in events.iterrows():
        try:
            row = key.loc[(event.entry_date, event.sector)]
        except KeyError:
            continue
        rows.append(
            {
                **event.to_dict(),
                "sector_ret20": float(row.sector_ret20) if np.isfinite(row.sector_ret20) else None,
                "sector_breadth20": float(row.sector_breadth20)
                if np.isfinite(row.sector_breadth20)
                else None,
                "sector_n": int(row.n),
            }
        )
    enriched = pd.DataFrame(rows)
    enriched.to_csv(OUT / "c11_events_sector_strength.csv", index=False)
    if enriched.empty:
        print("no enriched events")
        return
    strong = enriched[enriched.sector_ret20 > 0]
    weak = enriched[enriched.sector_ret20 <= 0]
    broad = enriched[enriched.sector_breadth20 >= 0.5]
    narrow = enriched[enriched.sector_breadth20 < 0.5]
    summary = {
        "events": len(enriched),
        "all": {
            "n": len(enriched),
            "win": round(float((enriched.return_pct > 0).mean()), 3),
            "mean": round(float(enriched.return_pct.mean()), 2),
            "median": round(float(enriched.return_pct.median()), 2),
        },
        "sector_ret20_positive": {
            "n": len(strong),
            "win": round(float((strong.return_pct > 0).mean()), 3),
            "mean": round(float(strong.return_pct.mean()), 2),
            "median": round(float(strong.return_pct.median()), 2),
        },
        "sector_ret20_negative": {
            "n": len(weak),
            "win": round(float((weak.return_pct > 0).mean()), 3),
            "mean": round(float(weak.return_pct.mean()), 2),
            "median": round(float(weak.return_pct.median()), 2),
        },
        "sector_breadth_ge_50": {
            "n": len(broad),
            "win": round(float((broad.return_pct > 0).mean()), 3),
            "mean": round(float(broad.return_pct.mean()), 2),
            "median": round(float(broad.return_pct.median()), 2),
        },
        "sector_breadth_lt_50": {
            "n": len(narrow),
            "win": round(float((narrow.return_pct > 0).mean()), 3),
            "mean": round(float(narrow.return_pct.mean()), 2),
            "median": round(float(narrow.return_pct.median()), 2),
        },
        "top_events": enriched.sort_values("return_pct", ascending=False)
        .head(10)
        .to_dict("records"),
    }
    (OUT / "c11_sector_strength_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
