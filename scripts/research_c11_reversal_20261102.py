"""C11 sector reversal + monthly cluster analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/stock_cache"
OUT = ROOT / "artifacts/project_c_20261102"


def main() -> None:
    sectors = dict(pd.read_csv(CACHE / "sector_map.csv").values)
    frames = []
    for path in sorted(CACHE.glob("*.parquet")):
        if path.stem in {"sample_200", "sample_500"}:
            continue
        df = pd.read_parquet(path)
        if "trade_date" not in df.columns or len(df) < 200:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df.close.astype(float)
        df["ma20"] = close.rolling(20).mean()
        df["ret"] = close.pct_change()
        df["above"] = (close > df.ma20).astype(float)
        df["sector"] = sectors.get(path.stem, "未知")
        frames.append(df[["trade_date", "sector", "ret", "above"]])
    all_df = pd.concat(frames, ignore_index=True)
    daily = (
        all_df.groupby(["trade_date", "sector"])
        .agg(ret=("ret", "mean"), breadth=("above", "mean"))
        .reset_index()
        .sort_values(["sector", "trade_date"])
    )
    for window in (5, 20):
        daily[f"ret{window}"] = daily.groupby("sector")["ret"].transform(
            lambda s, w=window: (1 + s.fillna(0)).rolling(w).apply(np.prod, raw=True) - 1
        )
    daily["breadth_chg5"] = daily.groupby("sector")["breadth"].transform(lambda s: s - s.shift(5))
    key = daily.set_index(["trade_date", "sector"])
    events = pd.read_csv(OUT / "c11_events_500.csv")
    events["entry_date"] = pd.to_datetime(events.entry_date).dt.date
    rows = []
    for _, event in events.iterrows():
        try:
            row = key.loc[(event.entry_date, event.sector)]
        except KeyError:
            continue
        rows.append(
            {
                **event.to_dict(),
                "sector_ret5": float(row.ret5),
                "sector_ret20": float(row.ret20),
                "sector_breadth_chg5": float(row.breadth_chg5)
                if np.isfinite(row.breadth_chg5)
                else None,
            }
        )
    enriched = pd.DataFrame(rows)
    enriched.to_csv(OUT / "c11_events_reversal.csv", index=False)
    # Monthly independent signals.
    enriched["month"] = pd.to_datetime(enriched.entry_date).dt.to_period("M").astype(str)
    monthly = (
        enriched.groupby("month")
        .agg(
            signals=("code", "size"),
            unique=("code", "nunique"),
            win=("return_pct", lambda x: round(float((x > 0).mean()), 3)),
            mean=("return_pct", lambda x: round(float(x.mean()), 2)),
        )
        .reset_index()
    )
    summary = {
        "events": len(enriched),
        "all": {
            "n": len(enriched),
            "win": round(float((enriched.return_pct > 0).mean()), 3),
            "mean": round(float(enriched.return_pct.mean()), 2),
            "median": round(float(enriched.return_pct.median()), 2),
        },
        "sector_reversal": {
            "neg20_pos5_n": int(((enriched.sector_ret20 < 0) & (enriched.sector_ret5 > 0)).sum()),
            "neg20_pos5_mean": round(
                float(
                    enriched[
                        (enriched.sector_ret20 < 0) & (enriched.sector_ret5 > 0)
                    ].return_pct.mean()
                ),
                2,
            )
            if ((enriched.sector_ret20 < 0) & (enriched.sector_ret5 > 0)).any()
            else None,
            "neg20_pos5_win": round(
                float(
                    (
                        enriched[
                            (enriched.sector_ret20 < 0) & (enriched.sector_ret5 > 0)
                        ].return_pct
                        > 0
                    ).mean()
                ),
                3,
            )
            if ((enriched.sector_ret20 < 0) & (enriched.sector_ret5 > 0)).any()
            else None,
            "neg20_breadth_up_n": int(
                ((enriched.sector_ret20 < 0) & (enriched.sector_breadth_chg5 > 0)).sum()
            ),
        },
        "monthly": monthly.to_dict("records"),
        "max_unique_month": int(monthly.unique.max()) if len(monthly) else 0,
        "months_with_2plus_unique": int((monthly.unique >= 2).sum()) if len(monthly) else 0,
    }
    (OUT / "c11_reversal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
