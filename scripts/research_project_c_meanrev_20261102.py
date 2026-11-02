"""Project C5: small-cap range-bound mean reversion, buy near lower band, sell T+1.

Research only. Current-listed sample -> survivorship bias; no broker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/stock_cache"
OUT = ROOT / "artifacts/project_c_20261102"
INITIAL = 10_000.0
BUY_COST = 0.0015
SELL_COST = 0.0020
MAX_POS = 5
SMALL_CAP_PCT = 0.50  # bottom half by point-in-time market cap
RANGE_20 = 0.15
TREND_20 = 0.08
DROP_3 = -0.03


def load() -> dict[str, pd.DataFrame]:
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
        df["ma20"] = close.rolling(20).mean()
        df["ret20"] = close / close.shift(20) - 1
        df["range20"] = (high.rolling(20).max() - low.rolling(20).min()) / close
        df["ret3"] = close / close.shift(3) - 1
        df["low20"] = low.rolling(20).min()
        df["mcap"] = close * df.outstanding_share.astype(float)
        data[path.stem] = df
    return data


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
        "months_ge_12pct": int((monthly >= 0.12).sum()),
        "months_ge_20pct": int((monthly >= 0.20).sum()),
        "months_ge_30pct": int((monthly >= 0.30).sum()),
        "best_month": round(float(monthly.max()) * 100, 1) if len(monthly) else 0.0,
        "worst_month": round(float(monthly.min()) * 100, 1) if len(monthly) else 0.0,
        "monthly_median": round(float(monthly.median()) * 100, 2) if len(monthly) else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    codes = sorted(data)
    dates = sorted(set.union(*(set(df.trade_date) for df in data.values())))
    maps = {c: {d: i for i, d in enumerate(df.trade_date)} for c, df in data.items()}
    # Cross-sectional small-cap rank per date.
    mcap = pd.DataFrame({c: data[c].set_index("trade_date").mcap for c in codes}).sort_index()
    small = mcap.rank(axis=1, pct=True) <= SMALL_CAP_PCT
    cash = INITIAL
    positions: dict[str, dict] = {}
    curve = []
    peak = INITIAL
    gate_next = False
    stops = 0
    trades = 0

    for i, td in enumerate(dates):
        # scheduled T+1 close exits
        for code in list(positions):
            if positions[code]["exit_date"] != str(td):
                continue
            idx = maps[code].get(td)
            if idx is None:
                continue
            close = float(data[code].iloc[idx].close)
            cash += positions[code]["shares"] * close * (1 - SELL_COST)
            del positions[code]

        if gate_next:
            for code in list(positions):
                idx = maps[code].get(td)
                if idx is None:
                    continue
                close = float(data[code].iloc[idx].close)
                cash += positions[code]["shares"] * close * (1 - SELL_COST)
                del positions[code]
            gate_next = False
            peak = cash

        # mark equity before new entries
        equity = cash + sum(
            pos["shares"] * float(data[code].iloc[maps[code][td]].close)
            for code, pos in positions.items()
            if td in maps[code]
        )
        peak = max(peak, equity)
        if equity / peak - 1 <= -0.10:
            gate_next = True
            stops += 1

        # new candidates at close (same-close fill; 14:50 production proxy)
        has_small = td in small.index and bool(small.loc[td].any())
        if not gate_next and i + 1 < len(dates) and has_small:
            candidates = []
            for code in codes:
                idx = maps[code].get(td)
                if idx is None or idx < 21:
                    continue
                if code in positions:
                    continue
                row = data[code].iloc[idx]
                if not bool(small.loc[td, code]):
                    continue
                if not np.isfinite(row.ma20) or row.close <= 0:
                    continue
                if abs(row.ret20) > TREND_20 or row.range20 > RANGE_20:
                    continue
                if row.ret3 > DROP_3 or row.close > row.ma20:
                    continue
                if row.close < row.low20 * 1.01:
                    continue
                score = (row.close - row.low20) / max(row.close, 1e-9)
                candidates.append((score, code, float(row.close)))
            candidates.sort()
            slots = MAX_POS - len(positions)
            for _score, code, close in candidates[:slots]:
                slots_now = MAX_POS - len(positions)
                if slots_now <= 0:
                    break
                budget = cash / slots_now
                shares = int(budget / close / 100) * 100
                if shares <= 0:
                    continue
                cash -= shares * close * (1 + BUY_COST)
                positions[code] = {"shares": shares, "exit_date": str(dates[i + 1])}
                trades += 1

        equity = cash + sum(
            pos["shares"] * float(data[code].iloc[maps[code][td]].close)
            for code, pos in positions.items()
            if td in maps[code]
        )
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
    result.update({"trades": trades, "stops": stops, "stocks": len(codes), "days": len(frame)})
    payload = {
        "project": "Project C5",
        "candidate": "Small-cap range-bound pullback, buy close T, sell close T+1",
        "constraints": {
            "capital": INITIAL,
            "leverage": "none",
            "max_positions": MAX_POS,
            "target_monthly": [0.12, 0.30],
        },
        "filters": {
            "small_cap_pct": SMALL_CAP_PCT,
            "range20": RANGE_20,
            "abs_ret20": TREND_20,
            "ret3": DROP_3,
        },
        "result": result,
        "bias": (
            "Current-listed sample; survivorship bias; same-close entry is a "
            "14:50 production proxy."
        ),
    }
    (OUT / "c5_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(OUT / "c5_curve.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
