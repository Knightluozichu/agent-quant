"""Project C6: <10bn market cap, range low buy, scissors (MA/MACD) exit, stop loss.

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
MAX_POS = 4
MAX_MCAP = 3_000_000_000.0
RANGE_60 = 0.25
TREND_60 = 0.10
STOP_LOSS = -0.07
MAX_HOLD = 30
DRAWDOWN_60 = -0.15


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
        df["ma5"] = close.rolling(5).mean()
        df["ma10"] = close.rolling(10).mean()
        df["ma20"] = close.rolling(20).mean()
        df["prior5_high"] = high.shift(1).rolling(5).max()
        df["ma60"] = close.rolling(60).mean()
        df["low60"] = low.rolling(60).min()
        df["high60"] = high.rolling(60).max()
        df["range60"] = (df.high60 - df.low60) / close
        df["ret60"] = close / close.shift(60) - 1
        df["vol20"] = df.volume.rolling(20).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["dif"] = ema12 - ema26
        df["dea"] = df.dif.ewm(span=9, adjust=False).mean()
        df["hist"] = df.dif - df.dea
        df["prev_hist"] = df["hist"].shift(1)
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


def run(data: dict[str, pd.DataFrame], exit_mode: str) -> dict:
    codes = sorted(data)
    dates = sorted(set.union(*(set(df.trade_date) for df in data.values())))
    maps = {c: {d: i for i, d in enumerate(df.trade_date)} for c, df in data.items()}
    cash = INITIAL
    positions: dict[str, dict] = {}
    curve: list[dict] = []
    peak = INITIAL
    gate_next = False
    stops = 0
    trades = 0
    entries = 0
    candidate_stocks: set[str] = set()

    for i, td in enumerate(dates):
        # T+1 open exits
        for code in list(positions):
            if positions[code].get("exit_date") == str(td):
                idx = maps[code].get(td)
                if idx is None:
                    continue
                open_px = float(data[code].iloc[idx].open)
                if open_px > 0:
                    cash += positions[code]["shares"] * open_px * (1 - SELL_COST)
                    del positions[code]

        if gate_next:
            for code in list(positions):
                idx = maps[code].get(td)
                if idx is None:
                    continue
                open_px = float(data[code].iloc[idx].open)
                if open_px > 0:
                    cash += positions[code]["shares"] * open_px * (1 - SELL_COST)
                    del positions[code]
            gate_next = False
            peak = cash

        equity = cash + sum(
            pos["shares"] * float(data[code].iloc[maps[code][td]].close)
            for code, pos in positions.items()
            if td in maps[code]
        )
        peak = max(peak, equity)
        if equity / peak - 1 <= -0.10:
            gate_next = True
            stops += 1

        # exit signals at close, execute next open
        for code, pos in list(positions.items()):
            if pos.get("exit_date"):
                continue
            idx = maps[code].get(td)
            if idx is None or idx < 2:
                continue
            row = data[code].iloc[idx]
            prev = data[code].iloc[idx - 1]
            close = float(row.close)
            if close <= pos["entry_price"] * (1 + STOP_LOSS):
                pos["exit_date"] = str(dates[i + 1]) if i + 1 < len(dates) else None
                continue
            dead = (
                row.ma5 < row.ma10 and prev.ma5 >= prev.ma10
                if exit_mode == "ma"
                else row.dif < row.dea and prev.dif >= prev.dea
            )
            if dead or i - pos["entry_i"] >= MAX_HOLD:
                pos["exit_date"] = str(dates[i + 1]) if i + 1 < len(dates) else None

        # new entries at close if slots and not gated
        if not gate_next and i + 1 < len(dates):
            candidates = []
            for code in codes:
                idx = maps[code].get(td)
                if idx is None or idx < 61 or code in positions:
                    continue
                row = data[code].iloc[idx]
                if not np.isfinite(row.low60) or row.close <= 0 or row.mcap >= MAX_MCAP:
                    continue
                if row.close / row.high60 - 1 > DRAWDOWN_60:
                    continue
                if row.close >= 3:
                    continue
                if row.close > row.ma20:
                    continue
                if row.close <= row.ma10 or row.ma5 <= row.ma10:
                    continue
                if row.close <= row.prior5_high:
                    continue
                if row.close / row.high60 - 1 > -0.20 or row.close / row.high60 - 1 < -0.30:
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
                score = row.close / row.high60 - 1
                candidates.append((score, row.mcap, code, float(row.close)))
            candidates.sort()
            slots = MAX_POS - len(positions)
            selected = candidates[:slots]
            budget = cash / len(selected) if selected else 0.0
            for _loc, _mcap, code, close in selected:
                shares = int(budget / close / 100) * 100
                if shares <= 0:
                    continue
                cash -= shares * close * (1 + BUY_COST)
                positions[code] = {
                    "shares": shares,
                    "entry_price": close,
                    "entry_i": i,
                    "exit_date": None,
                }
                trades += 1
                entries += 1
                candidate_stocks.add(code)

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
    result.update(
        {
            "trades": trades,
            "entries": entries,
            "stops": stops,
            "stocks_in_sample": len(codes),
            "distinct_stocks_traded": len(candidate_stocks),
            "days": len(frame),
        }
    )
    return {"result": result, "curve": frame}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    payload = {
        "project": "Project C11",
        "constraints": {"capital": INITIAL, "max_positions": MAX_POS, "max_mcap": MAX_MCAP},
    }
    for mode in ("ma", "macd"):
        run_result = run(data, mode)
        payload[mode] = run_result["result"]
        run_result["curve"].to_csv(OUT / f"c11_{mode}_curve.csv", index=False)
    payload["bias"] = (
        "Current-listed sample; survivorship bias; same-close entry is a 14:50 production proxy."
    )
    (OUT / "c11_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
