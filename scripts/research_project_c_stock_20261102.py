"""Project C3: A-share limit-up continuation, 200-stock random sample.

Research only. Current-listed sample -> survivorship bias; T+1; no leverage.
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
MAX_POS = 2
HOLD_DAYS = 1  # buy T+1 open, sell T+2 open
LIMIT_TOL = 0.005


def load_stocks() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(CACHE.glob("*.parquet")):
        if path.stem in {"sample_200"}:
            continue
        df = pd.read_parquet(path)
        if "trade_date" not in df.columns or len(df) < 200:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        data[path.stem] = df
    return data


def limit_pct(code: str) -> float:
    if code.startswith(("sz30", "sh688")):
        return 0.20
    return 0.10


def metrics(curve: pd.DataFrame) -> dict[str, float | int]:
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
    data = load_stocks()
    codes = sorted(data)
    dates = sorted(set.union(*(set(df.trade_date) for df in data.values())))
    maps = {code: {d: i for i, d in enumerate(df.trade_date)} for code, df in data.items()}
    last_close: dict[str, float] = {}
    cash = INITIAL
    positions: dict[str, dict] = {}
    pending_buys: list[tuple[str, str]] = []  # (code, exit_date)
    pending_sell: set[str] = set()
    curve: list[dict] = []
    peak = INITIAL
    stops = 0
    events = 0
    fills = 0

    def price(code: str, td, field: str) -> float | None:
        idx = maps[code].get(td)
        if idx is None:
            return None
        return float(data[code].iloc[idx][field])

    for i, td in enumerate(dates):
        # sells first (limit-down open cannot fill; retry next day)
        for code in list(pending_sell):
            if code not in positions:
                pending_sell.discard(code)
                continue
            open_px = price(code, td, "open")
            prev_close = last_close.get(code)
            if open_px is None or prev_close is None:
                continue
            if open_px <= prev_close * (1 - limit_pct(code)) + 0.001:
                continue
            cash += positions[code]["shares"] * open_px * (1 - SELL_COST)
            del positions[code]
            pending_sell.discard(code)

        # buys at open, skip one-word limit-up
        for code, exit_date in list(pending_buys):
            if code in positions:
                pending_buys.remove((code, exit_date))
                continue
            open_px = price(code, td, "open")
            prev_close = last_close.get(code)
            if open_px is None or prev_close is None:
                continue
            if open_px >= prev_close * (1 + limit_pct(code)) - 0.001:
                pending_buys.remove((code, exit_date))
                continue
            slots = MAX_POS - len(positions)
            if slots <= 0:
                break
            budget = cash / slots
            shares = int(budget / open_px / 100) * 100
            if shares <= 0:
                pending_buys.remove((code, exit_date))
                continue
            cash -= shares * open_px * (1 + BUY_COST)
            positions[code] = {"shares": shares, "exit_date": exit_date}
            pending_buys.remove((code, exit_date))
            fills += 1

        # mark to market and update last close
        for code, df in data.items():
            idx = maps[code].get(td)
            if idx is not None:
                last_close[code] = float(df.iloc[idx].close)
        equity = cash + sum(
            pos["shares"] * last_close.get(code, 0.0) for code, pos in positions.items()
        )
        peak = max(peak, equity)
        if equity / peak - 1 <= -0.10 and not pending_sell:
            pending_sell = set(positions)
            stops += 1
            peak = equity

        # scheduled exits
        for code in list(positions):
            if positions[code]["exit_date"] == str(td):
                pending_sell.add(code)

        # new signals at close: limit-up, volume expansion, not held
        candidates: list[tuple[float, str]] = []
        for code in codes:
            idx = maps[code].get(td)
            if idx is None or idx < 61:
                continue
            if code in positions or any(item[0] == code for item in pending_buys):
                continue
            df = data[code]
            close = float(df.iloc[idx].close)
            prev = float(df.iloc[idx - 1].close)
            high = float(df.iloc[idx].high)
            if prev <= 0:
                continue
            change = close / prev - 1
            if change < limit_pct(code) - LIMIT_TOL:
                continue
            if close < high * 0.999:
                continue
            vol = float(df.iloc[idx].volume)
            avg = float(df.iloc[idx - 20 : idx].volume.mean())
            if avg <= 0 or vol / avg < 1.5:
                continue
            candidates.append((vol / avg, code))
        candidates.sort(reverse=True)
        if not pending_sell:
            for _ratio, code in candidates[: max(0, MAX_POS - len(positions))]:
                buy_idx = i + 1
                exit_idx = buy_idx + HOLD_DAYS
                if exit_idx < len(dates):
                    pending_buys.append((code, str(dates[exit_idx])))
                    events += 1

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
        {"stops": stops, "events": events, "fills": fills, "stocks": len(codes), "days": len(frame)}
    )
    payload = {
        "project": "Project C3",
        "candidate": "A-share limit-up continuation, random 200-stock sample",
        "constraints": {
            "capital": INITIAL,
            "leverage": "none",
            "max_positions": MAX_POS,
            "target_monthly": [0.12, 0.30],
        },
        "result": result,
        "bias": "Current-listed sample; survivorship bias; no delisted stocks; no minute data.",
    }
    (OUT / "c3_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(OUT / "c3_curve.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
