"""Project C: no-leverage, A-share/ETF high-return candidate research.

Candidate C1: weekly cross-sectional momentum on high-vol ETFs, top-2 equal
weight, hard -10% portfolio drawdown gate. No broker, no orders, research only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_qixing_v3 as rq  # noqa: E402

DATA = ROOT / "data/cross_asset"
OUT = ROOT / "artifacts/project_c_20261102"
INITIAL = 10_000.0
COST = 0.0015
MAX_DD = -0.10
LOOKBACK = 20
TOP_N = 2
EXTRA = {
    "513500": "标普500ETF",
    "511260": "十年国债ETF",
    "513030": "德国ETF",
    "510900": "恒生中国企业ETF",
    "159981": "能源化工ETF",
}


def load(codes: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for code in codes:
        path = DATA / f"{code}.parquet"
        if path.exists():
            data[code] = pd.read_parquet(path).sort_values("trade_date").reset_index(drop=True)
    return data


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


def run_c1(data: dict[str, pd.DataFrame], codes: list[str], defense: str) -> dict:
    dates = sorted(set.intersection(*(set(data[c].trade_date) for c in [*codes, defense])))
    start = dates[130]
    end = dates[-1]
    maps = {c: {d: i for i, d in enumerate(data[c].trade_date)} for c in [*codes, defense]}
    cash = INITIAL
    holdings: dict[str, int] = {}
    pending: list[str] | None = None
    curve: list[dict] = []
    peak = INITIAL
    stops = 0
    stop_until: int | None = None

    for day_index, td in enumerate(dates):
        if td < start:
            continue
        open_px = {
            c: float(data[c].iloc[maps[c][td]].open) for c in [*codes, defense] if td in maps[c]
        }
        close_px = {
            c: float(data[c].iloc[maps[c][td]].close) for c in [*codes, defense] if td in maps[c]
        }

        if pending is not None:
            equity_open = cash + sum(shares * open_px.get(c, 0.0) for c, shares in holdings.items())
            targets = [c for c in pending if c in open_px and open_px[c] > 0]
            for code in list(holdings):
                if code not in targets:
                    cash += holdings[code] * open_px.get(code, 0.0) * (1 - COST)
                    del holdings[code]
            if targets:
                target_value = equity_open / len(targets)
                for code in targets:
                    price = open_px[code]
                    current = holdings.get(code, 0) * price
                    delta = target_value - current
                    if delta > price * 100:
                        shares = int(delta / price / 100) * 100
                        if shares:
                            cash -= shares * price * (1 + COST)
                            holdings[code] = holdings.get(code, 0) + shares
                    elif delta < -price * 100 and holdings.get(code):
                        shares = min(holdings[code], int(abs(delta) / price / 100) * 100)
                        if shares:
                            cash += shares * price * (1 - COST)
                            holdings[code] -= shares
                            if holdings[code] == 0:
                                del holdings[code]
            pending = None

        equity = cash + sum(shares * close_px.get(c, 0.0) for c, shares in holdings.items())
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if drawdown <= MAX_DD and pending is None:
            pending = [defense]
            stops += 1
            stop_until = day_index + 5
            peak = equity  # V5-style reset after de-risking; research convention.

        # Weekly signal: every 5 trading days, after close, execute next open.
        if day_index % 5 == 0 and pending is None:
            ranked: list[tuple[str, float]] = []
            for code in codes:
                if code not in close_px:
                    continue
                df = data[code]
                idx = maps[code][td]
                if idx < LOOKBACK:
                    continue
                close = float(df.iloc[idx].close)
                past = float(df.iloc[idx - LOOKBACK].close)
                if past <= 0:
                    continue
                mom = close / past - 1
                if mom > 0:
                    ranked.append((code, mom))
            ranked.sort(key=lambda item: -item[1])
            if stop_until is not None and day_index < stop_until:
                pending = [defense]
            else:
                pending = [code for code, _ in ranked[:TOP_N]] or [defense]

        curve.append(
            {
                "trade_date": td,
                "equity": equity,
                "cash": cash,
                "holding": ",".join(sorted(holdings)) or None,
            }
        )

    frame = pd.DataFrame(curve)
    result = metrics(frame)
    result.update({"trades_stops": stops, "start": str(start), "end": str(end), "days": len(frame)})
    return {"metrics": result, "curve": frame}


def run_c2(data: dict[str, pd.DataFrame], codes: list[str], defense: str) -> dict:
    """C2: daily 20-day breakout, top-1, 10-day MA exit, -10% hard gate."""
    dates = sorted(set.intersection(*(set(data[c].trade_date) for c in [*codes, defense])))
    start = dates[130]
    maps = {c: {d: i for i, d in enumerate(data[c].trade_date)} for c in [*codes, defense]}
    cash = INITIAL
    holdings: dict[str, int] = {}
    pending: list[str] | None = None
    curve: list[dict] = []
    peak = INITIAL
    hold_days = 0
    stops = 0

    def close_of(code: str, td) -> float | None:
        idx = maps[code].get(td)
        return None if idx is None else float(data[code].iloc[idx].close)

    for _day_index, td in enumerate(dates):
        if td < start:
            continue
        open_px = {
            c: float(data[c].iloc[maps[c][td]].open) for c in [*codes, defense] if td in maps[c]
        }
        close_px = {
            c: float(data[c].iloc[maps[c][td]].close) for c in [*codes, defense] if td in maps[c]
        }

        if pending is not None:
            equity_open = cash + sum(shares * open_px.get(c, 0.0) for c, shares in holdings.items())
            targets = [c for c in pending if c in open_px and open_px[c] > 0]
            for code in list(holdings):
                if code not in targets:
                    cash += holdings[code] * open_px.get(code, 0.0) * (1 - COST)
                    del holdings[code]
            if targets:
                per = equity_open / len(targets)
                for code in targets:
                    price = open_px[code]
                    shares = int(per / price / 100) * 100
                    if shares:
                        cash -= shares * price * (1 + COST)
                        holdings[code] = holdings.get(code, 0) + shares
            pending = None
            hold_days = 0

        equity = cash + sum(shares * close_px.get(c, 0.0) for c, shares in holdings.items())
        peak = max(peak, equity)
        if equity / peak - 1 <= MAX_DD and pending is None:
            pending = [defense]
            stops += 1
            peak = equity
            hold_days = 0

        if pending is None:
            current = next((c for c in holdings if c != defense), None)
            if current:
                hold_days += 1
                idx = maps[current].get(td)
                closes = data[current].iloc[: idx + 1].close.astype(float).to_numpy()
                ma10 = float(closes[-10:].mean()) if len(closes) >= 10 else closes[-1]
                if hold_days >= 10 or float(closes[-1]) < ma10:
                    pending = [defense]
            else:
                ranked: list[tuple[str, float]] = []
                for code in codes:
                    idx = maps[code].get(td)
                    if idx is None or idx < 21:
                        continue
                    closes = data[code].iloc[: idx + 1].close.astype(float).to_numpy()
                    prior_high = float(closes[-21:-1].max())
                    if float(closes[-1]) > prior_high and closes[-1] > closes[-21]:
                        ranked.append((code, float(closes[-1] / closes[-21] - 1)))
                ranked.sort(key=lambda item: -item[1])
                pending = [ranked[0][0]] if ranked else [defense]

        curve.append(
            {
                "trade_date": td,
                "equity": equity,
                "cash": cash,
                "holding": ",".join(sorted(holdings)) or None,
            }
        )

    frame = pd.DataFrame(curve)
    result = metrics(frame)
    result.update(
        {"trades_stops": stops, "start": str(start), "end": str(dates[-1]), "days": len(frame)}
    )
    return {"metrics": result, "curve": frame}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    codes = [*rq.ETF_POOL, *[c for c in EXTRA if c not in rq.ETF_POOL]]
    data = load([*codes, rq.DEFENSE])
    c1 = run_c1(data, codes, rq.DEFENSE)
    c2 = run_c2(data, codes, rq.DEFENSE)
    payload = {
        "project": "Project C",
        "candidate": "C1: weekly top-2 cross-sectional momentum + -10% hard drawdown gate",
        "constraints": {
            "capital": INITIAL,
            "instruments": "A-share ETFs/LOFs only (research cache)",
            "leverage": "none",
            "max_drawdown_gate": MAX_DD,
            "target_monthly": [0.12, 0.30],
        },
        "cost_per_side": COST,
        "result": c1["metrics"],
        "verdict": "C1 is a first falsification candidate, not a promise.",
        "c2": {
            "candidate": "C2: daily 20-day breakout, top-1, 10-day MA exit, -10% hard gate",
            "result": c2["metrics"],
        },
    }
    (OUT / "c1_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    c1["curve"].to_csv(OUT / "c1_curve.csv", index=False)
    c2["curve"].to_csv(OUT / "c2_curve.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
