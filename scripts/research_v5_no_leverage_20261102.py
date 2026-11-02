"""No-leverage V5 iteration research: universe expansion, upper bounds, leverage reference.

Research only. No production writes, no broker, no orders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import qixing_v5 as v5  # noqa: E402
import review_rotation_20260916 as rr  # noqa: E402
import run_qixing_v3 as rq  # noqa: E402

DATA = ROOT / "data/cross_asset"
OUT = ROOT / "artifacts/v5_no_leverage_20261102"
BASE = dict(rq.ETF_POOL)
EXTRA = {
    "513500": "标普500ETF",
    "511260": "十年国债ETF",
    "513030": "德国ETF",
    "510900": "恒生中国企业ETF",
    "159981": "能源化工ETF",
}
INITIAL = 100_000.0


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
    return {
        "final_equity": round(float(equity[-1]), 2),
        "total_return": round(float(equity[-1] / INITIAL - 1) * 100, 1),
        "cagr": round(float((equity[-1] / INITIAL) ** (1 / years) - 1) * 100, 1) if years else 0.0,
        "sharpe": round(float(returns.mean() / vol * np.sqrt(252)), 2) if vol else 0.0,
        "max_drawdown": round(float((path / np.maximum.accumulate(path) - 1).min()) * 100, 1),
        "trades": 0,
        "years": round(years, 1),
    }


def run_v5(data: dict[str, pd.DataFrame], pool: dict[str, str], dates: list, start, end) -> dict:
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    with (
        patch.object(rr.rq, "ETF_POOL", dict(pool)),
        patch.object(rr.live, "ETF_POOL", dict(pool)),
        patch.object(rr.live, "POOL_VERSION", f"research-{len(pool)}"),
        patch.object(rr.live, "risk_assess", v5.assess_v5),
    ):
        result = rr.replay_next_open(
            data,
            start=start,
            end=end,
            grid=grid,
            calendar=dates,
            mode="V4",
            cost_multiplier=1.0,
        )
    result["metrics"] = {**metrics(result["curve"]), "trades": len(result["trades"])}
    return result


def perfect_yearly_bound(data: dict[str, pd.DataFrame], codes: list[str], start, end) -> dict:
    frame = pd.DataFrame(
        {code: data[code].set_index("trade_date")["close"] for code in codes}
    ).dropna()
    frame.index = pd.to_datetime(frame.index)
    frame = frame[
        (frame.index >= pd.Timestamp(str(start))) & (frame.index <= pd.Timestamp(str(end)))
    ]
    previous = frame.iloc[0]
    yearly: dict[int, float] = {}
    for year, group in frame.groupby(frame.index.year):
        end = group.iloc[-1]
        yearly[int(year)] = float((end / previous).max() - 1)
        previous = end
    compound = float(np.prod([1 + value for value in yearly.values()]) - 1)
    years = (frame.index[-1] - frame.index[0]).days / 365.25
    return {
        "years": round(years, 1),
        "yearly_best": {str(year): round(value * 100, 1) for year, value in yearly.items()},
        "compound_return": round(compound * 100, 1),
        "cagr": round(((1 + compound) ** (1 / years) - 1) * 100, 1),
    }


def leverage_reference(curve: pd.DataFrame) -> list[dict[str, float]]:
    returns = curve.equity.pct_change().fillna(0).to_numpy()
    years = (
        pd.to_datetime(curve.trade_date.iloc[-1]) - pd.to_datetime(curve.trade_date.iloc[0])
    ).days / 365.25
    rows = []
    for leverage in (1.0, 1.1, 1.2, 1.5):
        financing = 0.05 / 252 * (leverage - 1)
        equity = INITIAL * np.cumprod(1 + leverage * returns - financing)
        rows.append(
            {
                "leverage": leverage,
                "cagr": round(float((equity[-1] / INITIAL) ** (1 / years) - 1) * 100, 1),
                "max_drawdown": round(
                    float((equity / np.maximum.accumulate(equity) - 1).min()) * 100, 1
                ),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_codes = [*BASE, *[code for code in EXTRA if code not in BASE]]
    data = load([*all_codes, rq.DEFENSE])
    dates = sorted(
        set.intersection(*(set(data[code].trade_date) for code in [*all_codes, rq.DEFENSE]))
    )
    start, end = dates[130], dates[-1]

    pools = {
        "V5_8": BASE,
        "V5_10": {**BASE, **{code: EXTRA[code] for code in ("513500", "511260", "513030")}},
        "V5_12": {**BASE, **EXTRA},
    }
    results = {name: run_v5(data, pool, dates, start, end) for name, pool in pools.items()}
    bounds = {
        "V5_8": perfect_yearly_bound(data, list(BASE), start, end),
        "V5_12": perfect_yearly_bound(data, all_codes, start, end),
    }
    leverage = leverage_reference(results["V5_8"]["curve"])

    payload = {
        "start": str(start),
        "end": str(end),
        "days": len(dates),
        "data_source": "data/cross_asset Sina cache; local research",
        "cost_per_side": 0.0015,
        "results": {name: item["metrics"] for name, item in results.items()},
        "perfect_yearly": bounds,
        "leverage_reference_not_allowed": leverage,
        "yearly": {
            name: {
                str(year): round(float(group.equity.iloc[-1] / group.equity.iloc[0] - 1) * 100, 1)
                for year, group in item["curve"].groupby(
                    pd.to_datetime(item["curve"].trade_date).dt.year
                )
            }
            for name, item in results.items()
        },
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results["V5_8"]["curve"].to_csv(OUT / "curve_v5_8.csv", index=False)
    results["V5_12"]["curve"].to_csv(OUT / "curve_v5_12.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
