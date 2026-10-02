"""Fixed-parameter oil-proxy comparison from frozen bars; research only, no deployment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_rotation_20260916 import replay_next_open

OUT = Path(__file__).resolve().parents[1] / "artifacts/oil_proxy_20260916"
CUTOFF = date(2026, 9, 15)
# Economic shortlist, fixed before viewing replay results. Broad energy funds are not pure oil.
PROXIES = {"513350": "标普油气ETF富国", "561360": "石油ETF国泰", "159697": "石油ETF鹏华"}


def prepare_bars(frame: pd.DataFrame, code: str, cutoff: date) -> pd.DataFrame:
    frame = frame.rename(columns={"date": "trade_date"}).copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date).dt.date
    frame = frame.loc[frame.trade_date <= cutoff].sort_values("trade_date").reset_index(drop=True)
    if frame.trade_date.duplicated().any():
        raise ValueError(f"{code}: duplicate dates")
    if frame.empty or frame.trade_date.iloc[-1] != cutoff:
        raise ValueError(f"{code}: missing cutoff observation")
    prices = frame[["open", "high", "low", "close"]].astype(float)
    if (
        not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
        or (prices.high < prices[["open", "close", "low"]].max(axis=1)).any()
        or (prices.low > prices[["open", "close", "high"]].min(axis=1)).any()
        or not np.isfinite(frame.volume.to_numpy()).all()
        or (frame.volume < 0).any()
    ):
        raise ValueError(f"{code}: invalid OHLC/volume")
    frame["symbol"] = code
    return frame


def main() -> None:
    original, original_live = dict(rq.ETF_POOL), dict(live.ETF_POOL)
    codes = list(dict.fromkeys([*original, rq.DEFENSE, *PROXIES]))
    data = {
        c: prepare_bars(pd.read_parquet(OUT / "inputs" / f"{c}.parquet"), c, CUTOFF) for c in codes
    }
    original_dates = sorted(
        set.intersection(*(set(data[c].trade_date) for c in [*original, rq.DEFENSE]))
    )
    shared = sorted(set.intersection(*(set(data[c].trade_date) for c in codes)))
    grid = set(original_dates[130::5])
    start, end = shared[130], shared[-1]
    scenarios = {
        "original": original,
        "exclude_501018": {c: n for c, n in original.items() if c != "501018"},
    }
    for proxy, label in PROXIES.items():
        scenarios[f"replace_with_{proxy}"] = {
            (proxy if c == "501018" else c): (label if c == "501018" else n)
            for c, n in original.items()
        }
    report: dict[str, Any] = {
        "cutoff": str(CUTOFF),
        "start": str(start),
        "end": str(end),
        "grid_anchor": str(original_dates[130]),
        "v4_params": asdict(live.v4.V4_PARAMS),
        "momentum_periods": rq.MOM_PERIODS,
        "momentum_weights": rq.MOM_WEIGHTS,
        "initial_simulated_cash": 100000,
        "single_side_cost": rq.FEE + rq.SLIPPAGE,
        "broker_eligibility": "UNVERIFIED",
        "deployment": "RESEARCH_ONLY",
        "execution": "T close signal, next common observed day open, integer lots",
        "a_share_filter": "Original filter applies only to 159915; not extended or optimized",
        "input_sha256": {
            c: hashlib.sha256((OUT / "inputs" / f"{c}.parquet").read_bytes()).hexdigest()
            for c in codes
        },
        "data_audit": {},
        "proxy_diagnostics": {},
        "scenarios": {},
    }
    union = set.union(*(set(d.trade_date) for d in data.values()))
    for c, frame in data.items():
        missing = sorted(d for d in union - set(frame.trade_date) if start <= d <= end)
        report["data_audit"][c] = {
            "rows": len(frame),
            "first": str(frame.trade_date.iloc[0]),
            "missing_in_window": list(map(str, missing)),
        }
    for c in PROXIES:
        frame = data[c]
        # Pair equal return intervals only; a missing bar must not turn one side into 2 days.
        returns = {}
        previous_dates = {}
        for code in ["501018", c]:
            bars = data[code].set_index("trade_date")
            returns[code] = bars.close.pct_change()
            previous_dates[code] = pd.Series(bars.index, index=bars.index).shift()
        pair = pd.DataFrame(returns).loc[start:end].dropna()
        intervals = pd.DataFrame(previous_dates).reindex(pair.index)
        pair = pair.loc[intervals["501018"] == intervals[c]]
        recent = frame.tail(20)
        report["proxy_diagnostics"][c] = {
            "corr_daily_price_return_501018": float(pair.corr().to_numpy()[0, 1]),
            "paired_observations": len(pair),
            "median_recent20_turnover_yuan": float(recent.amount.median()),
            "last_close": float(frame.close.iloc[-1]),
        }
    try:
        for name, pool in scenarios.items():
            rq.ETF_POOL.clear()
            rq.ETF_POOL.update(pool)
            live.ETF_POOL.clear()
            live.ETF_POOL.update(pool)
            entry: dict[str, Any] = {}
            for multiplier in (1.0, 2.0, 3.0):
                result = replay_next_open(
                    data,
                    start=start,
                    end=end,
                    grid=grid,
                    cost_multiplier=multiplier,
                    calendar=shared,
                )
                entry[f"cost_{multiplier:g}x"] = result["metrics"]
                buys: dict[str, str] = {}
                for trade in result["trades"]:
                    assert trade["signal_date"] < trade["date"]
                    if trade["action"] == "buy":
                        buys[trade["code"]] = trade["date"]
                    elif trade["code"] in {"561360", "159697", "159915"}:
                        assert buys[trade["code"]] < trade["date"], "Domestic ETF T+1 violation"
                if multiplier == 1:
                    curve = result["curve"]
                    curve.to_csv(OUT / f"{name}_next_open.csv", index=False)
                    (OUT / f"{name}_trades.json").write_text(
                        json.dumps(result["trades"], ensure_ascii=False, indent=2)
                    )
                    entry["trade_legs"] = len(result["trades"])
                    entry["last_trades"] = result["trades"][-6:]
                    entry["unfilled_next_open"] = result["unfilled_next_open"]
                    previous, annual = 100000.0, {}
                    for year, group in curve.groupby(curve.trade_date.map(lambda d: d.year)):
                        final = float(group.equity.iloc[-1])
                        annual[str(year)] = final / previous - 1
                        previous = final
                    entry["calendar_year_returns_partial_edges"] = annual
            idx = {c: len(data[c]) - 1 for c in [*pool, rq.DEFENSE]}
            target, candidates, _, _ = live.select_target(data, idx, None)
            entry["latest_close_ranking_not_order"] = {"target": target, "candidates": candidates}
            report["scenarios"][name] = entry
    finally:
        rq.ETF_POOL.clear()
        rq.ETF_POOL.update(original)
        live.ETF_POOL.clear()
        live.ETF_POOL.update(original_live)
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({n: v["cost_1x"] for n, v in report["scenarios"].items()}, indent=2))


if __name__ == "__main__":
    main()
