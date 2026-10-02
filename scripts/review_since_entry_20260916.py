"""Current fixed V4 since the earliest recorded entry date, not personal performance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import replay_next_open

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/since_entry_20260916"
START, END = date(2026, 7, 28), date(2026, 9, 15)
INITIAL = 100000.0


def summarize_trades(
    trades: list[dict[str, Any]], last_prices: dict[str, float], *, cost: float
) -> dict[str, Any]:
    """Independently reconcile replay legs to cash, realized and unrealized P&L."""
    cash, paid, realized = INITIAL, 0.0, 0.0
    holding, shares, basis = None, 0, 0.0
    enriched = []
    for trade in trades:
        leg = dict(trade)
        code, price = leg["code"], leg["price"]
        if leg["action"] == "buy":
            if holding is not None:
                raise ValueError("buy with existing holding")
            holding, shares = code, leg["shares"]
            fee = shares * price * cost
            basis = shares * price + fee
            cash -= basis
        elif leg["action"] == "sell":
            if holding != code:
                raise ValueError("unmatched sell")
            fee = shares * price * cost
            proceeds = shares * price - fee
            leg["realized_pnl"] = proceeds - basis
            realized += leg["realized_pnl"]
            cash += proceeds
            leg["shares"] = shares
            holding, shares, basis = None, 0, 0.0
        else:
            raise ValueError("unknown action")
        paid += fee
        leg["estimated_cost"] = fee
        leg["cash_after"] = cash
        enriched.append(leg)
    market_value = shares * last_prices[holding] if holding else 0.0
    unrealized = market_value - basis
    assert math.isclose(realized + unrealized, cash + market_value - INITIAL, abs_tol=1e-7)
    return {
        "cash": cash,
        "holding": holding,
        "shares": shares,
        "market_value": market_value,
        "ending_equity": cash + market_value,
        "estimated_cost_paid": paid,
        "realized_pnl": realized,
        "open_position_pnl_after_buy_cost": unrealized,
        "total_pnl": cash + market_value - INITIAL,
        "trades": enriched,
    }


def main() -> None:
    original, original_live = dict(rq.ETF_POOL), dict(live.ETF_POOL)
    paths = {c: OUT / "inputs" / f"{c}.parquet" for c in [*original, rq.DEFENSE]}
    data = {c: prepare_bars(pd.read_parquet(p), c, END) for c, p in paths.items()}
    dates = sorted(set.intersection(*(set(d.trade_date) for d in data.values())))
    assert START in dates and END in dates
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    last_prices = {c: float(d.close.iloc[-1]) for c, d in data.items()}
    report: dict[str, Any] = {
        "start": str(START),
        "end": str(END),
        "initial_capital": INITIAL,
        "start_basis": "Earliest recorded trade date; actual first-ever investment not confirmed",
        "initial_state": "Fresh cash only; no personal fills, peak, cooldown or pending orders",
        "mode": "V4",
        "v4_params": asdict(live.v4.V4_PARAMS),
        "momentum_periods": rq.MOM_PERIODS,
        "momentum_weights": rq.MOM_WEIGHTS,
        "grid_anchor": str(dates[130]),
        "force_entry_on_start": False,
        "single_side_cost": rq.FEE + rq.SLIPPAGE,
        "execution": (
            "T close signal; next common observed day open; final holdings marked to close"
        ),
        "strategy_version_caveat": (
            "Current V4 applied retrospectively, not historical deployed versions"
        ),
        "input_sha256": {c: hashlib.sha256(p.read_bytes()).hexdigest() for c, p in paths.items()},
        "pool": original,
        "scenarios": {},
    }
    try:
        for name, pool in {
            "current_original_pool": original,
            "exclude_inaccessible_501018": {c: n for c, n in original.items() if c != "501018"},
        }.items():
            rq.ETF_POOL.clear()
            rq.ETF_POOL.update(pool)
            live.ETF_POOL.clear()
            live.ETF_POOL.update(pool)
            scenario: dict[str, Any] = {}
            for multiplier in (1, 2, 3):
                replay = replay_next_open(
                    data,
                    start=START,
                    end=END,
                    grid=grid,
                    calendar=dates,
                    cost_multiplier=multiplier,
                )
                accounting = summarize_trades(
                    replay["trades"],
                    last_prices,
                    cost=(rq.FEE + rq.SLIPPAGE) * multiplier,
                )
                assert math.isclose(
                    accounting["ending_equity"],
                    replay["metrics"]["final_equity"],
                    abs_tol=1e-7,
                )
                assert replay["curve"].iloc[0].equity == INITIAL
                assert all(str(START) <= t["signal_date"] < t["date"] for t in replay["trades"])
                if name == "exclude_inaccessible_501018":
                    assert all(t["code"] != "501018" for t in replay["trades"])
                scenario[f"cost_{multiplier}x"] = {**replay["metrics"], **accounting}
                if multiplier == 1:
                    replay["curve"].to_csv(OUT / f"{name}_curve.csv", index=False)
                    scenario["unfilled_next_open"] = replay["unfilled_next_open"]
            report["scenarios"][name] = scenario
    finally:
        rq.ETF_POOL.clear()
        rq.ETF_POOL.update(original)
        live.ETF_POOL.clear()
        live.ETF_POOL.update(original_live)
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(
        json.dumps(
            {k: v["cost_1x"] for k, v in report["scenarios"].items()}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
