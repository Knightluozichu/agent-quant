"""Frozen-parameter research only: original / exclude oil / replace oil with 513350.

Next-open replay uses only data through signal close T, and fills at T+1 open.
No optimization, no broker, no production configuration or locked-test changes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq

if TYPE_CHECKING:
    from datetime import date

OUT = Path(__file__).resolve().parents[1] / "artifacts/reconciliation_20260916"


def metrics(curve: pd.DataFrame) -> dict[str, Any]:
    equity = curve.equity.to_numpy()
    returns = np.diff(np.r_[100000.0, equity]) / np.r_[100000.0, equity[:-1]]
    vol = float(returns.std(ddof=1))
    return {
        "start": str(curve.iloc[0].trade_date),
        "end": str(curve.iloc[-1].trade_date),
        "days": len(curve),
        "total_return": float(equity[-1] / 100000 - 1),
        "max_drawdown": float(
            np.min(equity / np.maximum.accumulate(np.r_[100000, equity])[1:] - 1)
        ),
        "sharpe_daily_rf0": float(returns.mean() / vol * np.sqrt(252)) if vol else 0.0,
        "final_equity": float(equity[-1]),
    }


def replay_next_open(
    data: dict[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    grid: set[date],
    cost_multiplier: float = 1.0,
    calendar: list[date] | None = None,
    mode: str = "V4",
    protector: Any | None = None,
) -> dict[str, Any]:
    if mode not in {"V3-G", "V4"}:
        raise ValueError("Research mode must be V3-G or V4")
    codes = [*live.ETF_POOL, live.DEFENSE]
    dates = sorted(set.intersection(*(set(data[c].trade_date) for c in codes)))
    if calendar is not None:
        if not set(calendar).issubset(dates):
            raise ValueError("Comparison calendar contains unavailable bars")
        dates = calendar
    dates = [d for d in dates if d <= end]
    maps: dict[str, dict[date, int]] = {
        c: dict(zip(data[c].trade_date, range(len(data[c])), strict=True)) for c in codes
    }
    state = live.default_state(100000.0)
    initial_state = copy.deepcopy(state)
    queued = None
    rows, trades, decisions = [], [], []
    for day_index, td in enumerate(dates):
        if td < start:
            continue
        if queued:
            target, exposure, early, signal_date = queued
            if state["holding"]:
                code = state["holding"]
                price = float(data[code].iloc[maps[code][td]].open)
                assert np.isfinite(price) and price > 0
                state["cash"] += (
                    state["shares"] * price * (1 - (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                )
                trades.append(
                    {
                        "date": str(td),
                        "signal_date": str(signal_date),
                        "action": "sell",
                        "code": code,
                        "price": price,
                    }
                )
                state.update(holding=None, shares=0, entry_price=0.0)
            price = float(data[target].iloc[maps[target][td]].open)
            assert np.isfinite(price) and price > 0
            shares = int(state["cash"] * exposure * 0.99 / price / 100) * 100
            if shares:
                state["cash"] -= shares * price * (1 + (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                state.update(holding=target, shares=shares, entry_price=price, entry_date=str(td))
                trades.append(
                    {
                        "date": str(td),
                        "signal_date": str(signal_date),
                        "action": "buy",
                        "code": target,
                        "price": price,
                        "shares": shares,
                    }
                )
                # Research lock starts on the fill day, not the signal/confirmation day.
                if early:
                    state["v4_state"]["last_early_rotation_date"] = str(td)
            state["risk_exposure"] = 1.0
            queued = None
        # Enforce causality at the input boundary, not just inside feature functions.
        history = {c: data[c].iloc[: maps[c][td] + 1] for c in codes}
        idx = {c: len(history[c]) - 1 for c in codes}
        equity = state["cash"]
        if state["holding"]:
            equity += state["shares"] * float(history[state["holding"]].iloc[-1].close)
        assert state["cash"] >= 0
        rows.append(
            {"trade_date": td, "equity": equity, "cash": state["cash"], "holding": state["holding"]}
        )
        state["peak_equity"] = max(state["peak_equity"], equity)
        # 账户级熔断 (H4 研究, 默认 None 零行为变化): T 收盘判定, T+1 开盘执行.
        if protector is not None:
            holding_value = (
                state["shares"] * float(history[state["holding"]].iloc[-1].close)
                if state["holding"]
                else 0.0
            )
            fired = protector.on_close(
                td, equity, state["holding"], holding_value, rq.DEFENSE
            )
            if fired:
                if state["holding"] and state["holding"] != rq.DEFENSE:
                    queued = (rq.DEFENSE, 1.0, False, td)
                continue  # 熔断日: 信号层全部屏蔽
            if not protector.entries_allowed(td):
                continue  # 冷却期: 不生成新轮动信号
        target, candidates, _, _ = live.select_target(history, idx, state["holding"])
        # Mirror the production intraday -3% gate using the observable close return.
        dropped = set()
        for code, _score in candidates:
            close = history[code].close.to_numpy()
            if close[-1] / close[-2] - 1 < -0.03 and (
                len(close) < 61
                or close[-1] / close[-61] - 1 >= 0.01
                or live.calc_momentum_score(close) <= 0
            ):
                dropped.add(code)
        if dropped:
            candidates = [(c, s) for c, s in candidates if c not in dropped]
            target = candidates[0][0] if candidates else live.DEFENSE
        visible_dates = dates[: day_index + 1]
        overlay = live.evaluate_v4_overlay(
            data=history,
            idx_map=idx,
            trading_dates=visible_dates,
            td=td,
            holding=state["holding"],
            base_target=target,
            candidates=candidates,
            scheduled_rebalance=td in grid,
            state=state,
            mode=mode,
        )
        spot = {
            c: {
                "price": float(history[c].iloc[-1].close),
                "prev_close": float(history[c].iloc[-2].close),
            }
            for c in codes
        }
        risk = live.risk_assess(
            target=overlay["target"],
            holding=state["holding"],
            state=state,
            data=history,
            td=td,
            idx_map=idx,
            is_rebalance=overlay["is_rebalance"],
            common_dates=visible_dates,
            spot_map=spot,
        )
        target = risk.final_target or live.DEFENSE
        snapshot = live._decision_snapshot(
            td=td,
            holding=state["holding"],
            candidates=candidates,
            overlay=overlay,
            risk=risk,
            final_target=target,
            spot_map=spot,
        )
        # No wall-clock timestamp in reproducible research output; ID excludes this field.
        snapshot["created_at"] = f"{td}T15:00:00+08:00"
        live._record_v4_state(state, overlay, snapshot)
        decisions.append(snapshot)
        if (overlay["is_rebalance"] or risk.action == live.ACTION_EMERGENCY) and target != state[
            "holding"
        ]:
            queued = (target, risk.exposure, mode == "V4" and overlay["decision"].triggered, td)
        state["risk_exposure"] = risk.exposure
        state["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
    curve = pd.DataFrame(rows)
    return {
        "curve": curve,
        "trades": trades,
        "metrics": metrics(curve),
        "unfilled_next_open": queued,
        "decisions": decisions,
        "initial_state": initial_state,
        "final_state": state,
    }


def main() -> None:
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy

    data = rq.load_data()
    candidate = pd.read_csv(OUT / "513350_source.csv").rename(columns={"date": "trade_date"})
    candidate["trade_date"] = pd.to_datetime(candidate.trade_date).dt.date
    candidate["symbol"] = "513350"
    candidate = candidate[["trade_date", "open", "close", "high", "low", "volume", "symbol"]]
    candidate.to_parquet(OUT / "513350.parquet", index=False)
    data["513350"] = candidate
    original = dict(rq.ETF_POOL)
    original_live = dict(live.ETF_POOL)
    original_dates = sorted(
        set.intersection(*(set(data[c].trade_date) for c in [*original, rq.DEFENSE]))
    )
    grid = set(original_dates[130::5])
    shared = sorted(set(original_dates) & set(candidate.trade_date))
    start, end = shared[130], shared[-1]
    scenarios = {
        "original": original,
        "exclude_501018": {c: n for c, n in original.items() if c != "501018"},
        "replace_with_513350": {
            ("513350" if c == "501018" else c): ("标普油气ETF富国" if c == "501018" else n)
            for c, n in original.items()
        },
    }
    report: dict[str, Any] = {
        "start": str(start),
        "end": str(end),
        "params": asdict(live.v4.V4_PARAMS),
        "grid_anchor": str(original_dates[130]),
        "scenarios": {},
        "close_mirror_diagnostic": {},
    }
    state = json.loads((OUT / "after_state.json").read_text())
    try:
        # Original historical same-close replay retained only as a regression diagnostic.
        mirror = run_full_pool_strategy(
            {c: data[c] for c in [*original, rq.DEFENSE]}, live.v4.V4_PARAMS
        )
        report["close_mirror_diagnostic"] = mirror["metrics"]
        report["close_mirror_diagnostic"]["warning"] = (
            "T close signals filled at T close; optimistic, NOT an executable return claim."
        )
        for name, pool in scenarios.items():
            rq.ETF_POOL.clear()
            rq.ETF_POOL.update(pool)
            live.ETF_POOL.clear()
            live.ETF_POOL.update(pool)
            entry = {}
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
                if multiplier == 1.0:
                    result["curve"].to_csv(OUT / f"{name}_next_open.csv", index=False)
                    entry["last_trades"] = result["trades"][-6:]
                    entry["unfilled_next_open"] = result["unfilled_next_open"]
            idx = {c: int(data[c].index[data[c].trade_date == end][0]) for c in [*pool, rq.DEFENSE]}
            target, candidates, _, _ = live.select_target(data, idx, None)
            overlay = live.evaluate_v4_overlay(
                data=data,
                idx_map=idx,
                trading_dates=original_dates,
                td=end,
                holding=None,
                base_target=target,
                candidates=candidates,
                scheduled_rebalance=end in grid,
                state=copy.deepcopy(state),
                mode="V4",
            )
            entry["latest_close"] = {
                "target": target,
                "candidates": candidates,
                "scheduled": end in grid,
                "would_trade": overlay["is_rebalance"],
                "overlay_block": overlay["decision"].blocked_by,
            }
            report["scenarios"][name] = entry
    finally:
        rq.ETF_POOL.clear()
        rq.ETF_POOL.update(original)
        live.ETF_POOL.clear()
        live.ETF_POOL.update(original_live)
    report["asset_factors"] = {
        c: asdict(live.v4.asset_factors(c, data[c].close.to_numpy())) for c in data
    }
    pair = pd.DataFrame(
        {c: data[c].set_index("trade_date").close for c in ["501018", "513350"]}
    ).dropna()
    report["oil_pair"] = {
        "daily_return_correlation": float(
            pair.pct_change().dropna().corr().to_numpy(dtype=float)[0, 1]
        ),
        "observations": len(pair) - 1,
    }
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
