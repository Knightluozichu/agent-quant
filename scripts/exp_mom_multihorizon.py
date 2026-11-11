"""Research: multi-horizon momentum with adaptive weights, strict 14:50 fills.

Isolation rules:
  - Read-only research. Production files/state are never written.
  - Prices are forward-adjusted total-return prices anchored to the latest
    raw price (qfq), so current lot sizes stay tradable.
  - Signal and fill both use the 14:50 price; prior days use full daily bars.
  - The production selector filters and V5 risk overlay are kept unchanged;
    only the momentum score is replaced.

Methods:
  B0       0.5*R10 + 0.5*R20 (current V5)
  EQ6      equal weights on R1,R3,R5,R10,R20,R60
  EQ6x.5   the literal 0.5 coefficient on every horizon (sum=3.0)
  SHORT3   equal weights on R1,R3,R5
  IC6      trailing cross-sectional rank-IC weighted horizons
  IV6      inverse-volatility weighted horizons
  IS_IC6   frozen weights optimized on in-sample rank IC only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import live_signal as live
import qixing_v5 as v5
import run_qixing_v3 as rq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "etf_5min"
OUT = ROOT / "artifacts" / "mom_multihorizon_20261105"
CODES = [
    "518880.SH",
    "159985.SZ",
    "501018.SH",
    "161226.SZ",
    "513100.SH",
    "159915.SZ",
    "511220.SH",
    "511880.SH",
]
POOL = [c.split(".")[0] for c in CODES]
HORIZONS = (1, 3, 5, 10, 20, 60)
GRID_DAYS = 5
IS_START = date(2020, 6, 19)
IS_END = date(2023, 6, 30)
OOS_START = date(2023, 7, 1)
BUY_COST = 0.0015
SELL_COST = 0.0020


def load_qfq() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in CODES:
        raw = pd.read_parquet(SRC / f"{symbol}.parquet").reset_index()
        raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
        raw["hm"] = pd.to_datetime(raw["trade_time"]).dt.strftime("%H:%M:%S")
        raw = raw[
            (raw["close"] > 0) & (raw["hm"] >= "09:30:00") & (raw["hm"] <= "15:00:00")
        ].copy()
        raw = raw.sort_values("trade_time")
        # Source has whole-day adj_factor gaps (e.g. 513100 on 2026-08-06).
        # Carry the last valid factor forward; filling 1.0 would corrupt qfq prices.
        raw["adj"] = (
            pd.to_numeric(raw["adj_factor"], errors="coerce").ffill().bfill()
        )
        adj_last = float(raw["adj"].iloc[-1])
        raw["adj_day"] = raw.groupby("trade_date")["adj"].transform(
            lambda s: s.mode().iloc[-1]
        )
        scale = raw["adj_day"] / adj_last
        for col in ("open", "high", "low", "close"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce") * scale

        daily = (
            raw.groupby("trade_date")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("vol", "sum"),
            )
            .reset_index()
        )
        p1450 = []
        for day, group in raw.groupby("trade_date"):
            group = group.sort_values("hm")
            at = group[group["hm"] == "14:55:00"]
            if at.empty:
                at = group[group["hm"] <= "14:50:00"]
            if at.empty:
                continue
            bar = at.iloc[-1]
            upto = group[group["hm"] <= bar["hm"]]
            p1450.append(
                (
                    day,
                    float(bar["open"]),
                    float(upto["open"].iloc[0]),
                    float(upto["high"].max()),
                    float(upto["low"].min()),
                )
            )
        p1450_df = pd.DataFrame(
            p1450,
            columns=["trade_date", "px1450", "open1450", "high1450", "low1450"],
        )
        daily = (
            daily.merge(p1450_df, on="trade_date", how="left")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        out[symbol.split(".")[0]] = daily
    return out


def common_dates(data: dict[str, pd.DataFrame]) -> list[date]:
    dates: set[date] | None = None
    for df in data.values():
        current = set(df["trade_date"])
        dates = current if dates is None else dates & current
    return sorted(dates or [])


def build_panel(
    data: dict[str, pd.DataFrame], dates: list[date]
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    px = {c: data[c].set_index("trade_date")["px1450"] for c in POOL}
    close = {c: data[c].set_index("trade_date")["close"] for c in POOL}
    return px, close


def horizon_return(
    px: dict[str, pd.Series],
    close: dict[str, pd.Series],
    dates: list[date],
    code: str,
    i: int,
    h: int,
) -> float:
    if i - h < 0:
        return np.nan
    p0 = close[code].get(dates[i - h], np.nan)
    p1 = px[code].get(dates[i], np.nan)
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return np.nan
    return float(p1 / p0 - 1.0)


def ic_weights(
    px: dict[str, pd.Series],
    close: dict[str, pd.Series],
    dates: list[date],
    grid: list[date],
    window: int = 52,
    shrink: float = 0.5,
) -> dict[date, dict[int, float]]:
    """Causal trailing rank-IC weights; forward return is the next grid slot."""
    index = {d: i for i, d in enumerate(dates)}
    out: dict[date, dict[int, float]] = {}
    for gi, t in enumerate(grid):
        ics: dict[int, list[float]] = {h: [] for h in HORIZONS}
        for s in grid[max(0, gi - window) : gi]:
            j = index[s]
            if j + GRID_DAYS >= len(dates):
                continue
            fwd = {
                c: px[c].get(dates[j + GRID_DAYS], np.nan) / px[c].get(s, np.nan) - 1.0
                for c in POOL
            }
            fvec = np.array([fwd[c] for c in POOL], dtype=float)
            if not np.isfinite(fvec).all():
                continue
            for h in HORIZONS:
                rvec = np.array(
                    [horizon_return(px, close, dates, c, j, h) for c in POOL], dtype=float
                )
                mask = np.isfinite(rvec) & np.isfinite(fvec)
                if mask.sum() >= 5:
                    ics[h].append(float(spearmanr(rvec[mask], fvec[mask]).statistic))
        means = {
            h: float(np.nanmean(ics[h])) if ics[h] else 0.0
            for h in HORIZONS
        }
        raw = np.array([max(means[h], 0.0) for h in HORIZONS])
        if raw.sum() <= 0:
            raw = np.ones(len(HORIZONS))
        raw = raw / raw.sum()
        mixed = shrink * raw + (1.0 - shrink) * (1.0 / len(HORIZONS))
        out[t] = {h: float(w) for h, w in zip(HORIZONS, mixed, strict=True)}
    return out


def iv_weights(
    px: dict[str, pd.Series],
    close: dict[str, pd.Series],
    dates: list[date],
    grid: list[date],
    lookback: int = 60,
    shrink: float = 0.5,
) -> dict[date, dict[int, float]]:
    index = {d: i for i, d in enumerate(dates)}
    out: dict[date, dict[int, float]] = {}
    for t in grid:
        i = index[t]
        vols = {}
        for h in HORIZONS:
            per_asset = []
            for c in POOL:
                vals = []
                for k in range(max(h, i - lookback + 1), i + 1):
                    p0 = close[c].get(dates[k - h], np.nan)
                    p1 = close[c].get(dates[k], np.nan)
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                        vals.append(p1 / p0 - 1.0)
                if len(vals) >= 20:
                    per_asset.append(float(np.std(vals)))
            vols[h] = float(np.mean(per_asset)) if per_asset else 1.0
        raw = np.array([1.0 / max(vols[h], 1e-8) for h in HORIZONS])
        raw = raw / raw.sum()
        mixed = shrink * raw + (1.0 - shrink) * (1.0 / len(HORIZONS))
        out[t] = {h: float(w) for h, w in zip(HORIZONS, mixed, strict=True)}
    return out


def is_static_ic_weights(
    px: dict[str, pd.Series],
    close: dict[str, pd.Series],
    dates: list[date],
    grid: list[date],
    step: float = 0.1,
) -> dict[int, float]:
    """Coarse simplex grid on in-sample rank IC (SLSQP stalls on Spearman)."""
    index = {d: i for i, d in enumerate(dates)}
    r_rows = []
    f_rows = []
    for s in grid:
        j = index[s]
        if j + GRID_DAYS >= len(dates) or dates[j + GRID_DAYS] > IS_END:
            break
        fwd = np.array(
            [
                px[c].get(dates[j + GRID_DAYS], np.nan) / px[c].get(s, np.nan) - 1.0
                for c in POOL
            ],
            dtype=float,
        )
        r = np.array(
            [
                [horizon_return(px, close, dates, c, j, h) for h in HORIZONS]
                for c in POOL
            ],
            dtype=float,
        )
        if np.isfinite(fwd).all() and np.isfinite(r).all():
            r_rows.append(r)
            f_rows.append(fwd)
    if not r_rows:
        return {h: 1.0 / len(HORIZONS) for h in HORIZONS}
    r_arr = np.stack(r_rows)  # (obs, assets, horizons)
    f_rank = rankdata(np.stack(f_rows), axis=1)
    f_rank = f_rank - f_rank.mean(axis=1, keepdims=True)

    best_w = None
    best_ic = -2.0
    best_rows: list[tuple[float, tuple[int, ...]]] = []
    units = round(1.0 / step)
    for combo in _compositions(units, len(HORIZONS)):
        w = np.array(combo, dtype=float) / units
        score = r_arr @ w
        s_rank = rankdata(score, axis=1)
        s_rank = s_rank - s_rank.mean(axis=1, keepdims=True)
        num = (s_rank * f_rank).sum(axis=1)
        den = np.sqrt((s_rank**2).sum(axis=1) * (f_rank**2).sum(axis=1))
        ic = float(np.nanmean(num / np.where(den > 0, den, np.nan)))
        best_rows.append((ic, combo))
        if ic > best_ic:
            best_ic = ic
            best_w = w
    best_rows.sort(reverse=True)
    print(
        "IS top weights:",
        [
            {"w": tuple(round(x, 1) for x in np.array(c) / units), "ic": round(v, 4)}
            for v, c in best_rows[:5]
        ],
        flush=True,
    )
    assert best_w is not None
    return {h: float(w) for h, w in zip(HORIZONS, best_w, strict=True)}


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


MODELS: dict[str, dict[int, float] | str] = {
    "B0": {10: 0.5, 20: 0.5},
    "B0_49": {10: 0.49, 20: 0.51},
    "B0_51": {10: 0.51, 20: 0.49},
    "B0_R5_10": {10: 0.45, 20: 0.45, 5: 0.10},
    "B0_R5_20": {10: 0.40, 20: 0.40, 5: 0.20},
    "B0_R1_10": {10: 0.45, 20: 0.45, 1: 0.10},
    "B0_R60_10": {10: 0.45, 20: 0.45, 60: 0.10},
    "B0_R60_05": {10: 0.475, 20: 0.475, 60: 0.05},
    "B0_R60_15": {10: 0.425, 20: 0.425, 60: 0.15},
    "B0_R60_20": {10: 0.40, 20: 0.40, 60: 0.20},
    "B0_R50_10": {10: 0.45, 20: 0.45, 50: 0.10},
    "B0_R70_10": {10: 0.45, 20: 0.45, 70: 0.10},
    "B0_R80_10": {10: 0.45, 20: 0.45, 80: 0.10},
    "R60_10_T20": {10: 0.40, 20: 0.50, 60: 0.10},
    "R60_10_T10": {10: 0.50, 20: 0.40, 60: 0.10},
    "EQ6": {h: 1.0 / len(HORIZONS) for h in HORIZONS},
    "EQ6x0.5": dict.fromkeys(HORIZONS, 0.5),
    "SHORT3": {1: 1 / 3, 3: 1 / 3, 5: 1 / 3},
    "IC6": "adaptive_ic",
    "IC6P": "adaptive_ic_pure",
    "IV6": "adaptive_iv",
    "IS_IC6": "is_static",
}


def expand_weights(
    weights_by_grid: dict[date, dict[int, float]], dates: list[date]
) -> dict[date, dict[int, float]]:
    """Carry the latest grid-date weights onto every trading day."""
    out: dict[date, dict[int, float]] = {}
    last: dict[int, float] | None = None
    for d in dates:
        if d in weights_by_grid:
            last = weights_by_grid[d]
        if last is not None:
            out[d] = last
    return out


def select_model(
    data: dict[str, pd.DataFrame],
    idx_map: dict[str, int],
    holding: str | None,
    td: date,
    model: str,
    weights: dict[int, float] | dict[date, dict[int, float]],
):
    a_share_weak = (
        rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
        if rq.USE_A_SHARE_FILTER
        else False
    )
    if model in ("IC6", "IC6P", "IV6"):
        w = weights[td]  # type: ignore[index]
    elif model == "IS_IC6":
        w = weights  # type: ignore[assignment]
    else:
        w = weights  # type: ignore[assignment]

    candidates = []
    for code in live.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        close = data[code]["close"].values[: idx + 1].astype(float)
        volume = data[code]["volume"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_SHORT_MOM_FILTER and not rq.check_short_momentum(close):
            continue
        if rq.USE_VOL_SPIKE_FILTER and not rq.check_volume_spike(volume, close):
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        if (
            rq.USE_LONG_MOM_FILTER
            and code in ("513100", "159915")
            and len(close) > rq.LONG_MOM_PERIOD
        ):
            long_mom = (close[-1] - close[-rq.LONG_MOM_PERIOD - 1]) / close[
                -rq.LONG_MOM_PERIOD - 1
            ]
            if long_mom < 0:
                continue
        score = 0.0
        for h in w:
            weight = float(w.get(h, 0.0))
            if weight and len(close) > h:
                score += weight * (close[-1] / close[-h - 1] - 1.0)
        if score > 0:
            candidates.append((code, float(score)))

    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else live.DEFENSE
    best_score = candidates[0][1] if candidates else 0.0
    threshold = 0.0 if best_score > 0.10 else 0.05

    if holding and holding in idx_map and holding != live.DEFENSE:
        hclose = data[holding]["close"].values[: idx_map[holding] + 1].astype(float)
        if len(hclose) > rq.DROP_LOOKBACK + 1 and len(hclose) > 61:
            drop = any(
                (hclose[i] - hclose[i - 1]) / hclose[i - 1] < rq.DROP_THRESHOLD
                for i in range(-rq.DROP_LOOKBACK, 0)
            )
            ret60 = (hclose[-1] - hclose[-61]) / hclose[-61]
            if drop and ret60 < 0.01:
                threshold = 0.0

    if holding and holding != live.DEFENSE:
        cur_score = dict(candidates).get(holding, -999)
        if cur_score > 0:
            target = best_target if best_score > cur_score + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target, candidates, best_score, a_share_weak


def metrics(curve: list[dict], initial: float) -> dict:
    frame = pd.DataFrame(curve)
    eq = frame["equity"].to_numpy(dtype=float)
    path = np.r_[initial, eq]
    years = (
        pd.to_datetime(frame["trade_date"].iloc[-1])
        - pd.to_datetime(frame["trade_date"].iloc[0])
    ).days / 365.25
    rets = np.diff(path) / path[:-1]
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    annual_factor = np.sqrt(252 / GRID_DAYS)
    monthly = (
        frame.set_index(pd.to_datetime(frame["trade_date"]))
        .equity.resample("ME")
        .last()
        .pct_change()
        .dropna()
    )
    yearly: dict[int, float] = {}
    frame["year"] = pd.to_datetime(frame["trade_date"]).dt.year
    prev_equity = initial
    for year, group in frame.groupby("year"):
        yearly[int(year)] = float(group["equity"].iloc[-1] / prev_equity - 1)
        prev_equity = float(group["equity"].iloc[-1])
    return {
        "final_equity": round(float(eq[-1]), 2),
        "total_return_pct": round(float(eq[-1] / initial - 1) * 100, 2),
        "cagr_pct": round(float((eq[-1] / initial) ** (1 / years) - 1) * 100, 2)
        if years > 0
        else 0.0,
        "sharpe": round(float(rets.mean() / vol * annual_factor), 2) if vol else 0.0,
        "max_drawdown_pct": round(float((path / np.maximum.accumulate(path) - 1).min()) * 100, 2),
        "years": round(years, 2),
        "best_month_pct": round(float(monthly.max()) * 100, 2) if len(monthly) else 0.0,
        "worst_month_pct": round(float(monthly.min()) * 100, 2) if len(monthly) else 0.0,
        "trades": int(frame["trades"].iloc[-1]),
        "holding": frame["holding"].iloc[-1] if isinstance(frame["holding"].iloc[-1], str) else None,
        "yearly": yearly,
        "start": str(frame["trade_date"].iloc[0]),
        "end": str(frame["trade_date"].iloc[-1]),
    }


def segment_metrics(curve: list[dict], start: date, end: date, initial: float) -> dict | None:
    rows = [r for r in curve if start <= r["trade_date"] <= end]
    if not rows:
        return None
    prior = [r for r in curve if r["trade_date"] < start]
    base = float(prior[-1]["equity"]) if prior else initial
    result = metrics(rows, base)
    result["segment_start"] = str(start)
    result["segment_end"] = str(end)
    return result


def run(
    model: str,
    weights,
    data: dict[str, pd.DataFrame],
    dates: list[date],
    grid: set[date],
    mode: str,
    initial: float,
    cost_mult: float = 1.0,
) -> tuple[dict, list[dict], dict]:
    cash = initial
    holding = None
    shares = 0
    entry_price = 0.0
    state = live.default_state(initial)
    state.update({"cash": cash, "holding": None, "shares": 0, "entry_price": 0.0})
    curve: list[dict] = []
    trades = 0
    codes = list(data)
    decision_counts: dict[str, int] = {}

    for i, td in enumerate(dates):
        today = {}
        for code in codes:
            row = data[code][data[code]["trade_date"] == td]
            if row.empty or not np.isfinite(row["px1450"].iloc[0]):
                today = {}
                break
            today[code] = row.iloc[0]
        if not today:
            continue

        hist = {}
        idx_map = {}
        for code in codes:
            base = data[code][data[code]["trade_date"] < td].copy()
            r = today[code]
            base = pd.concat(
                [
                    base,
                    pd.DataFrame(
                        [
                            {
                                "trade_date": td,
                                "open": float(r["open1450"]) if "open1450" in r else float(r["open"]),
                                "high": float(r["high1450"]) if "high1450" in r else float(r["high"]),
                                "low": float(r["low1450"]) if "low1450" in r else float(r["low"]),
                                "close": float(r["px1450"]),
                                "volume": float(r["volume"]),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            hist[code] = base
            idx_map[code] = len(base) - 1

        target, candidates, _best, _weak = select_model(
            hist, idx_map, holding, td, model, weights
        )
        scheduled = td in grid
        if mode == "V3-G":
            final_target = target
            rebalance = scheduled and target != holding
        else:
            state["cash"] = cash
            state["holding"] = holding
            state["shares"] = shares
            state["entry_price"] = entry_price
            overlay = live.evaluate_v4_overlay(
                data=hist,
                idx_map=idx_map,
                trading_dates=dates[: i + 1],
                td=td,
                holding=holding,
                base_target=target,
                candidates=candidates,
                scheduled_rebalance=scheduled,
                state=state,
                mode="V4",
            )
            risk = (
                v5.assess_v5(
                    target=overlay["target"],
                    holding=holding,
                    state=state,
                    data=hist,
                    td=td,
                    idx_map=idx_map,
                    is_rebalance=overlay["is_rebalance"],
                    common_dates=dates[: i + 1],
                    spot_map=None,
                )
                if mode == "V5"
                else live.risk_assess(
                    target=overlay["target"],
                    holding=holding,
                    state=state,
                    data=hist,
                    td=td,
                    idx_map=idx_map,
                    is_rebalance=overlay["is_rebalance"],
                    common_dates=dates[: i + 1],
                    spot_map=None,
                )
            )
            final_target = risk.final_target or live.DEFENSE
            rebalance = overlay["is_rebalance"] or risk.action == live.ACTION_EMERGENCY

        price = float(today[final_target]["px1450"]) if final_target in today else 0.0
        if rebalance and final_target != holding and price > 0:
            if holding and shares:
                cash += shares * float(today[holding]["px1450"]) * (1 - SELL_COST * cost_mult)
            shares = int(cash * 0.99 / price / 100) * 100
            if shares:
                cash -= shares * price * (1 + BUY_COST * cost_mult)
                holding = final_target
                entry_price = price
                trades += 1
            else:
                holding = None
                shares = 0
        equity = cash + shares * float(today[holding]["px1450"]) if holding else cash
        state["peak_equity"] = max(float(state.get("peak_equity", initial)), equity)
        decision_counts[final_target or "CASH"] = decision_counts.get(final_target or "CASH", 0) + 1
        curve.append(
            {"trade_date": td, "equity": equity, "holding": holding, "trades": trades}
        )

    result = metrics(curve, initial)
    result["mode"] = mode
    result["model"] = model
    result["decision_counts"] = decision_counts
    return result, curve, {"weights": weights}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="B0")
    parser.add_argument("--modes", default="V5")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cost-mult", type=float, default=1.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_qfq()
    dates_all = common_dates(data)
    anchor = dates_all.index(IS_START)
    grid = dates_all[anchor::GRID_DAYS]
    grid_set = set(grid)
    run_dates = [d for d in dates_all if d >= IS_START]
    run_grid = set(d for d in grid_set if d >= IS_START)
    px, close = build_panel(data, dates_all)

    print(
        f"data={dates_all[0]}..{dates_all[-1]} n={len(dates_all)} "
        f"grid={grid[0]}..{grid[-1]} n={len(grid)}",
        flush=True,
    )

    adaptive_cache: dict[str, dict] = {}
    static_weights: dict[int, float] | None = None
    report = {
        "meta": {
            "data_start": str(dates_all[0]),
            "data_end": str(dates_all[-1]),
            "grid_start": str(grid[0]),
            "grid_end": str(grid[-1]),
            "grid_slots": len(grid),
            "is": [str(IS_START), str(IS_END)],
            "oos": [str(OOS_START), str(dates_all[-1])],
            "capital": args.capital,
            "buy_cost": BUY_COST * args.cost_mult,
            "sell_cost": SELL_COST * args.cost_mult,
            "price_basis": "qfq total-return, anchored to latest raw price",
            "fill": "14:50 strict",
        },
        "models": {},
    }

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for model in requested_models:
        spec = MODELS[model]
        if spec == "adaptive_ic":
            if "IC6" not in adaptive_cache:
                print("precompute IC6 weights...", flush=True)
                adaptive_cache["IC6"] = ic_weights(px, close, dates_all, grid)
            weights = expand_weights(adaptive_cache["IC6"], run_dates)
        elif spec == "adaptive_ic_pure":
            if "IC6P" not in adaptive_cache:
                print("precompute IC6P weights...", flush=True)
                adaptive_cache["IC6P"] = ic_weights(
                    px, close, dates_all, grid, shrink=1.0
                )
            weights = expand_weights(adaptive_cache["IC6P"], run_dates)
        elif spec == "adaptive_iv":
            if "IV6" not in adaptive_cache:
                print("precompute IV6 weights...", flush=True)
                adaptive_cache["IV6"] = iv_weights(px, close, dates_all, grid)
            weights = expand_weights(adaptive_cache["IV6"], run_dates)
        elif spec == "is_static":
            if static_weights is None:
                print("optimize IS_IC6 weights...", flush=True)
                static_weights = is_static_ic_weights(px, close, dates_all, grid)
                print("IS_IC6 weights:", static_weights, flush=True)
            weights = static_weights
        else:
            weights = spec

        for mode in requested_modes:
            key = f"{model}_{mode}"
            started = time.time()
            result, curve, extra = run(
                model,
                weights,
                deepcopy(data),
                run_dates,
                run_grid,
                mode,
                args.capital,
                args.cost_mult,
            )
            result["is"] = segment_metrics(curve, IS_START, IS_END, args.capital)
            result["oos"] = segment_metrics(curve, OOS_START, dates_all[-1], args.capital)
            result["rolling_3y"] = {}
            for start_year in (2020, 2021, 2022, 2023):
                start = max(IS_START, date(start_year, 1, 1))
                end = min(dates_all[-1], date(start_year + 2, 12, 31))
                seg = segment_metrics(curve, start, end, args.capital)
                if seg:
                    result["rolling_3y"][f"{start_year}-{start_year + 2}"] = seg
            sample_key = next(iter(weights), None)
            if isinstance(sample_key, date):
                rows = [weights[d] for d in run_dates if d in weights]
                arr = np.array([[r[h] for h in HORIZONS] for r in rows])
                result["weights"] = "adaptive"
                result["weights_mean"] = {
                    str(h): round(float(v), 4)
                    for h, v in zip(HORIZONS, arr.mean(axis=0), strict=True)
                }
                result["weights_last"] = {
                    str(h): round(float(v), 4)
                    for h, v in zip(HORIZONS, arr[-1], strict=True)
                }
            else:
                result["weights"] = {str(k): v for k, v in weights.items()}
            result["elapsed_s"] = round(time.time() - started, 1)
            report["models"][key] = result
            (OUT / f"curve_{key}{args.tag}.csv").write_text(
                pd.DataFrame(curve).to_csv(index=False), encoding="utf-8"
            )
            print(
                f"{key}: total={result['total_return_pct']}% cagr={result['cagr_pct']}% "
                f"sharpe={result['sharpe']} mdd={result['max_drawdown_pct']}% "
                f"trades={result['trades']} ({result['elapsed_s']}s)",
                flush=True,
            )

    path = OUT / f"report{args.tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
