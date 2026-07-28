"""通用回测引擎: 可插拔 select 函数, 复用 run_qixing_v3 的数据与费率机制.

设计要点 (无前瞻):
  - select_fn(data, idx_map, holding, params) -> 目标ETF代码, 只用截至当日数据.
  - 费率/滑点/调仓与实盘 V3 完全一致 (万五手续费 + 0.1%滑点 + 整数手).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_qixing_v3 as rq  # noqa: E402

FEE = rq.FEE
SLIPPAGE = rq.SLIPPAGE
DEFENSE = rq.DEFENSE
ETF_POOL = rq.ETF_POOL
WARMUP = 130


def get_common_dates(data: dict) -> list:
    """全部ETF的公共交易日 (升序)."""
    common: set | None = None
    for code in ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        common = set(dates) if common is None else (common & set(dates))
    if common is None:
        return []
    if DEFENSE in data:
        common &= set(data[DEFENSE]["trade_date"].tolist())
    return sorted(common)


def build_idx_map(data: dict, td) -> dict:
    """各ETF在td的历史索引位置 (需>=WARMUP根K线, 否则排除)."""
    idx_map = {}
    for code in list(ETF_POOL.keys()) + [DEFENSE]:
        if code not in data:
            continue
        df = data[code]
        n = int((df["trade_date"] <= td).sum())
        if n < WARMUP:
            continue
        idx_map[code] = n - 1
    return idx_map


def backtest(data: dict, select_fn, params: dict, rebalance_days: int = 5,
             initial: float = 100_000.0, start_date=None, end_date=None) -> dict:
    """通用回测.

    Args:
        select_fn: select_fn(data, idx_map, holding, params) -> 目标代码 (None=防御).
        params: 策略参数 (含可选 rebalance_days, 其余传给 select_fn).
        start_date/end_date: 评估区间 (None=全段). 动量回看自动用区间前的历史数据.

    Returns:
        dict: total_return/ann_return/sharpe/max_drawdown/yearly/n_trades/
              equity_curve(DataFrame)/final_equity.
    """
    rebalance_days = int(params.get("rebalance_days", rebalance_days))
    all_dates = get_common_dates(data)[WARMUP:]  # 跳过预热期, 保证各ETF有足够历史
    if start_date is not None:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date is not None:
        all_dates = [d for d in all_dates if d <= end_date]
    rebalance_dates = all_dates[::rebalance_days]

    cash = initial
    holding: str | None = None
    shares = 0
    equity_history = []
    n_trades = 0

    for td in rebalance_dates:
        idx_map = build_idx_map(data, td)
        target = select_fn(data, idx_map, holding, params)
        if target is None:
            target = DEFENSE

        if target != holding:
            # 卖出当前持仓
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    cash += shares * float(row.iloc[0]["close"]) * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    holding = None
                    shares = 0
            # 买入目标 (预留1%现金, 按100股整数)
            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = float(row.iloc[0]["close"])
                    buy_shares = int(cash * 0.99 / price / 100) * 100
                    if buy_shares > 0:
                        cash -= buy_shares * price * (1 + FEE + SLIPPAGE)
                        holding = target
                        shares = buy_shares
                        n_trades += 1

        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return {"total_return": 0.0, "ann_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "yearly": {}, "n_trades": 0,
                "equity_curve": pd.DataFrame(), "final_equity": initial}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = eq_df["equity"].iloc[-1] / initial - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    periods_per_year = 252 / rebalance_days
    ann_vol = daily_rets.std() * np.sqrt(periods_per_year) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1 if total_return > -1 else -1.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = float(((eq_df["equity"] - cummax) / cummax).min())

    yearly: dict[int, dict] = {}
    prev = initial
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yearly[int(year)] = {"return": float(end_val / prev - 1)}
        prev = end_val

    return {
        "total_return": float(total_return), "ann_return": float(ann_ret),
        "sharpe": float(sharpe), "max_drawdown": max_dd, "yearly": yearly,
        "n_trades": n_trades, "equity_curve": eq_df,
        "final_equity": float(eq_df["equity"].iloc[-1]),
    }
