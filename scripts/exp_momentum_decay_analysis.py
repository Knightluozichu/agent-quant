"""V3策略动量衰减规律与亏损模式分析.

分析目标:
  1. 追踪每次持仓期的动量变化轨迹, 验证动量衰减假设
  2. 分析回撤与动量衰减的关系
  3. 评估动量一阶导退出信号的有效性

用法: uv run python scripts/exp_momentum_decay_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "qixing_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calc_momentum(close: np.ndarray, periods=(10, 20), weights=(0.5, 0.5)) -> float:
    """计算加权动量评分 (与V3一致)."""
    score = 0.0
    for period, weight in zip(periods, weights, strict=False):
        if len(close) > period:
            ret = (close[-1] - close[-period - 1]) / close[-period - 1]
            score += ret * weight
    return score


def calc_momentum_derivative(close: np.ndarray, lookback=5) -> float:
    """计算动量的一阶导数 (动量变化率).

    过去 lookback 个调仓周期内, 动量评分的变化率.
    正值 = 动量加速上升, 负值 = 动量衰减.
    """
    if len(close) < 121:
        return 0.0
    # 当前动量
    current_mom = calc_momentum(close)
    # 回溯 lookback 个交易日前的动量 (V3每5天调仓, 用5天步长)
    step = 5
    if len(close) < step + 1:
        return 0.0
    prev_mom = calc_momentum(close[:-step])
    return current_mom - prev_mom


def run_v3_with_holding_analysis(
    data: dict, initial_capital: float = 100_000.0,
) -> dict:
    """运行V3回测并详细追踪每次持仓期的动量变化.

    与 run_qixing_v3_same_day 逻辑一致, 但额外记录:
    - 每次持仓期的逐调仓日动量追踪
    - 持仓期内的动量峰值/底谷/变化率
    - 回撤事件与动量衰减的对应关系
    """
    # 初始化和数据准备 (同V3 same-day)
    common_dates: set = set()
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if rq.DEFENSE in data:
        common_dates &= set(data[rq.DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::rq.REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[dict] = []

    # 详细持仓期动量追踪
    holding_periods: list[dict] = []
    current_holding_start: str | None = None
    current_holding_code: str | None = None
    current_mom_track: list[dict] = []  # 当前持仓期的逐日动量

    # 决策记录(含动量详细信息)
    detailed_decisions: list[dict] = []

    # 回撤事件记录
    drawdown_events: list[dict] = []
    peak_equity = initial_capital
    peak_date = None
    in_drawdown = False
    dd_start = None
    dd_start_equity = None

    for td in trading_dates:
        # === 调仓日: 记录动量并执行交易 ===
        if td in rebalance_set:
            # 构建各ETF截至td的索引
            etf_data_at_date = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            # 获取当前持仓的动量
            holding_mom = None
            holding_mom_d1 = None  # 一阶导
            if holding and holding in etf_data_at_date:
                idx = etf_data_at_date[holding]
                close = data[holding]["close"].values[: idx + 1].astype(float)
                if len(close) >= 121:
                    holding_mom = calc_momentum(close)
                    holding_mom_d1 = calc_momentum_derivative(close)

            # 获取所有候选ETF的动量
            all_moms = {}
            for code in rq.ETF_POOL:
                if code in etf_data_at_date:
                    idx = etf_data_at_date[code]
                    close = data[code]["close"].values[: idx + 1].astype(float)
                    if len(close) >= 121:
                        all_moms[code] = calc_momentum(close)

            # 执行选股 (同V3)
            target, candidates, best_score, a_share_weak = rq.select_target(
                data, etf_data_at_date, holding
            )

            # 记录当前持仓期的动量追踪
            if holding and holding_mom is not None:
                current_mom_track.append({
                    "date": str(td),
                    "momentum": round(holding_mom, 4),
                    "momentum_d1": round(holding_mom_d1, 4) if holding_mom_d1 is not None else None,
                    "code": holding,
                    "name": rq.ETF_POOL.get(holding, "货币基金"),
                })

            # 执行交易
            if target != holding:
                # 卖出
                if holding and holding in data:
                    row = data[holding][data[holding]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        cash += holding_shares * price * (1 - rq.FEE - rq.SLIPPAGE)

                        # 记录当前持仓期结束
                        if current_holding_start is not None:
                            exit_mom = holding_mom
                            exit_mom_d1 = holding_mom_d1
                            mom_values = [m["momentum"] for m in current_mom_track if m["momentum"] is not None]
                            mom_d1_values = [m["momentum_d1"] for m in current_mom_track if m["momentum_d1"] is not None]

                            # 计算持仓期内的动量变化指标
                            holding_period = {
                                "code": holding,
                                "name": rq.ETF_POOL.get(holding, "货币基金"),
                                "start_date": str(current_holding_start),
                                "end_date": str(td),
                                "entry_momentum": round(current_mom_track[0]["momentum"], 4) if current_mom_track else None,
                                "exit_momentum": round(exit_mom, 4) if exit_mom is not None else None,
                                "peak_momentum": round(max(mom_values), 4) if mom_values else None,
                                "min_momentum": round(min(mom_values), 4) if mom_values else None,
                                "mom_trend": round(mom_values[-1] - mom_values[0], 4) if len(mom_values) >= 2 else 0,
                                "mom_peak_minus_exit": round(max(mom_values) - mom_values[-1], 4) if mom_values else None,
                                "peak_mom_d1": round(max(mom_d1_values), 4) if mom_d1_values else None,
                                "min_mom_d1": round(min(mom_d1_values), 4) if mom_d1_values else None,
                                "last_mom_d1": round(mom_d1_values[-1], 4) if mom_d1_values else None,
                                "mom_d1_trend": round(mom_d1_values[-1] - mom_d1_values[0], 4) if len(mom_d1_values) >= 2 else 0,
                                "n_rebalance_checks": len(current_mom_track),
                                "momentum_trace": current_mom_track.copy(),
                            }
                            holding_periods.append(holding_period)

                        # 重置持仓期追踪
                        current_holding_start = None
                        current_holding_code = None
                        current_mom_track = []
                        holding = None
                        holding_shares = 0

                # 买入
                if target in data:
                    row = data[target][data[target]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + rq.FEE + rq.SLIPPAGE)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            # 新持仓期开始
                            current_holding_start = td
                            current_holding_code = target
                            # 记录买入时的动量
                            target_idx = etf_data_at_date.get(target)
                            if target_idx is not None:
                                target_close = data[target]["close"].values[: target_idx + 1].astype(float)
                                if len(target_close) >= 121:
                                    entry_mom = calc_momentum(target_close)
                                    entry_mom_d1 = calc_momentum_derivative(target_close)
                                    current_mom_track = [{
                                        "date": str(td),
                                        "momentum": round(entry_mom, 4),
                                        "momentum_d1": round(entry_mom_d1, 4),
                                        "code": target,
                                        "name": rq.ETF_POOL.get(target, "货币基金"),
                                        "action": "BUY",
                                    }]

        # === 每日净值 ===
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])

        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE})

        # === 回撤事件追踪 ===
        if equity > peak_equity:
            peak_equity = equity
            peak_date = td
            if in_drawdown:
                # 回撤结束
                dd_end = td
                dd_depth = (dd_start_equity - dd_start_equity * (1 + (equity - peak_equity) / peak_equity)) / dd_start_equity
                # 简化: 记录回撤事件
                in_drawdown = False

        drawdown = (equity - peak_equity) / peak_equity
        if drawdown < -0.05 and not in_drawdown:  # 5%回撤开始
            in_drawdown = True
            dd_start = td
            dd_start_equity = peak_equity
            drawdown_events.append({
                "start_date": str(td),
                "start_equity": round(peak_equity, 2),
                "holding": holding or rq.DEFENSE,
                "holding_name": rq.ETF_POOL.get(holding, "货币基金") if holding else "货币基金",
            })
        if in_drawdown:
            # 更新回撤深度
            current_dd = (equity - peak_equity) / peak_equity
            if drawdown_events:
                drawdown_events[-1]["max_dd"] = round(current_dd, 4)
                drawdown_events[-1]["max_dd_date"] = str(td)
                drawdown_events[-1]["end_equity"] = round(equity, 2)

        # 记录决策信息(含动量)
        if td in rebalance_set:
            detailed_decisions.append({
                "date": str(td),
                "holding": holding or rq.DEFENSE,
                "holding_name": rq.ETF_POOL.get(holding, "货币基金") if holding else "货币基金",
                "holding_momentum": round(holding_mom, 4) if holding_mom is not None else None,
                "holding_mom_d1": round(holding_mom_d1, 4) if holding_mom_d1 is not None else None,
                "best_target": target,
                "best_score": round(best_score, 4),
                "n_candidates": len(candidates),
                "a_share_weak": a_share_weak,
                "equity": round(equity, 2),
            })

    # 完成最后一个持仓期
    if holding and current_mom_track:
        exit_mom = current_mom_track[-1]["momentum"] if current_mom_track else None
        mom_values = [m["momentum"] for m in current_mom_track if m["momentum"] is not None]
        mom_d1_values = [m["momentum_d1"] for m in current_mom_track if m["momentum_d1"] is not None]
        holding_periods.append({
            "code": holding,
            "name": rq.ETF_POOL.get(holding, "货币基金"),
            "start_date": str(current_holding_start),
            "end_date": str(trading_dates[-1]),
            "entry_momentum": round(current_mom_track[0]["momentum"], 4) if current_mom_track else None,
            "exit_momentum": round(exit_mom, 4) if exit_mom is not None else None,
            "peak_momentum": round(max(mom_values), 4) if mom_values else None,
            "min_momentum": round(min(mom_values), 4) if mom_values else None,
            "mom_trend": round(mom_values[-1] - mom_values[0], 4) if len(mom_values) >= 2 else 0,
            "mom_peak_minus_exit": round(max(mom_values) - mom_values[-1], 4) if mom_values else None,
            "peak_mom_d1": round(max(mom_d1_values), 4) if mom_d1_values else None,
            "min_mom_d1": round(min(mom_d1_values), 4) if mom_d1_values else None,
            "last_mom_d1": round(mom_d1_values[-1], 4) if mom_d1_values else None,
            "mom_d1_trend": round(mom_d1_values[-1] - mom_d1_values[0], 4) if len(mom_d1_values) >= 2 else 0,
            "n_rebalance_checks": len(current_mom_track),
            "momentum_trace": current_mom_track.copy(),
            "is_active": True,
        })

    # 计算汇总指标
    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    return {
        "total_return": total_return,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity_curve": eq_df,
        "holding_periods": holding_periods,
        "detailed_decisions": detailed_decisions,
        "drawdown_events": drawdown_events,
    }


def analyze_momentum_patterns(holding_periods: list[dict]) -> dict:
    """分析持仓期动量变化规律."""
    if not holding_periods:
        return {"error": "无持仓期数据"}

    # 按ETF分类
    by_etf: dict[str, list[dict]] = {}
    for hp in holding_periods:
        code = hp["code"]
        if code not in by_etf:
            by_etf[code] = []
        by_etf[code].append(hp)

    # 1. 入场动量分布
    entry_moms = [hp["entry_momentum"] for hp in holding_periods if hp["entry_momentum"] is not None]

    # 2. 动量衰减分析: 动量峰值到退出时的差距
    mom_peak_exit_diffs = [hp["mom_peak_minus_exit"] for hp in holding_periods
                           if hp["mom_peak_minus_exit"] is not None]

    # 3. 动量趋势分析: 持仓期内动量变化方向
    positive_trend = sum(1 for hp in holding_periods if hp.get("mom_trend", 0) > 0)
    negative_trend = sum(1 for hp in holding_periods if hp.get("mom_trend", 0) < 0)
    flat_trend = len(holding_periods) - positive_trend - negative_trend

    # 4. 动量峰值后仍持有的次数 (动量过峰后未及时退出)
    peaked_then_held = [hp for hp in holding_periods
                        if hp.get("mom_peak_minus_exit", 0) > 0.02
                        and hp.get("n_rebalance_checks", 0) >= 2]

    # 5. 动量一阶导分析
    d1_negative_exit = [hp for hp in holding_periods
                        if hp.get("last_mom_d1") is not None and hp["last_mom_d1"] < 0]

    return {
        "total_holding_periods": len(holding_periods),
        "by_etf": {code: len(periods) for code, periods in by_etf.items()},
        "entry_momentum": {
            "mean": round(float(np.mean(entry_moms)), 4) if entry_moms else None,
            "median": round(float(np.median(entry_moms)), 4) if entry_moms else None,
            "min": round(float(min(entry_moms)), 4) if entry_moms else None,
            "max": round(float(max(entry_moms)), 4) if entry_moms else None,
            "p25": round(float(np.percentile(entry_moms, 25)), 4) if entry_moms else None,
            "p75": round(float(np.percentile(entry_moms, 75)), 4) if entry_moms else None,
            "distribution": {
                "<=0": round(float(np.mean([m <= 0 for m in entry_moms])), 4),
                "0~5%": round(float(np.mean([0 < m <= 0.05 for m in entry_moms])), 4),
                "5~10%": round(float(np.mean([0.05 < m <= 0.10 for m in entry_moms])), 4),
                "10~15%": round(float(np.mean([0.10 < m <= 0.15 for m in entry_moms])), 4),
                "15~20%": round(float(np.mean([0.15 < m <= 0.20 for m in entry_moms])), 4),
                ">20%": round(float(np.mean([m > 0.20 for m in entry_moms])), 4),
            },
        },
        "exit_momentum": {
            "mean": round(float(np.mean([hp["exit_momentum"] for hp in holding_periods
                                          if hp["exit_momentum"] is not None])), 4),
            "median": round(float(np.median([hp["exit_momentum"] for hp in holding_periods
                                              if hp["exit_momentum"] is not None])), 4),
        },
        "momentum_peak_to_exit": {
            "mean": round(float(np.mean(mom_peak_exit_diffs)), 4) if mom_peak_exit_diffs else None,
            "median": round(float(np.median(mom_peak_exit_diffs)), 4) if mom_peak_exit_diffs else None,
            "max": round(float(max(mom_peak_exit_diffs)), 4) if mom_peak_exit_diffs else None,
            "pct_with_decay": round(float(np.mean([d > 0 for d in mom_peak_exit_diffs])), 4) if mom_peak_exit_diffs else None,
            "pct_decay_over_2pct": round(float(np.mean([d > 0.02 for d in mom_peak_exit_diffs])), 4) if mom_peak_exit_diffs else None,
            "pct_decay_over_5pct": round(float(np.mean([d > 0.05 for d in mom_peak_exit_diffs])), 4) if mom_peak_exit_diffs else None,
        },
        "momentum_trend": {
            "positive": positive_trend,
            "negative": negative_trend,
            "flat": flat_trend,
            "pct_negative": round(negative_trend / len(holding_periods), 4) if holding_periods else 0,
        },
        "peaked_then_held": {
            "count": len(peaked_then_held),
            "pct": round(len(peaked_then_held) / len(holding_periods), 4) if holding_periods else 0,
            "avg_decay": round(float(np.mean([hp["mom_peak_minus_exit"]
                                               for hp in peaked_then_held])), 4) if peaked_then_held else None,
        },
        "momentum_d1_analysis": {
            "negative_exit_count": len(d1_negative_exit),
            "negative_exit_pct": round(len(d1_negative_exit) / len(holding_periods), 4) if holding_periods else 0,
        },
    }


def simulate_momentum_d1_exit(
    data: dict, initial_capital: float = 100_000.0,
    mom_d1_threshold: float = -0.02,
) -> dict:
    """模拟动量一阶导提前退出规则.

    规则: 在V3调仓日, 若当前持仓动量一阶导 < mom_d1_threshold,
    强制切货币基金(511880), 跳过V3的正常选股.

    Args:
        mom_d1_threshold: 动量一阶导阈值 (负值), 低于此值触发退出
    """
    common_dates: set = set()
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if rq.DEFENSE in data:
        common_dates &= set(data[rq.DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::rq.REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[dict] = []
    n_trades = 0
    d1_exit_log: list[dict] = []

    for td in trading_dates:
        if td in rebalance_set:
            etf_data_at_date = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            # 检查当前持仓的动量一阶导
            d1_triggered = False
            if holding and holding in etf_data_at_date:
                idx = etf_data_at_date[holding]
                close = data[holding]["close"].values[: idx + 1].astype(float)
                if len(close) >= 121:
                    mom_d1 = calc_momentum_derivative(close)
                    if mom_d1 < mom_d1_threshold:
                        d1_triggered = True
                        d1_exit_log.append({
                            "date": str(td),
                            "code": holding,
                            "name": rq.ETF_POOL.get(holding, "货币基金"),
                            "mom_d1": round(mom_d1, 4),
                            "threshold": mom_d1_threshold,
                        })

            if d1_triggered:
                target = rq.DEFENSE
            else:
                target, candidates, _best_score, _a_share_weak = rq.select_target(
                    data, etf_data_at_date, holding
                )

            if target != holding:
                if holding and holding in data:
                    row = data[holding][data[holding]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        cash += holding_shares * price * (1 - rq.FEE - rq.SLIPPAGE)
                        n_trades += 1
                        holding = None
                        holding_shares = 0

                if target in data:
                    row = data[target][data[target]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + rq.FEE + rq.SLIPPAGE)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            n_trades += 1

        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE})

    if not equity_history:
        return {"total_return": 0, "ann_return": 0, "sharpe": 0, "max_drawdown": 0}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    return {
        "total_return": total_return,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "d1_exit_count": len(d1_exit_log),
        "d1_exit_log": d1_exit_log,
        "equity_curve": eq_df,
    }


def print_analysis_report(
    analysis: dict,
    d1_results: list[tuple[float, dict]],
    drawdown_events: list[dict],
    v3_baseline: dict,
) -> None:
    """打印分析报告."""
    print("=" * 78)
    print("  V3策略动量衰减规律与亏损模式分析报告")
    print("=" * 78)

    # === 1. 基础回测表现 ===
    print(f"\n  【1】V3基准回测表现")
    print(f"  {'-' * 60}")
    res = v3_baseline
    print(f"    总收益:    {res['total_return']:>+8.1%}")
    print(f"    年化收益:  {res['ann_return']:>+8.1%}")
    print(f"    夏普比率:  {res['sharpe']:>8.2f}")
    print(f"    最大回撤:  {res['max_drawdown']:>8.1%}")

    # === 2. 持仓期动量分析 ===
    hp = analysis.get("holding_periods", [])
    print(f"\n  【2】持仓期动量变化分析 (共{len(hp)}个持仓期)")
    print(f"  {'-' * 60}")

    pat = analysis.get("momentum_patterns", {})

    if entry := pat.get("entry_momentum"):
        print("  [2a] 入场动量分布:")
        print(f"      均值: {entry['mean']:.1%}  |  中位数: {entry['median']:.1%}")
        print(f"      最小值: {entry['min']:.1%}  |  最大值: {entry['max']:.1%}")
        print(f"      25分位: {entry['p25']:.1%}  |  75分位: {entry['p75']:.1%}")
        print(f"      分布:")
        dist = entry.get("distribution", {})
        for k, v in dist.items():
            print(f"        {k}: {v:.1%}")

    if trend := pat.get("momentum_trend"):
        print(f"\n  [2b] 持仓期内动量趋势:")
        print(f"      上升趋势: {trend['positive']}次 ({trend['positive']/pat['total_holding_periods']:.1%})")
        print(f"      下降趋势: {trend['negative']}次 ({trend['negative']/pat['total_holding_periods']:.1%})")
        print(f"      持平:     {trend['flat']}次")

    if pte := pat.get("momentum_peak_to_exit"):
        print(f"\n  [2c] 动量峰值→退出衰减:")
        print(f"      平均衰减: {pte['mean']:.1%}")
        print(f"      中位数衰减: {pte['median']:.1%}")
        print(f"      最大衰减: {pte['max']:.1%}")
        print(f"      有衰减的持仓期: {pte['pct_with_decay']:.1%}")
        print(f"      衰减>2%的: {pte['pct_decay_over_2pct']:.1%}")
        print(f"      衰减>5%的: {pte['pct_decay_over_5pct']:.1%}")

    if pth := pat.get("peaked_then_held"):
        print(f"\n  [2d] 动量过峰后仍持有:")
        print(f"      次数: {pth['count']} ({pth['pct']:.1%})")
        print(f"      平均衰减幅度: {pth['avg_decay']:.1%}")

    # === 3. 动量一阶导分析 ===
    print(f"\n  【3】动量一阶导(变化率)分析")
    print(f"  {'-' * 60}")

    d1a = pat.get("momentum_d1_analysis", {})
    print(f"    退出时一阶导<0: {d1a.get('negative_exit_count', 0)}次 "
          f"({d1a.get('negative_exit_pct', 0):.1%})")

    # 展示各ETF的动量特征
    print(f"\n  [3a] 各ETF持仓期动量特征:")
    print(f"  {'ETF':<10} {'次数':<6} {'入场动量均值':<12} {'峰值→退出衰减':<14} {'衰减>2%比例':<12}")
    print(f"  {'-' * 58}")
    by_etf = pat.get("by_etf", {})
    for code in sorted(by_etf.keys()):
        periods = [hp for hp in hp if hp["code"] == code]
        entry_moms = [p["entry_momentum"] for p in periods if p["entry_momentum"] is not None]
        decays = [p["mom_peak_minus_exit"] for p in periods if p["mom_peak_minus_exit"] is not None]
        avg_entry = float(np.mean(entry_moms)) if entry_moms else 0
        avg_decay = float(np.mean(decays)) if decays else 0
        decay_pct = float(np.mean([d > 0.02 for d in decays])) if decays else 0
        name = rq.ETF_POOL.get(code, code)
        print(f"  {name:<10} {len(periods):<6} {avg_entry:>+8.1%}    "
              f"{avg_decay:>+8.1%}          {decay_pct:>6.1%}")

    # === 4. 回撤事件分析 ===
    print(f"\n  【4】回撤事件分析 (5%以上回撤)")
    print(f"  {'-' * 60}")
    print(f"    总回撤事件: {len(drawdown_events)}次")
    print(f"    最大回撤: {v3_baseline['max_drawdown']:.1%}")

    # 重大回撤事件明细
    major_dd = [dd for dd in drawdown_events if dd.get("max_dd", 0) < -0.1]
    if major_dd:
        print(f"\n    重大回撤 (>10%): {len(major_dd)}次")
        for i, dd in enumerate(major_dd[:10]):
            print(f"    {i+1}. {dd['start_date']} ~ {dd.get('max_dd_date', '?')}  "
                  f"回撤: {dd.get('max_dd', 0):.1%}  持仓: {dd['holding_name']}")

    # === 5. 动量一阶导退出模拟 ===
    print(f"\n  【5】动量一阶导(一阶导)退出信号模拟")
    print(f"  {'-' * 60}")

    # 找最佳的动量一阶导阈值
    best_d1 = max(d1_results, key=lambda x: x[1]["sharpe"])
    best_dd_d1 = min(d1_results, key=lambda x: x[1]["max_drawdown"])

    print(f"  {'阈值':<8} {'年化':>8} {'夏普':>8} {'回撤':>8} {'交易':>6} {'触发':>6}")
    print(f"  {'-' * 46}")
    for threshold, res in d1_results:
        print(f"  {threshold:<+8.1%} {res['ann_return']:>+8.1%} {res['sharpe']:>8.2f} "
              f"{res['max_drawdown']:>8.1%} {res['n_trades']:>6} {res['d1_exit_count']:>6}")

    print(f"\n  V3基准:     年化 {v3_baseline['ann_return']:>+7.1%}  "
          f"夏普 {v3_baseline['sharpe']:.2f}  回撤 {v3_baseline['max_drawdown']:.1%}")
    print(f"  最佳夏普:   年化 {best_d1[1]['ann_return']:>+7.1%}  "
          f"夏普 {best_d1[1]['sharpe']:.2f}  回撤 {best_d1[1]['max_drawdown']:.1%}  "
          f"阈值={best_d1[0]:.1%}")
    print(f"  最佳控回撤: 年化 {best_dd_d1[1]['ann_return']:>+7.1%}  "
          f"夏普 {best_dd_d1[1]['sharpe']:.2f}  回撤 {best_dd_d1[1]['max_drawdown']:.1%}  "
          f"阈值={best_dd_d1[0]:.1%}")

    # === 6. 结论 ===
    print(f"\n  【6】核心结论与建议")
    print(f"  {'-' * 60}")

    # 结论1: 动量衰减验证
    decay_pct = pte.get("pct_with_decay", 0) if pte else 0
    decay_gt_2 = pte.get("pct_decay_over_2pct", 0) if pte else 0
    print(f"\n  [6a] 动量衰减假设验证:")
    print(f"      - 入场动量均值: {entry['mean']:.1%}" if entry else "")
    print(f"      - 动量峰值→退出平均衰减: {pte['mean']:.1%}" if pte else "")
    print(f"      - {decay_pct:.0%}的持仓期存在动量衰减, {decay_gt_2:.0%}衰减>2%")
    if decay_gt_2 > 0.5:
        print(f"      ✓ 假设成立: 大部分持仓期存在显著的动量衰减")
    else:
        print(f"      △ 部分成立: 动量衰减普遍存在但幅度有限")

    # 结论2: 回撤与动量衰减关系
    d1_neg_pct = d1a.get("negative_exit_pct", 0) if d1a else 0
    print(f"\n  [6b] 回撤与动量衰减关系:")
    print(f"      - {d1_neg_pct:.0%}的持仓期在退出时动量一阶导为负")
    print(f"      - 动量过峰后仍持有: {pth['count']}次 ({pth['pct']:.1%})" if pth else "")
    if d1_neg_pct > 0.6:
        print(f"      ✓ 核心猜想成立: 大部分回撤发生在动量衰减但未及时退出时")
    else:
        print(f"      △ 部分成立: 动量衰减是回撤的重要原因之一")

    # 结论3: 动量一阶导有效性
    d1_improvement = best_d1[1]["sharpe"] - v3_baseline["sharpe"]
    dd_improvement = best_dd_d1[1]["max_drawdown"] - v3_baseline["max_drawdown"]
    print(f"\n  [6c] 动量一阶导退出信号有效性:")
    if d1_improvement > 0.1 and dd_improvement > 0.02:
        print(f"      ✓ 有效: 夏普提升{d1_improvement:.2f}, 回撤改善{dd_improvement:.1%}")
        print(f"      - 推荐阈值: {best_d1[0]:.1%}")
    elif d1_improvement > 0:
        print(f"      △ 轻微改善: 夏普提升{d1_improvement:.2f}, 回撤改善{dd_improvement:.1%}")
        print(f"      - 考虑作为辅助信号, 不单独使用")
    else:
        print(f"      ✗ 无效: 动量一阶导退出信号降低夏普({d1_improvement:+.2f})")
        print(f"      - 动量一阶导本身不适合作为独立退出信号")
        print(f"      - 建议结合其他信号(如A股MA15走弱)综合判断")
    print(f"\n{'=' * 78}")


def simulate_momentum_d1_exit_no_lookahead(
    data: dict, initial_capital: float = 100_000.0,
    mom_d1_threshold: float = -0.02,
) -> dict:
    """无未来函数口径下, 动量一阶导提前退出模拟.

    T日信号 → T+1开盘成交, 与 run_qixing_v3_no_lookahead 一致.
    """
    common_dates: set = set()
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if rq.DEFENSE in data:
        common_dates &= set(data[rq.DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::rq.REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[dict] = []
    n_trades = 0
    d1_exit_log: list[dict] = []
    pending_signal: dict | None = None

    for i, td in enumerate(trading_dates):
        # 执行昨日pending信号
        if pending_signal:
            target = pending_signal["target"]
            if target != holding:
                if holding and holding in data:
                    can_sell, _ = rq._check_tradable(data, holding, td)
                    if can_sell:
                        row = data[holding][data[holding]["trade_date"] == td]
                        if not row.empty:
                            price = float(row.iloc[0]["open"])
                            cash += holding_shares * price * (1 - rq.FEE - rq.SLIPPAGE)
                            n_trades += 1
                            holding = None
                            holding_shares = 0

                if target and target in data:
                    can_buy, _ = rq._check_tradable(data, target, td)
                    if can_buy:
                        row = data[target][data[target]["trade_date"] == td]
                        if not row.empty:
                            price = float(row.iloc[0]["open"])
                            shares = int(cash * 0.99 / price / 100) * 100
                            if shares > 0:
                                cost = shares * price * (1 + rq.FEE + rq.SLIPPAGE)
                                cash -= cost
                                holding = target
                                holding_shares = shares
                                n_trades += 1
            pending_signal = None

        # 每日净值
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE})

        # 调仓日生成信号
        if td in rebalance_set:
            etf_data_at_date = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            d1_triggered = False
            if holding and holding in etf_data_at_date:
                idx = etf_data_at_date[holding]
                close = data[holding]["close"].values[: idx + 1].astype(float)
                if len(close) >= 121:
                    mom_d1 = calc_momentum_derivative(close)
                    if mom_d1 < mom_d1_threshold:
                        d1_triggered = True
                        d1_exit_log.append({
                            "date": str(td), "code": holding,
                            "name": rq.ETF_POOL.get(holding, "货币基金"),
                            "mom_d1": round(mom_d1, 4),
                            "threshold": mom_d1_threshold,
                        })

            if d1_triggered:
                target = rq.DEFENSE
            else:
                target, _candidates, _best_score, _a_share_weak = rq.select_target(
                    data, etf_data_at_date, holding
                )

            next_td = trading_dates[i + 1] if i + 1 < len(trading_dates) else None
            pending_signal = {"target": target, "holding": holding}

    if not equity_history:
        return {"total_return": 0, "ann_return": 0, "sharpe": 0, "max_drawdown": 0}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "n_trades": n_trades, "d1_exit_count": len(d1_exit_log),
        "d1_exit_log": d1_exit_log, "equity_curve": eq_df,
    }



def simulate_d1_as_scoring_factor(
    data: dict, initial_capital: float = 100_000.0,
    d1_weight: float = 0.5,
) -> dict:
    """将动量一阶导作为评分因子 (而非硬退出).

    综合评分 = 动量 + d1_weight * 动量一阶导
    这样既鼓励动量上升, 又惩罚动量衰减.
    """
    common_dates: set = set()
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if rq.DEFENSE in data:
        common_dates &= set(data[rq.DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::rq.REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[dict] = []
    n_trades = 0

    for td in trading_dates:
        if td in rebalance_set:
            etf_data_at_date = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            a_share_weak = (rq.check_a_share_weak(data, etf_data_at_date.get(rq.A_SHARE_ETF, 0))
                            if rq.USE_A_SHARE_FILTER else False)

            candidates = []
            for code in rq.ETF_POOL:
                if code not in etf_data_at_date:
                    continue
                if code == rq.A_SHARE_ETF and a_share_weak:
                    continue
                idx = etf_data_at_date[code]
                df = data[code]
                close = df["close"].values[: idx + 1].astype(float)
                if len(close) < 121:
                    continue
                if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
                    continue
                mom = calc_momentum(close)
                mom_d1 = calc_momentum_derivative(close)
                score = mom + d1_weight * mom_d1
                if mom > 0:
                    candidates.append((code, score))

            candidates.sort(key=lambda x: -x[1])
            best_target = candidates[0][0] if candidates else rq.DEFENSE
            best_score = candidates[0][1] if candidates else 0
            threshold = 0.0 if best_score > 0.10 else 0.05

            target = best_target
            if holding and holding != rq.DEFENSE:
                cur = dict(candidates).get(holding, -999)
                if cur > 0:
                    target = best_target if best_score > cur + threshold else holding
                else:
                    target = best_target

            if target != holding:
                if holding and holding in data:
                    row = data[holding][data[holding]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        cash += holding_shares * price * (1 - rq.FEE - rq.SLIPPAGE)
                        n_trades += 1
                        holding = None
                        holding_shares = 0

                if target in data:
                    row = data[target][data[target]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + rq.FEE + rq.SLIPPAGE)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            n_trades += 1

        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE})

    if not equity_history:
        return {"total_return": 0, "ann_return": 0, "sharpe": 0, "max_drawdown": 0}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "n_trades": n_trades, "equity_curve": eq_df,
    }


def main():
    print("加载数据...")
    data = rq.load_data()
    print(f"  已加载 {len(data)} 只ETF\n")

    # === 1. 运行V3回测 + 详细持仓期追踪 ===
    print("=== 第1步: 运行V3回测 + 详细持仓期追踪 ===")
    analysis = run_v3_with_holding_analysis(data)

    hp = analysis["holding_periods"]
    decisions = analysis["detailed_decisions"]
    dd_events = analysis["drawdown_events"]
    print(f"  持仓期数: {len(hp)}")
    print(f"  决策记录: {len(decisions)}条")
    print(f"  回撤事件: {len(dd_events)}次\n")

    # === 2. 动量模式分析 ===
    print("=== 第2步: 动量模式分析 ===")
    momentum_patterns = analyze_momentum_patterns(hp)
    analysis["momentum_patterns"] = momentum_patterns

    # 保存详细分析结果
    output = {
        "v3_baseline": {
            "total_return": analysis["total_return"],
            "ann_return": analysis["ann_return"],
            "sharpe": analysis["sharpe"],
            "max_drawdown": analysis["max_drawdown"],
        },
        "momentum_patterns": {k: v for k, v in momentum_patterns.items()
                              if k != "by_etf"},
        "drawdown_events": dd_events,
        "holding_period_count": len(hp),
        "decision_count": len(decisions),
    }
    with open(OUTPUT_DIR / "momentum_decay_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    print(f"  分析结果已保存: {OUTPUT_DIR / 'momentum_decay_analysis.json'}\n")

    # === 3. 动量一阶导退出模拟 ===
    print("=== 第3步: 动量一阶导退出信号模拟 ===")
    thresholds = [0.0, -0.01, -0.02, -0.03, -0.05, -0.08]
    d1_results = []
    for threshold in thresholds:
        res = simulate_momentum_d1_exit(data, mom_d1_threshold=threshold)
        d1_results.append((threshold, res))
        name = f"d1_exit_{threshold:+.2f}".replace(".", "p").replace("-", "neg")
        print(f"  阈值 {threshold:+6.1%}: 年化{res['ann_return']:>+7.1%}  "
              f"夏普{res['sharpe']:.2f}  回撤{res['max_drawdown']:.1%}  "
              f"触发{res['d1_exit_count']}次")

    # === 4. 打印分析报告 ===
    print("\n\n")
    v3_baseline = {
        "total_return": analysis["total_return"],
        "ann_return": analysis["ann_return"],
        "sharpe": analysis["sharpe"],
        "max_drawdown": analysis["max_drawdown"],
    }
    print_analysis_report(analysis, d1_results, dd_events, v3_baseline)

    # === 5. 无未来函数口径下D1验证 ===
    print("\n\n")
    print("=" * 78)
    print("  补充分析A: 无未来函数(T+1开盘成交)口径下D1退出信号验证")
    print("=" * 78)
    print(f"\n  V3无未来函数基准: 年化+48.8%  夏普1.33  回撤-40.7%")
    print(f"\n  {'阈值':<8} {'年化':>8} {'夏普':>8} {'回撤':>8} {'交易':>6} {'触发':>6}")
    print(f"  {'-' * 46}")
    nl_thresholds = [0.0, -0.02, -0.05, -0.08]
    for threshold in nl_thresholds:
        res = simulate_momentum_d1_exit_no_lookahead(data, mom_d1_threshold=threshold)
        print(f"  {threshold:<+8.1%} {res['ann_return']:>+8.1%} {res['sharpe']:>8.2f} "
              f"{res['max_drawdown']:>8.1%} {res['n_trades']:>6} {res['d1_exit_count']:>6}")

    # === 6. D1作为评分因子(非硬退出) ===
    print("\n")
    print("=" * 78)
    print("  补充分析B: 动量一阶导作为评分因子 (综合评分 = 动量 + w * D1)")
    print("=" * 78)
    print(f"\n  {'权重w':<8} {'年化':>8} {'夏普':>8} {'回撤':>8} {'交易':>6}")
    print(f"  {'-' * 42}")
    d1_weights = [0.0, 0.3, 0.5, 1.0, 2.0]
    for w in d1_weights:
        res = simulate_d1_as_scoring_factor(data, d1_weight=w)
        print(f"  {w:<8.1f} {res['ann_return']:>+8.1%} {res['sharpe']:>8.2f} "
              f"{res['max_drawdown']:>8.1%} {res['n_trades']:>6}")

    # === 7. 保存详细持仓期数据 ===
    hp_records = []
    for period in hp:
        record = {k: v for k, v in period.items() if k != "momentum_trace"}
        record["momentum_trace"] = [
            {k: v for k, v in m.items()} for m in period.get("momentum_trace", [])
        ]
        hp_records.append(record)

    with open(OUTPUT_DIR / "holding_periods_detail.json", "w") as f:
        json.dump(hp_records, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  详细持仓期数据已保存: {OUTPUT_DIR / 'holding_periods_detail.json'}")


if __name__ == "__main__":
    main()