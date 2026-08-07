"""V3策略动量衰减改进版 — 基于量化分析结果.

核心观察 (来自 exp_momentum_decay_analysis.py):
  1. 80%持仓期有动量衰减, 65%呈下降趋势 — 衰减是动量策略固有属性
  2. 74%退出时动量一阶导(D1)为负, 但D1硬退出在无未来函数口径下完全失效
  3. D1作为评分因子(w=0.3)有轻微改善 (夏普2.09→2.10, 同日口径)
  4. 纳指ETF衰减最严重(94.1%持仓期衰减>2%), 且不受A股MA15保护

改进方案:
  V3-D1:   动量一阶导作为评分因子 (综合分 = 动量 + 0.3×D1)
  V3-MDC:  Momentum Decay Consistency — 连续2期动量衰减时降低换仓阈值
  V3-Combo: D1评分 + MDC 组合

用法: uv run python scripts/exp_v3_modified.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "qixing_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================

def calc_momentum(close: np.ndarray, periods=(10, 20), weights=(0.5, 0.5)) -> float:
    """加权动量评分 (与V3一致)."""
    score = 0.0
    for period, weight in zip(periods, weights, strict=False):
        if len(close) > period:
            ret = (close[-1] - close[-period - 1]) / close[-period - 1]
            score += ret * weight
    return score


def calc_momentum_d1(close: np.ndarray, step=5) -> float:
    """动量一阶导 (动量变化率). 正值=加速, 负值=衰减."""
    if len(close) < 121:
        return 0.0
    current_mom = calc_momentum(close)
    if len(close) < step + 1:
        return 0.0
    prev_mom = calc_momentum(close[:-step])
    return current_mom - prev_mom


# ============================================================
# select 函数 — 可插拔
# ============================================================

def select_v3_vanilla(data, idx_map, holding, params) -> str:
    """纯V3选股 (基准)."""
    a_share_weak = (rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
                    if rq.USE_A_SHARE_FILTER else False)
    candidates = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        df = data[code]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        score = calc_momentum(close)
        if score > 0:
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0
    threshold = 0.0 if best_score > 0.10 else 0.05
    if holding and holding != rq.DEFENSE:
        cur = dict(candidates).get(holding, -999)
        if cur > 0:
            target = best_target if best_score > cur + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target


def select_v3_d1_weight(data, idx_map, holding, params) -> str:
    """V3 + D1评分因子: 综合分 = 动量 + w × D1."""
    w = params.get("d1_weight", 0.3)
    a_share_weak = (rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
                    if rq.USE_A_SHARE_FILTER else False)
    candidates = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        df = data[code]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        mom = calc_momentum(close)
        d1 = calc_momentum_d1(close)
        score = mom + w * d1
        if mom > 0:  # 仍要求正动量
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0
    threshold = 0.0 if best_score > 0.10 else 0.05
    if holding and holding != rq.DEFENSE:
        cur = dict(candidates).get(holding, -999)
        if cur > 0:
            target = best_target if best_score > cur + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target


def select_v3_mdc(data, idx_map, holding, params) -> str:
    """V3 + Momentum Decay Consistency.

    若当前持仓连续2期动量一阶导为负, 将换仓阈值从0.05降至0.0,
    使策略更容易切换到新目标.
    """
    mom_history = params.get("_mom_history", {})
    a_share_weak = (rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
                    if rq.USE_A_SHARE_FILTER else False)
    candidates = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        df = data[code]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        score = calc_momentum(close)
        if score > 0:
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0

    # 计算当前持仓的D1方向
    mdc_trigger = False
    if holding and holding in idx_map and holding != rq.DEFENSE:
        idx = idx_map[holding]
        df = data[holding]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) >= 121:
            d1 = calc_momentum_d1(close)
            # 更新持仓的动量历史
            if holding not in mom_history:
                mom_history[holding] = []
            mom_history[holding].append(d1)
            # 保留最近2期
            if len(mom_history[holding]) > 2:
                mom_history[holding] = mom_history[holding][-2:]
            # 连续2期D1为负
            if len(mom_history[holding]) >= 2 and all(d < 0 for d in mom_history[holding]):
                mdc_trigger = True

    threshold = 0.0 if best_score > 0.10 else (0.0 if mdc_trigger else 0.05)
    if holding and holding != rq.DEFENSE:
        cur = dict(candidates).get(holding, -999)
        if cur > 0:
            target = best_target if best_score > cur + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target


def select_v3_combo(data, idx_map, holding, params) -> str:
    """V3-D1 + V3-MDC 组合."""
    w = params.get("d1_weight", 0.3)
    mom_history = params.get("_mom_history", {})
    a_share_weak = (rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
                    if rq.USE_A_SHARE_FILTER else False)
    candidates = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        df = data[code]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        mom = calc_momentum(close)
        d1 = calc_momentum_d1(close)
        score = mom + w * d1
        if mom > 0:
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0

    mdc_trigger = False
    if holding and holding in idx_map and holding != rq.DEFENSE:
        idx = idx_map[holding]
        df = data[holding]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) >= 121:
            d1 = calc_momentum_d1(close)
            if holding not in mom_history:
                mom_history[holding] = []
            mom_history[holding].append(d1)
            if len(mom_history[holding]) > 2:
                mom_history[holding] = mom_history[holding][-2:]
            if len(mom_history[holding]) >= 2 and all(d < 0 for d in mom_history[holding]):
                mdc_trigger = True

    threshold = 0.0 if best_score > 0.10 else (0.0 if mdc_trigger else 0.05)
    if holding and holding != rq.DEFENSE:
        cur = dict(candidates).get(holding, -999)
        if cur > 0:
            target = best_target if best_score > cur + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target


# ============================================================
# 回测引擎 (无未来函数: T日信号 → T+1开盘成交)
# ============================================================

def backtest_no_lookahead(
    data: dict,
    select_fn,
    params: dict,
    initial_capital: float = 100_000.0,
) -> dict:
    """无未来函数回测 (T日收盘信号 → T+1开盘成交).

    与 run_qixing_v3_no_lookahead 逻辑一致, 但使用可插拔 select_fn.
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

    # 状态
    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    pending_signal: dict | None = None
    equity_history: list[dict] = []
    trade_log: list[dict] = []
    decision_log: list[dict] = []
    n_trades = 0
    signal_counter = 0

    # 初始化 select_fn 的持久状态 (如 _mom_history)
    fn_params = dict(params)

    for i, td in enumerate(trading_dates):
        # 执行昨日pending信号
        if pending_signal:
            sig_id = pending_signal["signal_id"]
            target = pending_signal["target"]

            if target != holding:
                sell_ok = True
                if holding and holding in data:
                    can_sell, reason = rq._check_tradable(data, holding, td)
                    if not can_sell:
                        sell_ok = False
                        trade_log.append({
                            "signal_id": sig_id, "date": str(td),
                            "action": "sell", "code": holding,
                            "status": "cancelled", "reason": reason,
                        })
                    else:
                        row = data[holding][data[holding]["trade_date"] == td]
                        price = float(row.iloc[0]["open"])
                        cash += holding_shares * price * (1 - rq.FEE - rq.SLIPPAGE)
                        trade_log.append({
                            "signal_id": sig_id, "date": str(td),
                            "action": "sell", "code": holding,
                            "shares": holding_shares, "price": price,
                            "status": "executed",
                        })
                        holding = None
                        holding_shares = 0

                if sell_ok and target and target in data:
                    can_buy, reason = rq._check_tradable(data, target, td)
                    if not can_buy:
                        trade_log.append({
                            "signal_id": sig_id, "date": str(td),
                            "action": "buy", "code": target,
                            "status": "cancelled", "reason": reason,
                        })
                    else:
                        row = data[target][data[target]["trade_date"] == td]
                        price = float(row.iloc[0]["open"])
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + rq.FEE + rq.SLIPPAGE)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            trade_log.append({
                                "signal_id": sig_id, "date": str(td),
                                "action": "buy", "code": target,
                                "shares": shares, "price": price,
                                "amount": round(cost, 2),
                                "status": "executed",
                            })

            pending_signal = None

        # 每日净值
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE})

        # 调仓日: 生成信号
        if td in rebalance_set:
            idx_map = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                idx_map[code] = mask.sum() - 1

            target = select_fn(data, idx_map, holding, fn_params)
            if target is None:
                target = rq.DEFENSE

            signal_counter += 1
            sig_id = f"SIG-{signal_counter:06d}"
            next_td = trading_dates[i + 1] if i + 1 < len(trading_dates) else None
            pending_signal = {
                "signal_id": sig_id,
                "signal_date": td,
                "target": target,
                "holding": holding,
            }
            decision_log.append({
                "date": str(td), "signal_id": sig_id,
                "target": target, "holding": holding or rq.DEFENSE,
                "execution_date": str(next_td) if next_td else None,
            })

    # 最后一个信号未执行
    if pending_signal:
        trade_log.append({
            "signal_id": pending_signal["signal_id"],
            "date": str(pending_signal["signal_date"]),
            "action": "none", "code": pending_signal["target"],
            "status": "unexecuted", "reason": "最后交易日无T+1可执行",
        })

    if not equity_history:
        return {"error": "no data", "equity_curve": pd.DataFrame()}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    # 年度统计
    yearly = {}
    prev_val = initial_capital
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": yr, "max_dd": dd}
        prev_val = end_val

    n_executed = sum(1 for t in trade_log if t.get("status") == "executed")
    n_cancelled = sum(1 for t in trade_log if t.get("status") in ("cancelled", "unexecuted"))

    # 持仓分布
    from collections import Counter
    holding_counts = Counter(eq_df["holding"].tolist())

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "yearly": yearly, "n_trades": n_executed,
        "n_cancelled": n_cancelled,
        "equity_curve": eq_df, "trade_log": trade_log,
        "decision_log": decision_log,
        "holding_distribution": dict(holding_counts.most_common()),
        "final_equity": float(eq_df["equity"].iloc[-1]),
    }


# ============================================================
# 全口径对比 (同日收盘 + T+1开盘)
# ============================================================

def backtest_same_day(
    data: dict,
    select_fn,
    params: dict,
    initial_capital: float = 100_000.0,
) -> dict:
    """同日收盘成交回测 (对齐实盘14:50)."""
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
    trade_log: list[dict] = []
    n_trades = 0
    fn_params = dict(params)

    for td in trading_dates:
        if td in rebalance_set:
            idx_map = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                idx_map[code] = mask.sum() - 1

            target = select_fn(data, idx_map, holding, fn_params)
            if target is None:
                target = rq.DEFENSE

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
        return {"error": "no data", "equity_curve": pd.DataFrame()}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    yearly = {}
    prev_val = initial_capital
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": yr, "max_dd": dd}
        prev_val = end_val

    from collections import Counter
    holding_counts = Counter(eq_df["holding"].tolist())

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "yearly": yearly, "n_trades": n_trades,
        "equity_curve": eq_df, "trade_log": trade_log,
        "holding_distribution": dict(holding_counts.most_common()),
        "final_equity": float(eq_df["equity"].iloc[-1]),
    }


# ============================================================
# 报告打印
# ============================================================

def print_comparison(name: str, res: dict, initial: float):
    """打印单策略详细报告."""
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return

    eq = res["equity_curve"]
    final = res["final_equity"]

    print(f"\n  {'=' * 60}")
    print(f"  {name}")
    print(f"  {'=' * 60}")
    print(f"  初始资金: {initial:>10,.0f}")
    print(f"  最终资产: {final:>10,.0f}")
    print(f"  总收益:   {res['total_return']:>+10.1%}")
    print(f"  年化收益: {res['ann_return']:>+10.1%}")
    print(f"  夏普比率: {res['sharpe']:>10.2f}")
    print(f"  最大回撤: {res['max_drawdown']:>10.1%}")
    print(f"  交易次数: {res['n_trades']:>10}次")
    if "n_cancelled" in res:
        print(f"  取消次数: {res['n_cancelled']:>10}次")

    print(f"\n  年度明细:")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
    print(f"  {'-' * 46}")
    prev = initial
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        print(f"  {year:<6} {prev:>10,.0f} {end_val:>10,.0f} {yr:>+8.1%} {dd:>8.1%}")
        prev = end_val

    print(f"\n  持仓分布 (共{len(eq)}天):")
    dist = res.get("holding_distribution", {})
    total_days = len(eq)
    for code, count in sorted(dist.items(), key=lambda x: -x[1]):
        name_etf = rq.ETF_POOL.get(code, "货币基金")
        print(f"    {name_etf:<10} {count:>6}天 ({count/total_days:>5.1%})")


def print_all_results(results: dict, initial: float):
    """打印所有策略对比."""
    print("\n" + "=" * 78)
    print("  V3策略改进版 — 完整对比分析")
    print("  本金: 10万 | 成交: T+1开盘 (无未来函数) | 周期: 6年")
    print("=" * 78)

    # 汇总表
    print(f"\n  {'策略':<14} {'最终资产':>12} {'年化':>8} {'夏普':>8} {'回撤':>8} {'交易':>6}")
    print(f"  {'-' * 58}")
    for name, res in results.items():
        if "error" in res:
            continue
        print(f"  {name:<14} {res['final_equity']:>12,.0f} {res['ann_return']:>+8.1%} "
              f"{res['sharpe']:>8.2f} {res['max_drawdown']:>8.1%} {res['n_trades']:>6}")

    # 详细报告
    for name, res in results.items():
        print_comparison(name, res, initial)


# ============================================================
# 主入口
# ============================================================

def main():
    data = rq.load_data()
    print(f"加载数据: {len(data)}只ETF")
    INITIAL = 100_000.0

    # === 策略定义 ===
    strategies = {
        "V3基准": (select_v3_vanilla, {}),
        "V3-D1 w=0.3": (select_v3_d1_weight, {"d1_weight": 0.3}),
        "V3-D1 w=0.5": (select_v3_d1_weight, {"d1_weight": 0.5}),
        "V3-MDC": (select_v3_mdc, {}),
        "V3-Combo w=0.3": (select_v3_combo, {"d1_weight": 0.3}),
    }

    results = {}
    for name, (select_fn, params) in strategies.items():
        print(f"\n回测: {name} ...")
        res = backtest_no_lookahead(data, select_fn, params, INITIAL)
        results[name] = res

    # 打印结果
    print_all_results(results, INITIAL)

    # 保存
    summary = {}
    for name, res in results.items():
        if "error" in res:
            summary[name] = {"error": res["error"]}
            continue
        summary[name] = {
            "total_return": res["total_return"],
            "ann_return": res["ann_return"],
            "sharpe": res["sharpe"],
            "max_drawdown": res["max_drawdown"],
            "n_trades": res["n_trades"],
            "final_equity": res["final_equity"],
            "yearly": {str(k): v for k, v in res["yearly"].items()},
        }

    with open(OUTPUT_DIR / "v3_modified_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n结果已保存: {OUTPUT_DIR / 'v3_modified_results.json'}")


if __name__ == "__main__":
    main()