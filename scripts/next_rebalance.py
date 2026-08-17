"""下个调仓日预览: 集合最新数据, 输出下个调仓日 + 调仓预览(买卖什么/多少股).

用途: 提前知道下个调仓日是哪天、还有几个交易日、届时大概率要买/卖什么,
      便于提前准备资金和操作。注意: 预览基于最新数据, 实际信号以调仓日当天数据为准。
用法: uv run python scripts/next_rebalance.py [--update]
  --update  先增量更新数据到最新(需网络/akshare)
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import live_signal as ls
import run_qixing_v3 as rq
from strategy_lab.engine import build_idx_map, get_common_dates

REBALANCE_DAYS = rq.REBALANCE_DAYS


def find_next_rebalance(trading_dates: list, last_rb_str: str | None):
    """返回 (下个调仓日, 距今交易日数). 数据不够时按日历估算。"""
    if last_rb_str is None:
        # 首次调仓: 立即(最新交易日即为调仓日)
        return (trading_dates[-1] if trading_dates else None, 0)
    last_rb = dt.datetime.strptime(last_rb_str, "%Y-%m-%d").date()
    # last_rb 在 trading_dates 中的索引
    idx = None
    for i, d in enumerate(trading_dates):
        if d == last_rb:
            idx = i
            break
    if idx is not None and idx + REBALANCE_DAYS < len(trading_dates):
        nxt = trading_dates[idx + REBALANCE_DAYS]
        # 距今交易日数
        latest_idx = len(trading_dates) - 1
        days_left = (idx + REBALANCE_DAYS) - latest_idx
        return nxt, days_left
    # 数据未覆盖下个调仓日 → 按日历估算(5交易日≈7自然日)
    est = last_rb + dt.timedelta(days=7)
    return est, None


def preview_trade(data, state, target, latest):
    """基于最新价格和账户状态, 估算调仓买卖(股数/金额)."""
    cash = state["cash"]
    holding = state.get("holding")
    shares = state.get("shares", 0)
    actions = []
    avail_cash = cash
    if target != holding:
        if holding and holding in data:
            p = ls.price_on(data, holding, latest)
            if p:
                proceeds = shares * p * (1 - rq.FEE - rq.SLIPPAGE)
                actions.append(("卖出", ls.name_of(holding), holding, shares, p, proceeds))
                avail_cash += proceeds
        if target and target in data and target != rq.DEFENSE:
            p = ls.price_on(data, target, latest)
            if p:
                buy_shares = int(avail_cash * 0.99 / p / 100) * 100
                if buy_shares > 0:
                    cost = buy_shares * p * (1 + rq.FEE + rq.SLIPPAGE)
                    actions.append(("买入", ls.name_of(target), target, buy_shares, p, cost))
    return actions


def main() -> None:
    update = "--update" in sys.argv
    data = ls.load_data()
    if update:
        data = ls.update_data(data)

    state = ls.load_state()
    if state is None:
        print("  ❌ 账户未初始化")
        return

    trading_dates = get_common_dates(data)
    latest = trading_dates[-1]
    last_rb = state.get("last_rebalance_date")
    holding = state.get("holding")

    next_rb, days_left = find_next_rebalance(trading_dates, last_rb)

    # 基于最新数据预览信号
    idx_map = build_idx_map(data, latest)
    target, candidates, _best_score, a_share_weak = rq.select_target(data, idx_map, holding)

    print("=" * 62)
    print("  七星V3 · 下个调仓日预览")
    print("=" * 62)
    print(f"  最新数据日   : {latest}")
    print(f"  上次调仓日   : {last_rb or '(无, 首次)'}")
    if days_left is not None:
        print(f"  下个调仓日   : {next_rb}  (还有 {days_left} 个交易日)")
    else:
        print(f"  下个调仓日   : 约 {next_rb} (数据未覆盖, 按日历估算)")
    hold_name = ls.name_of(holding) if holding else "空仓(货币)"
    print(f"  当前持仓     : {hold_name}" + (f" ({state.get('shares', 0)}股)" if holding else ""))
    if a_share_weak:
        print("  ⚠️  A股走弱(创业板<MA20), 已排除创业板")

    print("\n  【最新动量排行】")
    ranked = sorted(candidates, key=lambda x: -x[1])
    for code, score in ranked:
        tag = " ◀当前目标" if code == target else ""
        print(f"    {ls.name_of(code):<10} {score * 100:+6.2f}%{tag}")

    print("\n  【调仓预览】(基于最新数据, 实际以调仓日当天为准)")
    if target == holding:
        print(f"    → 继续持有 {ls.name_of(holding)}, 无需操作")
    else:
        actions = preview_trade(data, state, target, latest)
        if not actions:
            print(f"    → 目标 {ls.name_of(target)} (无价格数据, 无法估算)")
        for act, name, code, sh, p, amt in actions:
            print(f"    → {act} {name}({code})  {sh}股 @ {p:.3f} ≈ {amt:,.0f}元")

    print("=" * 62)
    print("  提示: 此为预览。调仓日14:50系统会推送正式信号, 以届时为准。")


if __name__ == "__main__":
    main()
