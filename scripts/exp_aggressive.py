"""激进版V3 vs 稳健版V3: 关闭防御(永不空仓)+总是切最强, 看收益/回撤的真实代价.

用法: uv run python scripts/exp_aggressive.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import backtest  # noqa: E402
from strategy_lab.strategies import v3_select  # noqa: E402

PARAMS = {"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5}


def v3_aggressive_select(data: dict, idx_map: dict, holding, params: dict):
    """激进版: 永远选动量最强(即使<0), 永不主动防御, 总是切换."""
    best, best_score = None, -999.0
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        s = rq.calc_momentum_score(close)
        if s > best_score:
            best_score, best = s, code
    return best if best else rq.DEFENSE


def add_months(d: date, m: int) -> date:
    y = d.year + (d.month - 1 + m) // 12
    return date(y, (d.month - 1 + m) % 12 + 1, 1)


def main() -> None:
    data = rq.load_data()
    print("=" * 68)
    print("  V3 稳健版 vs 激进版 | 10万本金 | 全段~6年")
    print("=" * 68)

    versions = [
        ("稳健版V3 (防御+换仓缓冲)", v3_select),
        ("激进版V3 (永不空仓+总切最强)", v3_aggressive_select),
    ]
    for name, sel in versions:
        res = backtest(data, sel, PARAMS, 5)
        print(f"\n  【{name}】")
        print(f"    10万 → {res['final_equity']/10000:.1f}万  "
              f"(累计 {res['total_return']*100:+.0f}%, 年化 {res['ann_return']*100:+.1f}%)")
        print(f"    夏普 {res['sharpe']:.2f} | 最大回撤 {res['max_drawdown']*100:.1f}% | "
              f"交易 {res['n_trades']} 次")
        yr = "  ".join(f"{y}:{v['return']*100:+.0f}%" for y, v in sorted(res["yearly"].items()))
        print(f"    逐年: {yr}")

    # 滚动4年: 看激进版的最坏情况
    print("\n" + "=" * 68)
    print("  激进版 滚动4年 (任意时点入场拿4年) — 重点看回撤")
    print("=" * 68)
    print(f"  {'入场':<13}{'10万→':<12}{'年化':>9}{'最大回撤':>10}")
    print("  " + "-" * 48)
    s = date(2020, 7, 1)
    finals, dds = [], []
    while add_months(s, 48) <= date(2026, 7, 21):
        res = backtest(data, v3_aggressive_select, PARAMS, 5,
                       initial=100000.0, start_date=s, end_date=add_months(s, 48))
        finals.append(res["final_equity"])
        dds.append(res["max_drawdown"] * 100)
        fw = res["final_equity"] / 10000
        print(f"  {str(s):<13}{f'{fw:.1f}万':<12}"
              f"{res['ann_return']*100:>+8.1f}%{res['max_drawdown']*100:>9.1f}%")
        s = add_months(s, 3)
    print("  " + "-" * 48)
    print(f"  激进版4年: 金额 {min(finals)/10000:.1f}万~{max(finals)/10000:.1f}万 | "
          f"最深回撤 {min(dds):.1f}%")
    print("=" * 68)


if __name__ == "__main__":
    main()
