"""实验: 滚动4年窗口 — 任意时点入场持有4年的收益分布.

从2020-07起每3个月取一个入场点, 各持有4年, 统计最终金额/年化分布与盈利概率。
用法: uv run python scripts/exp_rolling_4y.py
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
CAPITAL = 100_000.0
DATA_END = date(2026, 7, 21)


def add_months(d: date, m: int) -> date:
    y = d.year + (d.month - 1 + m) // 12
    mo = (d.month - 1 + m) % 12 + 1
    return date(y, mo, 1)


def main() -> None:
    data = rq.load_data()
    # 生成滚动窗口: 起点每3个月一步, 直到 起点+4年 <= 数据末尾
    windows = []
    s = date(2020, 7, 1)
    while add_months(s, 48) <= DATA_END:
        windows.append((s, add_months(s, 48)))
        s = add_months(s, 3)

    print("=" * 70)
    print("  滚动4年窗口 · V3策略 · 10万本金 (每3个月一个入场点, 持有4年)")
    print("=" * 70)
    print(f"  {'入场':<13}{'离场':<13}{'10万→':<12}{'年化':>9}{'回撤':>9}")
    print("  " + "-" * 60)

    results = []
    for s, e in windows:
        res = backtest(data, v3_select, PARAMS, 5, initial=CAPITAL, start_date=s, end_date=e)
        results.append((s, e, res["final_equity"], res["ann_return"], res["max_drawdown"]))
        fw = res["final_equity"] / 10000
        print(
            f"  {str(s):<13}{str(e):<13}{f'{fw:.1f}万':<12}"
            f"{res['ann_return'] * 100:>+8.1f}%{res['max_drawdown'] * 100:>8.1f}%"
        )

    finals = np.array([r[2] for r in results])
    anns = np.array([r[3] for r in results])
    print("  " + "-" * 60)
    print("  【4年后最终金额分布】")
    print(
        f"    最低 {finals.min() / 10000:.1f}万 | 中位 {np.median(finals) / 10000:.1f}万 | "
        f"平均 {finals.mean() / 10000:.1f}万 | 最高 {finals.max() / 10000:.1f}万"
    )
    print("  【年化收益分布】")
    print(
        f"    最低 {anns.min() * 100:+.1f}% | 中位 {np.median(anns) * 100:+.1f}% | "
        f"平均 {anns.mean() * 100:+.1f}% | 最高 {anns.max() * 100:+.1f}%"
    )
    profit = int((finals > CAPITAL).sum())
    print(f"  【4年后盈利概率】 {profit}/{len(finals)} = {profit / len(finals) * 100:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
