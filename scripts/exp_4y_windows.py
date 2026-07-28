"""实验: 10万本金, V3策略, 不同4年窗口回测 → 最终变成多少.

用法: uv run python scripts/exp_4y_windows.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import backtest  # noqa: E402
from strategy_lab.strategies import v3_select  # noqa: E402

PARAMS = {"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5}
CAPITAL = 100_000.0

# 多个4年窗口 (避免挑时段, 看真实区间)
WINDOWS = [
    ("前4年", date(2020, 7, 1), date(2024, 7, 1)),
    ("中段4年", date(2021, 7, 1), date(2025, 7, 1)),
    ("近4年", date(2022, 7, 1), date(2026, 7, 21)),
]


def main() -> None:
    data = rq.load_data()
    print("=" * 72)
    print("  10万本金 · V3策略 · 不同4年窗口回测 (最终账户总值)")
    print("=" * 72)
    print(f"  {'窗口':<10}{'区间':<26}{'10万→':<14}{'累计':>9}{'年化':>9}{'回撤':>9}")
    print("  " + "-" * 68)
    finals = []
    for label, s, e in WINDOWS:
        res = backtest(data, v3_select, PARAMS, 5, initial=CAPITAL, start_date=s, end_date=e)
        final = res["final_equity"]
        finals.append(final)
        print(f"  {label:<10}{str(s)+'~'+str(e):<26}{f'{final/10000:.1f}万':<14}"
              f"{res['total_return']*100:>+8.0f}%{res['ann_return']*100:>+8.1f}%"
              f"{res['max_drawdown']*100:>8.1f}%")
    # 全段参考
    res = backtest(data, v3_select, PARAMS, 5, initial=CAPITAL)
    full_w = res["final_equity"] / 10000
    print("  " + "-" * 68)
    print(f"  {'全段6.1年':<10}{'2020-07~2026-07':<26}{f'{full_w:.1f}万':<14}"
          f"{res['total_return']*100:>+8.0f}%{res['ann_return']*100:>+8.1f}%"
          f"{res['max_drawdown']*100:>8.1f}%")
    print("=" * 72)
    print(f"  4年窗口区间: 最低 {min(finals)/10000:.1f}万 ~ 最高 {max(finals)/10000:.1f}万")
    print("=" * 72)


if __name__ == "__main__":
    main()
