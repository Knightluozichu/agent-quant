"""实验: 动量权重敏感性测试.

固定动量周期=(10日,20日), 只变权重组合, 跑全段(~6年)回测, 对比总收益。
用法: uv run python scripts/exp_mom_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from strategy_lab.engine import backtest
from strategy_lab.strategies import v3_select

# 权重组合: (10日权重, 20日权重)
WEIGHT_COMBOS = [
    (0.5, 0.5),  # 当前V3基准
    (0.2, 0.8),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.7, 0.3),
]


def main() -> None:
    data = rq.load_data()
    base = {"mom_periods": (10, 20), "rebalance_days": 5}

    print("=" * 66)
    print("  动量权重敏感性测试  |  周期固定(10日,20日)  |  全段~6年回测")
    print("=" * 66)
    print(f"  {'权重(10日/20日)':<16}{'总收益':>12}{'年化':>10}{'夏普':>8}{'最大回撤':>10}")
    print("  " + "-" * 60)

    results = []
    for w in WEIGHT_COMBOS:
        params = dict(base)
        params["mom_weights"] = w
        res = backtest(data, v3_select, params, 5)
        results.append((w, res))
        tag = "  ←当前" if w == (0.5, 0.5) else ""
        print(
            f"  {w!s:<16}{res['total_return'] * 100:>+11.1f}%"
            f"{res['ann_return'] * 100:>+9.1f}%{res['sharpe']:>8.2f}"
            f"{res['max_drawdown'] * 100:>9.1f}%{tag}"
        )

    # 找最优
    best_w, best_res = max(results, key=lambda x: x[1]["total_return"])
    print("  " + "-" * 60)
    print(f"  最高总收益: 权重{best_w} → {best_res['total_return'] * 100:+.1f}%")
    print("=" * 66)


if __name__ == "__main__":
    main()
