"""实验: 七星V3 单日跌幅过滤窗口对比 (3日 vs 5日).

对比 DROP_LOOKBACK=3 (当前) 与 DROP_LOOKBACK=5 的年度收益差异.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as base


def run_with_lookback(data: dict, lookback: int) -> dict:
    """修改跌幅窗口后跑回测."""
    original = base.DROP_LOOKBACK
    base.DROP_LOOKBACK = lookback
    try:
        result = base.run_qixing_v3(data)
    finally:
        base.DROP_LOOKBACK = original
    return result


def main():
    print("=" * 70)
    print("  实验: 七星V3 单日跌幅过滤窗口对比")
    print("  DROP_LOOKBACK: 3日(当前) vs 5日")
    print("=" * 70)

    data = base.load_data()
    print(f"\n  数据: {len(data)}只ETF")

    results = {}
    for lb in [3, 5]:
        print(f"\n  >>> 跑 DROP_LOOKBACK={lb} ...")
        results[lb] = run_with_lookback(data, lb)

    # === 总体对比 ===
    print(f"\n{'=' * 70}")
    print(f"  总体指标对比")
    print(f"{'=' * 70}")
    print(f"  {'指标':<12} {'3日窗口':>12} {'5日窗口':>12} {'差异':>12}")
    print(f"  {'-' * 50}")

    r3, r5 = results[3], results[5]
    metrics = [
        ("总收益", r3["total_return"], r5["total_return"], True),
        ("年化收益", r3["ann_return"], r5["ann_return"], True),
        ("夏普比率", r3["sharpe"], r5["sharpe"], False),
        ("最大回撤", r3["max_drawdown"], r5["max_drawdown"], True),
        ("交易次数", r3["n_trades"], r5["n_trades"], False),
    ]
    for name, v3, v5, is_pct in metrics:
        if is_pct:
            print(f"  {name:<12} {v3:>+12.2%} {v5:>+12.2%} {v5 - v3:>+12.2%}")
        else:
            print(f"  {name:<12} {v3:>12.1f} {v5:>12.1f} {v5 - v3:>+12.1f}")

    # === 年度对比 ===
    print(f"\n{'=' * 70}")
    print(f"  年度收益对比")
    print(f"{'=' * 70}")
    print(f"  {'年份':<6} {'3日窗口':>10} {'5日窗口':>10} {'差异':>10} {'胜者':>6}")
    print(f"  {'-' * 46}")

    all_years = sorted(set(r3["yearly"].keys()) | set(r5["yearly"].keys()))
    wins_3, wins_5 = 0, 0
    for year in all_years:
        ret3 = r3["yearly"].get(year, {}).get("return", 0)
        ret5 = r5["yearly"].get(year, {}).get("return", 0)
        diff = ret5 - ret3
        winner = "5日" if diff > 0.001 else ("3日" if diff < -0.001 else "平")
        if diff > 0.001:
            wins_5 += 1
        elif diff < -0.001:
            wins_3 += 1
        print(f"  {year:<6} {ret3:>+10.2%} {ret5:>+10.2%} {diff:>+10.2%} {winner:>6}")

    print(f"  {'-' * 46}")
    print(f"  5日胜: {wins_5}年 | 3日胜: {wins_3}年")

    # === 年度回撤对比 ===
    print(f"\n{'=' * 70}")
    print(f"  年度最大回撤对比")
    print(f"{'=' * 70}")
    print(f"  {'年份':<6} {'3日窗口':>10} {'5日窗口':>10} {'差异':>10}")
    print(f"  {'-' * 40}")
    for year in all_years:
        dd3 = r3["yearly"].get(year, {}).get("max_dd", 0)
        dd5 = r5["yearly"].get(year, {}).get("max_dd", 0)
        print(f"  {year:<6} {dd3:>10.2%} {dd5:>10.2%} {dd5 - dd3:>+10.2%}")

    # === 结论 ===
    print(f"\n{'=' * 70}")
    ann_diff = r5["ann_return"] - r3["ann_return"]
    dd_diff = r5["max_drawdown"] - r3["max_drawdown"]
    sharpe_diff = r5["sharpe"] - r3["sharpe"]
    print(f"  结论:")
    print(f"    年化收益变化: {ann_diff:+.2%}")
    print(f"    最大回撤变化: {dd_diff:+.2%} (负=改善)")
    print(f"    夏普变化:     {sharpe_diff:+.3f}")
    if ann_diff > 0 and sharpe_diff > 0:
        print(f"    → 5日窗口优于3日窗口")
    elif ann_diff < 0 and sharpe_diff < 0:
        print(f"    → 3日窗口优于5日窗口")
    else:
        print(f"    → 两者各有优劣,需权衡")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
