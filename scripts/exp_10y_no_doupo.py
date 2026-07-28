"""实验: 七星V3 去掉豆粕ETF, 跑约10年回测 (2017~2026).

目的: 豆粕ETF(159985)2019年底才上市, 限制了完整池回测只能从2020起.
      去掉豆粕后, 回测可由南方原油(2016-06上市)拉到约2017, 近10年,
      用于检验策略在更长周期(含2017-2019)的稳健性.

注意: 这是独立实验脚本, 不修改生产 run_qixing_v3.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402

# === 去掉豆粕ETF ===
REMOVED = "159985"
del rq.ETF_POOL[REMOVED]
if REMOVED in rq.CATEGORIES.get("商品", []):
    rq.CATEGORIES["商品"].remove(REMOVED)


def main() -> None:
    print("=" * 70)
    print("  七星V3 实验 — 去掉豆粕ETF, 约10年回测")
    print(f"  ETF池: {list(rq.ETF_POOL.values())} + 货币基金(防御)")
    print("=" * 70)

    data = rq.load_data()
    print(f"\n  加载 {len(data)} 只ETF:")
    for code, df in data.items():
        name = rq.ETF_POOL.get(code, "货币基金")
        print(f"    {code} {name:8s}: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    result = rq.run_qixing_v3(data)
    if "error" in result:
        print(f"\n  ❌ 回测失败: {result['error']}")
        return

    eq = result["equity_curve"]
    start = eq["trade_date"].min().strftime("%Y-%m-%d")
    end = eq["trade_date"].max().strftime("%Y-%m-%d")
    years = (eq["trade_date"].max() - eq["trade_date"].min()).days / 365.25

    print("\n" + "=" * 70)
    print(f"  回测区间: {start} ~ {end}  ({years:.1f}年)")
    print("=" * 70)
    print(f"  累计收益:   {result['total_return'] * 100:+.1f}%")
    print(f"  年化收益:   {result['ann_return'] * 100:+.1f}%")
    print(f"  夏普比率:   {result['sharpe']:.2f}")
    print(f"  最大回撤:   {result['max_drawdown'] * 100:.1f}%")
    print(f"  交易次数:   {result['n_trades']}")

    print("\n  【逐年表现】")
    print(f"  {'年份':<6}{'收益':>10}{'最大回撤':>12}")
    for year, y in sorted(result["yearly"].items()):
        print(f"  {year:<6}{y['return'] * 100:>+9.1f}%{y['max_dd'] * 100:>11.1f}%")

    # 对比提示
    print("\n" + "-" * 70)
    print("  对比基准 — 完整池(含豆粕) 2020~2026:")
    print("    累计 +1997.5% | 年化 +65.1% | 夏普 1.73 | 回撤 -19.6%")
    print("-" * 70)


if __name__ == "__main__":
    main()
