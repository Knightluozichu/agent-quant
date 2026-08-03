"""同日收盘成交回测对比: T日信号 → T日近收盘成交 (对齐实盘 14:50 执行).

实盘流程: 调仓日 14:30 实时数据生成信号 → 14:50~15:00 执行。
信号时刻早于成交时刻, 无未来函数。本实验对比两种成交口径:
  A. 同日收盘成交 (对齐实盘, 日线收盘价近似 14:50 成交价)
  B. T+1开盘成交 (R1 保守基线)

用法: uv run python scripts/exp_same_day_close.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402

INITIAL = 100_000.0


def print_yearly(label: str, result: dict) -> None:
    eq = result["equity_curve"]
    print(f"\n  【{label}】")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
    print(f"  {'-' * 46}")
    prev = INITIAL
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = end_val / prev - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        print(f"  {year:<6} {prev:>10,.0f} {end_val:>10,.0f} {yr:>+8.2%} {dd:>8.2%}")
        prev = end_val
    final = eq["equity"].iloc[-1]
    print(f"  {'-' * 46}")
    print(
        f"  10万 → {final:,.0f} ({final / INITIAL - 1:+.1%}) | 年化 {result['ann_return']:+.1%} | "
        f"夏普 {result['sharpe']:.2f} | 最大回撤 {result['max_drawdown']:.1%} | "
        f"交易 {result['n_trades']}次 (取消 {result['n_cancelled']})"
    )


def main() -> None:
    data = rq.load_data()
    print("=" * 62)
    print("  七星V3 · 成交时点对比 (10万本金, 数据截至最新)")
    print("=" * 62)

    sd = rq.run_qixing_v3_same_day(data, INITIAL)
    r1 = rq.run_qixing_v3_no_lookahead(data, INITIAL)

    print_yearly("A. 同日收盘成交 (对齐实盘 14:50 执行)", sd)
    print_yearly("B. T+1开盘成交 (R1 保守基线)", r1)

    print(f"\n  【A模式近期决策 (最近6个调仓日)】")
    for d in sd["decision_log"][-6:]:
        print(f"    {d['date']} → {d['target_name']} (候选{d['n_candidates']}个)")

    eq = sd["equity_curve"]
    print(f"\n  A模式当前持仓: {rq.ETF_POOL.get(eq['holding'].iloc[-1], '货币基金')} "
          f"| 净值 {eq['equity'].iloc[-1]:,.0f} ({eq['trade_date'].iloc[-1].date()})")


if __name__ == "__main__":
    main()
