"""归因分析: 解剖某一年策略为什么赚得少/亏钱.

用法: uv run python scripts/exp_attribution.py [年份]   (默认2023)
输出: 该年调仓轨迹 + 持仓切换 + 月度收益 + 最痛的几笔, 定位亏损根源。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import backtest  # noqa: E402
from strategy_lab.strategies import v3_select  # noqa: E402

PARAMS = {"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5}
NAMES = {**rq.ETF_POOL, rq.DEFENSE: "货币基金"}


def analyze(year: int) -> None:
    data = rq.load_data()
    res = backtest(data, v3_select, PARAMS, 5)
    eq = res["equity_curve"].copy()
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    y = eq[eq["trade_date"].dt.year == year].copy()
    if y.empty:
        print(f"  {year}年无数据")
        return
    y["ret"] = y["equity"].pct_change() * 100

    print("=" * 64)
    print(f"  V3 归因分析 · {year}年 (全年收益 {res['yearly'][year]['return']*100:+.1f}%)")
    print("=" * 64)

    # 1. 持仓切换轨迹
    print(f"\n  【{year}年 换仓轨迹】")
    prev_hold = None
    switches = []
    for _, r in y.iterrows():
        if r["holding"] != prev_hold:
            switches.append((r["trade_date"], prev_hold, r["holding"]))
            prev_hold = r["holding"]
    for d, frm, to in switches:
        f = NAMES.get(frm, "空仓") if frm else "空仓"
        print(f"    {d.date()}: {f} → {NAMES.get(to, to)}")
    print(f"    全年共换仓 {len(switches)} 次")

    # 2. 月度收益
    y["month"] = y["trade_date"].dt.month
    monthly = y.groupby("month")["equity"].agg(["first", "last"])
    monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100
    print(f"\n  【{year}年 月度收益】")
    for m, row in monthly.iterrows():
        bar = "█" * int(abs(row["ret"]) / 2)
        sign = "🔴" if row["ret"] < 0 else "🟢"
        print(f"    {m:>2}月 {row['ret']:+6.1f}% {sign}{bar}")

    # 3. 最痛的调仓段 (单期跌幅最大)
    worst = y.dropna(subset=["ret"]).nsmallest(3, "ret")
    print(f"\n  【{year}年 最痛的3个调仓段】")
    for _, r in worst.iterrows():
        print(f"    {r['trade_date'].date()} 持{NAMES.get(r['holding'], r['holding'])} "
              f"单期 {r['ret']:+.1f}%")

    # 4. 各资产持有贡献
    print(f"\n  【{year}年 各资产持有期贡献】")
    contrib = y.groupby("holding")["ret"].sum().sort_values()
    for code, c in contrib.items():
        print(f"    {NAMES.get(code, code):8s}: {c:+.1f}%")
    print("=" * 64)


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    analyze(yr)
