"""逃离信号 · 全周期真实收益 + 跌势敏感性分析.

回应用户: 不看IS/OOS切分, 直接看全周期复权回测的真实总收益;
并分析各逃离信号触发后是继续跌(逃对)还是反弹(逃错)——即对跌势的敏感性。
用法: uv run python scripts/exp_escape_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from exp_escape_signal import backtest_escape, check_escape, price_on  # noqa: E402
from strategy_lab.engine import WARMUP, build_idx_map, get_common_dates  # noqa: E402

POOL_FULL = list(rq.ETF_POOL.keys())            # 8个(含豆粕, ~6.5年)
POOL_10Y = [c for c in POOL_FULL if c != "159985"]  # 去豆粕, ~10年

CONFIGS = [
    ("V3基线(无逃离)", None, None),
    ("回撤逃离-5%", "drawdown", 0.05),
    ("回撤逃离-8%", "drawdown", 0.08),
    ("单日大跌-5%", "singleday", 0.05),
    ("跌破20日线", "ma", 20),
]


def run_with_pool(pool, data):
    """临时切换ETF池跑回测, 返回各配置的全周期指标."""
    orig_pool = dict(rq.ETF_POOL)
    rq.ETF_POOL.clear()
    rq.ETF_POOL.update({c: orig_pool[c] for c in pool if c in orig_pool})
    try:
        rows = []
        for name, etype, param in CONFIGS:
            r = backtest_escape(data, etype, param)  # 全周期, 不切分
            rows.append((name, r))
        return rows
    finally:
        rq.ETF_POOL.clear()
        rq.ETF_POOL.update(orig_pool)


def trigger_analysis(data, etype, param, horizon=5) -> dict:
    """分析逃离信号触发后的走向: 继续跌(逃对) vs 反弹(逃错)."""
    dates = get_common_dates(data)[WARMUP:]
    rebalance_set = set(dates[::5])
    # 模拟V3持仓轨迹, 记录每个持有日的逃离信号触发 + 未来horizon日收益
    cash = 100000.0
    holding = None
    shares = 0
    peak = 0.0
    fwd_after_trigger = []  # 触发后horizon日收益
    date_idx = {d: i for i, d in enumerate(dates)}

    for td in dates:
        if td in rebalance_set:
            idx_map = build_idx_map(data, td)
            target, _, _, _ = rq.select_target(data, idx_map, holding)
            if target != holding:
                if holding and holding in data:
                    p = price_on(data, holding, td)
                    if p:
                        cash += shares * p
                        holding = None
                        shares = 0
                        peak = 0.0
                if target and target in data and target != rq.DEFENSE:
                    p = price_on(data, target, td)
                    if p:
                        sh = int(cash * 0.99 / p / 100) * 100
                        if sh > 0:
                            cash -= sh * p
                            holding = target
                            shares = sh
                            peak = p
        if holding and holding in data and holding != rq.DEFENSE and td not in rebalance_set:
            p = price_on(data, holding, td)
            if p:
                peak = max(peak, p)
                if check_escape(etype, param, data, holding, td, p, peak):
                    # 记录触发后horizon个交易日的收益
                    i = date_idx[td]
                    if i + horizon < len(dates):
                        p_future = price_on(data, holding, dates[i + horizon])
                        if p_future:
                            fwd_after_trigger.append((p_future - p) / p)
                    # 触发后离场
                    cash += shares * p
                    holding = None
                    shares = 0
                    peak = 0.0

    if not fwd_after_trigger:
        return {"n": 0}
    arr = np.array(fwd_after_trigger)
    return {
        "n": len(arr),
        "pct_continued_drop": float((arr < 0).mean() * 100),  # 继续跌=逃对
        "pct_rebounded": float((arr > 0).mean() * 100),        # 反弹=逃错
        "avg_fwd": float(arr.mean() * 100),
    }


def main() -> None:
    data = rq.load_data()

    for label, pool in [("全池8资产 ~6.5年(含豆粕)", POOL_FULL),
                        ("10年池 ~10年(去豆粕)", POOL_10Y)]:
        rows = run_with_pool(pool, data)
        span = ""
        print("=" * 70)
        print(f"  【{label}】 全周期真实收益 (不切分IS/OOS)")
        print("=" * 70)
        print(f"  {'策略':<16}{'10万→':>12}{'累计':>10}{'年化':>9}{'夏普':>8}{'回撤':>9}")
        print("  " + "-" * 60)
        for name, r in rows:
            final = (1 + r["total_return"]) * 10
            print(f"  {name:<16}{f'{final:.1f}万':>12}{r['total_return']*100:>+9.0f}%"
                  f"{r['ann_return']*100:>+8.1f}%{r['sharpe']:>8.2f}"
                  f"{r['max_drawdown']*100:>8.1f}%")
        print()

    # 跌势敏感性分析 (全池)
    print("=" * 70)
    print("  【跌势敏感性】 逃离信号触发后5日: 继续跌(逃对) vs 反弹(逃错)")
    print("=" * 70)
    print(f"  {'信号':<16}{'触发次数':>8}{'继续跌%':>10}{'反弹%':>9}{'平均后5日':>12}")
    print("  " + "-" * 56)
    for name, etype, param in CONFIGS:
        if etype is None:
            continue
        ta = trigger_analysis(data, etype, param)
        if ta["n"] == 0:
            print(f"  {name:<16}{'0':>8}")
            continue
        print(f"  {name:<16}{ta['n']:>8}{ta['pct_continued_drop']:>9.0f}%"
              f"{ta['pct_rebounded']:>8.0f}%{ta['avg_fwd']:>+11.2f}%")
    print("=" * 70)
    print("  判读: '继续跌%'高=该信号真能识别跌势; '反弹%'高=信号是假摔诱捕(逃错)。")


if __name__ == "__main__":
    main()
