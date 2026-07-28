"""自适应波动率管理版V3: 三种回测对比 (10年 / 样本内外 / 3年滚动).

波动率管理: 仓位 = min((目标波动率/近期20日已实现波动率)², 100%) × 满仓
近期波动率=20日已实现波动率(年化)。高波动减仓, 低波动满仓(不加杠杆)。
用法: uv run python scripts/exp_vol_managed.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import WARMUP, build_idx_map, get_common_dates  # noqa: E402

FEE = rq.FEE
SLIPPAGE = rq.SLIPPAGE
ORIG_POOL = dict(rq.ETF_POOL)
ORIG_DROP = rq.USE_DROP_FILTER


def price_on(data, code, td):
    if code not in data:
        return None
    row = data[code][data[code]["trade_date"] == td]
    return float(row.iloc[0]["close"]) if not row.empty else None


def backtest_vol(data, target_vol, vol_lookback=20, rebalance_days=5,
                 start_date=None, end_date=None, initial=100000):
    """波动率管理版V3回测. target_vol=None 即纯V3(满仓)."""
    dates = get_common_dates(data)[WARMUP:]
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    rebalance = dates[::rebalance_days]
    cash = initial
    holding = None
    shares = 0
    eq = []
    ntr = 0
    for td in rebalance:
        idx_map = build_idx_map(data, td)
        target, _, _, _ = rq.select_target(data, idx_map, holding)
        frac = 1.0
        if target_vol is not None and target and target != rq.DEFENSE and target in data:
            idx = idx_map.get(target)
            if idx is not None and idx >= vol_lookback:
                close = data[target]["close"].values[: idx + 1].astype(float)
                rets = np.diff(close[-vol_lookback - 1:]) / close[-vol_lookback - 1:-1]
                rvol = np.std(rets) * np.sqrt(252)
                if rvol > 0:
                    frac = min((target_vol / rvol) ** 2, 1.0)
        if target != holding:
            if holding and holding in data:
                p = price_on(data, holding, td)
                if p:
                    cash += shares * p * (1 - FEE - SLIPPAGE)
                    ntr += 1
                    holding = None
                    shares = 0
            if target and target in data and target != rq.DEFENSE:
                p = price_on(data, target, td)
                if p:
                    invest = cash * frac * 0.99
                    sh = int(invest / p / 100) * 100
                    if sh > 0:
                        cash -= sh * p * (1 + FEE + SLIPPAGE)
                        holding = target
                        shares = sh
                        ntr += 1
        equity = cash
        if holding and holding in data:
            p = price_on(data, holding, td)
            if p:
                equity += shares * p
        eq.append({"trade_date": pd.Timestamp(td), "equity": equity})
    eqdf = pd.DataFrame(eq)
    if eqdf.empty:
        return {"total": 0, "ann": 0, "sharpe": 0, "mdd": 0, "ntr": 0, "final": initial}
    tr = eqdf["equity"].iloc[-1] / initial - 1
    dr = eqdf["equity"].pct_change().dropna()
    av = dr.std() * np.sqrt(252 / rebalance_days) if len(dr) > 1 else 0
    span = (eqdf["trade_date"].iloc[-1] - eqdf["trade_date"].iloc[0]).days / 365.25
    ann = (1 + tr) ** (1 / max(span, 1e-9)) - 1 if tr > -1 else -1
    sharpe = ann / av if av > 0 else 0
    cm = eqdf["equity"].cummax()
    mdd = float(((eqdf["equity"] - cm) / cm).min())
    return {"total": tr * 100, "ann": ann * 100, "sharpe": sharpe,
            "mdd": mdd * 100, "ntr": ntr, "final": float(eqdf["equity"].iloc[-1])}


def set_pool(codes):
    rq.ETF_POOL.clear()
    rq.ETF_POOL.update({c: ORIG_POOL[c] for c in codes if c in ORIG_POOL})


def restore():
    rq.ETF_POOL.clear()
    rq.ETF_POOL.update(ORIG_POOL)
    rq.USE_DROP_FILTER = ORIG_DROP


# (名称, 目标波动率)  None=纯V3满仓
CONFIGS = [("V3基线(满仓)", None), ("波动率管理15%", 0.15), ("波动率管理20%", 0.20)]


def main() -> None:
    data = rq.load_data()
    rq.USE_DROP_FILTER = True

    # === 测试1: 10年回测 (去豆粕, 7资产, ~2016-2026) ===
    set_pool([c for c in ORIG_POOL if c != "159985"])
    print("=" * 72)
    print("  【测试1】10年回测 (~2016-2026, 7资产去豆粕)")
    print("=" * 72)
    print(f"  {'配置':<18}{'10万→':>12}{'累计':>10}{'年化':>9}{'夏普':>7}{'回撤':>8}{'交易':>7}")
    print("  " + "-" * 66)
    for name, tv in CONFIGS:
        r = backtest_vol(data, tv, start_date=date(2016, 1, 1))
        final_s = f"{r['final']/10000:.1f}万"
        print(f"  {name:<18}{final_s:>12}{r['total']:>+9.0f}%{r['ann']:>+8.1f}%"
              f"{r['sharpe']:>7.2f}{r['mdd']:>+7.1f}%{r['ntr']:>7}")

    # === 测试2: 样本内/样本外 (全8资产) ===
    restore()
    rq.USE_DROP_FILTER = True
    print("\n" + "=" * 72)
    print("  【测试2】样本内(2020-2023) / 样本外(2024-2026), 全8资产")
    print("=" * 72)
    print(f"  {'配置':<18}{'IS年化':>9}{'IS夏普':>8}{'IS回撤':>8}{'OOS年化':>10}{'OOS夏普':>9}{'OOS回撤':>9}")
    print("  " + "-" * 66)
    for name, tv in CONFIGS:
        is_r = backtest_vol(data, tv, start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
        oos_r = backtest_vol(data, tv, start_date=date(2024, 1, 1))
        print(f"  {name:<18}{is_r['ann']:>+8.1f}%{is_r['sharpe']:>8.2f}{is_r['mdd']:>+7.1f}%"
              f"{oos_r['ann']:>+9.1f}%{oos_r['sharpe']:>9.2f}{oos_r['mdd']:>+8.1f}%")

    # === 测试3: 3年滚动窗口 (全8资产) ===
    restore()
    rq.USE_DROP_FILTER = True
    windows = [(2020, 2022), (2021, 2023), (2022, 2024), (2023, 2025), (2024, 2026)]
    print("\n" + "=" * 72)
    print("  【测试3】3年滚动窗口 (全8资产), 各配置年化收益对比")
    print("=" * 72)
    hdr = f"  {'配置':<18}" + "".join(f"{f'{a}-{b}':>12}" for a, b in windows) + f"{'胜V3':>8}"
    print(hdr)
    print("  " + "-" * 78)
    base_results = {}
    for name, tv in CONFIGS:
        cells = []
        for a, b in windows:
            r = backtest_vol(data, tv, start_date=date(a, 1, 1), end_date=date(b, 12, 31))
            cells.append(r["ann"])
        if name == CONFIGS[0][0]:
            base_results = dict(zip(windows, cells))
            winstr = "-"
        else:
            wins = sum(1 for w, c in zip(windows, cells) if c > base_results[w])
            winstr = f"{wins}/{len(windows)}"
        cellstr = "".join(f"{c:>+11.1f}%" for c in cells)
        print(f"  {name:<18}{cellstr}{winstr:>8}")
    print("=" * 72)
    print("  判读: 波动率管理若真有效, 应在降回撤的同时不显著损失收益(夏普更高)。")
    restore()


if __name__ == "__main__":
    main()
