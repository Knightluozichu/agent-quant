"""趋势过滤逃离 · 决定性测试: 只在'前期大涨后大跌'(真跌高概率)时逃离, 能否跑赢V3?

把'真假跌识别'的发现落地: 趋势位置(前期涨幅)是最强判据。
测试'精准逃离'(只逃真跌)能否转化为真实收益优势。
用法: uv run python scripts/exp_trend_escape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from exp_escape_signal import price_on
from strategy_lab.engine import WARMUP, build_idx_map, get_common_dates

FEE = 0.0005
SLIPPAGE = 0.001


def escape_trigger(
    data: dict, code: str, td, drop_thr: float, ret60_thr: float, vol_filter: str | None
) -> bool:
    df = data[code]
    idx = int((df["trade_date"] <= td).sum())
    if idx < 61:
        return False
    close = df["close"].values[:idx].astype(float)
    day_ret = (close[-1] - close[-2]) / close[-2]
    if day_ret > drop_thr:
        return False  # 不是大跌
    ret60 = (close[-1] - close[-61]) / close[-61]
    if ret60 < ret60_thr:
        return False  # 前期没大涨 → 可能是假摔, 不逃
    if vol_filter == "缩量":
        vol = df["volume"].values[:idx].astype(float)
        m = vol[-21:-1].mean()
        vr = vol[-1] / m if m > 0 else 1.0
        if vr > 0.8:
            return False  # 不是缩量, 不逃
    return True


def backtest_trend_escape(
    data: dict,
    drop_thr: float,
    ret60_thr: float,
    vol_filter: str | None = None,
    initial: float = 100000,
    rebalance_days: int = 5,
    horizon: int = 5,
) -> dict:
    dates = get_common_dates(data)[WARMUP:]
    rebalance_set = set(dates[::rebalance_days])
    date_idx = {d: i for i, d in enumerate(dates)}
    cash = initial
    holding = None
    shares = 0
    eq = []
    ntr = 0
    n_esc = 0
    fwd_after = []
    for td in dates:
        if td in rebalance_set:
            idx_map = build_idx_map(data, td)
            target, _, _, _ = rq.select_target(data, idx_map, holding)
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
                        sh = int(cash * 0.99 / p / 100) * 100
                        if sh > 0:
                            cash -= sh * p * (1 + FEE + SLIPPAGE)
                            holding = target
                            shares = sh
                            ntr += 1
        if (
            holding
            and holding in data
            and holding != rq.DEFENSE
            and td not in rebalance_set
            and escape_trigger(data, holding, td, drop_thr, ret60_thr, vol_filter)
        ):
            p = price_on(data, holding, td)
            if p:
                i = date_idx[td]
                if i + horizon < len(dates):
                    pf = price_on(data, holding, dates[i + horizon])
                    if pf:
                        fwd_after.append((pf - p) / p)
                cash += shares * p * (1 - FEE - SLIPPAGE)
                ntr += 1
                n_esc += 1
                holding = None
                shares = 0
        equity = cash
        if holding and holding in data:
            p = price_on(data, holding, td)
            if p:
                equity += shares * p
        eq.append({"trade_date": pd.Timestamp(td), "equity": equity})

    eqdf = pd.DataFrame(eq)
    tr = eqdf["equity"].iloc[-1] / initial - 1
    dr = eqdf["equity"].pct_change().dropna()
    av = dr.std() * np.sqrt(252) if len(dr) > 1 else 0
    span = (eqdf["trade_date"].iloc[-1] - eqdf["trade_date"].iloc[0]).days / 365.25
    ann = (1 + tr) ** (1 / max(span, 1e-9)) - 1 if tr > -1 else -1
    sharpe = ann / av if av > 0 else 0
    cm = eqdf["equity"].cummax()
    mdd = float(((eqdf["equity"] - cm) / cm).min())
    fwd = np.array(fwd_after) if fwd_after else np.array([])
    acc = float((fwd < 0).mean() * 100) if len(fwd) else 0.0
    return {
        "final": (1 + tr) * 10,
        "ann": ann * 100,
        "sharpe": sharpe,
        "mdd": mdd * 100,
        "n_esc": n_esc,
        "acc": acc,
    }


def main() -> None:
    data = rq.load_data()
    # (名称, 是否逃离, 单日跌幅阈值, 前期涨幅阈值, 缩量过滤)
    configs = [
        ("V3基线(无逃离)", False, -0.05, 999, None),
        ("纯单日大跌-5%", True, -0.05, -999, None),
        ("趋势逃离(涨>20%后跌)", True, -0.05, 0.20, None),
        ("趋势逃离(涨>30%后跌)", True, -0.05, 0.30, None),
        ("趋势逃离(涨>40%后跌)", True, -0.05, 0.40, None),
        ("趋势逃离+缩量", True, -0.05, 0.30, "缩量"),
    ]
    print("=" * 80)
    print("  趋势过滤逃离 · 决定性测试 | 全周期真实收益 + 逃离准确率(逃对=触发后继续跌)")
    print("=" * 80)
    print(f"  {'策略':<22}{'10万→':>10}{'年化':>9}{'夏普':>8}{'回撤':>9}{'逃离次':>8}{'逃对率':>9}")
    print("  " + "-" * 72)
    for name, on, dt, rt, vf in configs:
        if not on:
            r = backtest_trend_escape(data, -0.05, 999, None)  # 永不触发
        else:
            r = backtest_trend_escape(data, dt, rt, vf)
        final_s = f"{r['final']:.1f}万"
        acc_s = f"{r['acc']:.0f}%" if r["n_esc"] else "-"
        print(
            f"  {name:<22}{final_s:>10}{r['ann']:>+8.1f}%"
            f"{r['sharpe']:>8.2f}{r['mdd']:>8.1f}%{r['n_esc']:>8}{acc_s:>9}"
        )
    print("=" * 80)
    print("  判读: 若'趋势逃离'既提高逃对率(>55%)又提升全周期收益/夏普 → 真有价值;")
    print("        若逃对率高但收益没提升 → 理论有用但实战不赚钱(手续费/踏空抵消)。")


if __name__ == "__main__":
    main()
