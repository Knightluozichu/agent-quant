"""紧急逃离信号实验: 在调仓周期内, 检测逃离信号提前离场避险.

测试3类逃离信号(回撤逃离/单日大跌/跌破均线), 对比 V3(无逃离) 基线,
样本内(2020-2023)找有效信号 → 样本外(2024-2026)验证是否过拟合。
用法: uv run python scripts/exp_escape_signal.py
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

FEE = 0.0005
SLIPPAGE = 0.001
IS_END = date(2024, 1, 1)


def price_on(data: dict, code: str, td):
    if code not in data:
        return None
    row = data[code][data[code]["trade_date"] == td]
    return float(row.iloc[0]["close"]) if not row.empty else None


def prev_close(data: dict, code: str, td):
    df = data[code]
    hist = df[df["trade_date"] < td]
    return float(hist.iloc[-1]["close"]) if not hist.empty else None


def n_day_ma(data: dict, code: str, td, n: int):
    df = data[code]
    idx = int((df["trade_date"] <= td).sum())
    if idx < n:
        return None
    return float(df["close"].values[idx - n : idx].astype(float).mean())


def check_escape(etype: str, param, data: dict, code: str, td, price, peak) -> bool:
    if etype == "drawdown":  # 从持仓期高点回撤 > param
        return peak > 0 and (price - peak) / peak < -param
    if etype == "singleday":  # 单日跌幅 > param
        pc = prev_close(data, code, td)
        return pc is not None and pc > 0 and (price - pc) / pc < -param
    if etype == "ma":  # 跌破 N 日均线
        ma = n_day_ma(data, code, td, int(param))
        return ma is not None and price < ma
    return False


def backtest_escape(
    data: dict,
    escape_type: str | None,
    escape_param,
    start_date=None,
    end_date=None,
    initial: float = 100000,
    rebalance_days: int = 5,
) -> dict:
    """V3 + 日频逃离检查. escape_type=None 即纯V3基线."""
    dates = get_common_dates(data)[WARMUP:]
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    rebalance_set = set(dates[::rebalance_days])

    cash = initial
    holding = None
    shares = 0
    peak = 0.0
    eq = []
    ntr = 0
    n_escape = 0

    for td in dates:
        # === 调仓日: V3选股 ===
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
                        peak = 0.0
                if target and target in data and target != rq.DEFENSE:
                    p = price_on(data, target, td)
                    if p:
                        sh = int(cash * 0.99 / p / 100) * 100
                        if sh > 0:
                            cash -= sh * p * (1 + FEE + SLIPPAGE)
                            holding = target
                            shares = sh
                            peak = p
                            ntr += 1

        # === 每日逃离检查 ===
        if (
            escape_type
            and holding
            and holding in data
            and holding != rq.DEFENSE
            and td not in rebalance_set
        ):
            p = price_on(data, holding, td)
            if p:
                peak = max(peak, p)
                if check_escape(escape_type, escape_param, data, holding, td, p, peak):
                    cash += shares * p * (1 - FEE - SLIPPAGE)
                    ntr += 1
                    n_escape += 1
                    holding = None
                    shares = 0
                    peak = 0.0

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
    return {
        "total_return": tr,
        "ann_return": ann,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_trades": ntr,
        "n_escape": n_escape,
    }


def main() -> None:
    data = rq.load_data()
    # 逃离信号配置: (名称, 类型, 参数)
    configs = [
        ("V3基线(无逃离)", None, None),
        ("回撤逃离-5%", "drawdown", 0.05),
        ("回撤逃离-8%", "drawdown", 0.08),
        ("回撤逃离-10%", "drawdown", 0.10),
        ("单日大跌-3%", "singleday", 0.03),
        ("单日大跌-5%", "singleday", 0.05),
        ("跌破10日线", "ma", 10),
        ("跌破20日线", "ma", 20),
    ]

    print("=" * 78)
    print("  紧急逃离信号实验 | 样本内2020-2023 → 样本外2024-2026")
    print("=" * 78)
    print(
        f"  {'策略':<16}{'IS年化':>9}{'IS夏普':>8}{'IS回撤':>9}{'OOS年化':>10}{'OOS夏普':>9}{'OOS回撤':>9}{'逃离次数':>9}"
    )
    print("  " + "-" * 74)
    for name, etype, param in configs:
        is_r = backtest_escape(data, etype, param, end_date=IS_END)
        oos_r = backtest_escape(data, etype, param, start_date=IS_END)
        print(
            f"  {name:<16}{is_r['ann_return'] * 100:>+8.1f}%{is_r['sharpe']:>8.2f}"
            f"{is_r['max_drawdown'] * 100:>8.1f}%{oos_r['ann_return'] * 100:>+9.1f}%"
            f"{oos_r['sharpe']:>9.2f}{oos_r['max_drawdown'] * 100:>8.1f}%"
            f"{oos_r['n_escape']:>9}"
        )
    print("=" * 78)
    print("  判读: 逃离信号若真有效, 应样本内外都提升夏普/降回撤; 若仅样本内好=过拟合。")


if __name__ == "__main__":
    main()
