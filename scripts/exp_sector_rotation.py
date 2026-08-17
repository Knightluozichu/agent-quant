"""行业轮动 vs V3跨资产轮动: 同期对比, 看激进流派的真实收益与回撤.

行业池=12个行业ETF, 机制与V3完全相同(10/20动量/周频/Top1/防御/换仓缓冲),
唯一区别是资产池。同期对比, 隔离"池子"这一个变量的影响。
用法: uv run python scripts/exp_sector_rotation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from exp_sector_cycles import SECTORS
from exp_sector_cycles import load_all as load_sectors
from strategy_lab.engine import backtest
from strategy_lab.strategies import v3_select

FEE = 0.0005
SLIPPAGE = 0.001
SECTOR_CODES = [c for c in SECTORS if c not in ("510300", "159915")]  # 12行业, 排除宽基


def mom_score(close: np.ndarray, periods=(10, 20), weights=(0.5, 0.5)) -> float:
    s = 0.0
    for p, w in zip(periods, weights):
        if len(close) > p:
            s += (close[-1] - close[-p - 1]) / close[-p - 1] * w
    return s


def sector_rotation(
    data: dict, codes: list, start_date, end_date, initial: float = 100000, rebalance_days: int = 5
) -> dict:
    common = None
    for c in codes:
        if c not in data:
            continue
        d = data[c]["trade_date"].tolist()
        common = set(d) if common is None else (common & set(d))
    common = sorted(common)
    dates = common[130:]
    dates = [
        d
        for d in dates
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
    ]
    rebalance = dates[::rebalance_days]

    cash = initial
    holding = None
    shares = 0
    eq = []
    ntr = 0
    for td in rebalance:
        scores = {}
        for c in codes:
            df = data[c]
            n = int((df["trade_date"] <= td).sum())
            if n < 121:
                continue
            close = df["close"].values[:n].astype(float)
            s = mom_score(close)
            if s > 0:
                scores[c] = s
        target = max(scores, key=scores.get) if scores else None
        # 换仓缓冲 (与V3一致)
        if holding and holding in scores:
            best_s = max(scores.values()) if scores else 0
            thr = 0.0 if best_s > 0.10 else 0.05
            if scores[holding] > 0 and best_s <= scores[holding] + thr:
                target = holding
        if target != holding:
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    cash += shares * float(row.iloc[0]["close"]) * (1 - FEE - SLIPPAGE)
                    ntr += 1
                    holding = None
                    shares = 0
            if target and target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = float(row.iloc[0]["close"])
                    sh = int(cash * 0.99 / price / 100) * 100
                    if sh > 0:
                        cash -= sh * price * (1 + FEE + SLIPPAGE)
                        holding = target
                        shares = sh
                        ntr += 1
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += shares * float(row.iloc[0]["close"])
        eq.append({"trade_date": pd.Timestamp(td), "equity": equity})

    eqdf = pd.DataFrame(eq)
    tr = eqdf["equity"].iloc[-1] / initial - 1
    dr = eqdf["equity"].pct_change().dropna()
    av = dr.std() * np.sqrt(252 / rebalance_days) if len(dr) > 1 else 0
    span = (eqdf["trade_date"].iloc[-1] - eqdf["trade_date"].iloc[0]).days / 365.25
    ann = (1 + tr) ** (1 / max(span, 1e-9)) - 1 if tr > -1 else -1
    sharpe = ann / av if av > 0 else 0
    cm = eqdf["equity"].cummax()
    mdd = float(((eqdf["equity"] - cm) / cm).min())
    eqdf["year"] = eqdf["trade_date"].dt.year
    yearly: dict = {}
    prev = initial
    for y, g in eqdf.groupby("year"):
        ev = g["equity"].iloc[-1]
        yearly[int(y)] = {"return": float(ev / prev - 1)}
        prev = ev
    return {
        "total_return": tr,
        "ann_return": ann,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_trades": ntr,
        "yearly": yearly,
        "final": float(eqdf["equity"].iloc[-1]),
    }


def main() -> None:
    sectors = load_sectors()
    codes = [c for c in SECTOR_CODES if c in sectors]
    common = None
    for c in codes:
        d = sectors[c]["trade_date"].tolist()
        common = set(d) if common is None else (common & set(d))
    common = sorted(common)
    start = common[130]
    end = pd.Timestamp("2026-07-21")

    print("=" * 66)
    print(f"  行业轮动 vs V3跨资产轮动 | 同期 {start.date()} ~ 2026-07 | 10万")
    print(f"  行业池: {len(codes)}个行业ETF (机制与V3完全相同, 仅池子不同)")
    print("=" * 66)

    sr = sector_rotation(sectors, codes, start, end)
    v3data = rq.load_data()
    v3 = backtest(
        v3data,
        v3_select,
        {"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5},
        5,
        start_date=start.date(),
        end_date=end.date(),
    )

    sr_final = sr["final"] / 10000
    v3_final = v3["final_equity"] / 10000
    print(f"\n  {'指标':<10}{'行业轮动':>14}{'V3跨资产':>14}")
    print("  " + "-" * 40)
    print(f"  {'10万→':<10}{f'{sr_final:.1f}万':>14}{f'{v3_final:.1f}万':>14}")
    print(
        f"  {'累计收益':<10}{sr['total_return'] * 100:>+13.0f}%{v3['total_return'] * 100:>+13.0f}%"
    )
    print(f"  {'年化':<10}{sr['ann_return'] * 100:>+13.1f}%{v3['ann_return'] * 100:>+13.1f}%")
    print(f"  {'夏普':<10}{sr['sharpe']:>14.2f}{v3['sharpe']:>14.2f}")
    print(f"  {'最大回撤':<10}{sr['max_drawdown'] * 100:>13.1f}%{v3['max_drawdown'] * 100:>13.1f}%")
    print(f"  {'交易次数':<10}{sr['n_trades']:>14}{v3['n_trades']:>14}")

    print("\n  逐年收益对比:")
    years = sorted(set(sr["yearly"]) | set(v3["yearly"]))
    for y in years:
        s = sr["yearly"].get(y, {}).get("return")
        vv = v3["yearly"].get(y, {}).get("return")
        ss = f"{s * 100:+.0f}%" if s is not None else "-"
        vvs = f"{vv * 100:+.0f}%" if vv is not None else "-"
        print(f"    {y}: 行业 {ss:>7}   V3 {vvs:>7}")
    print("=" * 66)


if __name__ == "__main__":
    main()
