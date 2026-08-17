"""滚动OOS + 修复基线: 严格验证因子是否真能增强V3.

修复基线: weight=0 精确复现真实V3(缓冲用原始动量阈值0.05, 与rq.select_target一致)。
滚动OOS: 逐年(2020-2026)独立回测, 统计因子跑赢基线的年份数(一致性检验)。
判据: 因子须在多数年份(>=4/7)跑赢基线才算稳健alpha, 否则是某段行情的过拟合。
用法: uv run python scripts/exp_rolling_oos.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from exp_factor_zoo import (
    f_breakout,
    f_low_vol,
    f_mom_short,
    f_risk_adj,
    f_strength,
    f_vol_trend,
)
from strategy_lab.engine import backtest

WEIGHT = 0.5


def make_selector_v2(factor_fn, weight):
    """因子增强选股器(修复版): 缓冲用原始动量阈值, weight=0时精确复现V3."""

    def select(data, idx_map, holding, params):
        a_share_weak = (
            rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
            if rq.USE_A_SHARE_FILTER
            else False
        )
        cands = []
        for code in rq.ETF_POOL:
            if code not in idx_map:
                continue
            if code == rq.A_SHARE_ETF and a_share_weak:
                continue
            idx = idx_map[code]
            df = data[code]
            close = df["close"].values[: idx + 1].astype(float)
            volume = df["volume"].values[: idx + 1].astype(float)
            if len(close) < 121:
                continue
            if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
                continue
            mom = rq.calc_momentum_score(close)
            if mom <= 0:
                continue
            f = factor_fn(close, volume) if factor_fn else 0.0
            cands.append({"code": code, "mom": mom, "f": f})
        if not cands:
            return rq.DEFENSE

        def z(a):
            a = np.array(a, dtype=float)
            s = a.std()
            return (a - a.mean()) / s if s > 0 else a - a.mean()

        mz = z([c["mom"] for c in cands])
        fz = z([c["f"] for c in cands])
        for i, c in enumerate(cands):
            c["score"] = mz[i] + weight * fz[i]
        best = max(cands, key=lambda c: c["score"])
        # 缓冲用原始动量阈值(与真实V3一致)
        thr = 0.0 if best["mom"] > 0.10 else 0.05
        if holding and holding != rq.DEFENSE:
            cur = next((c for c in cands if c["code"] == holding), None)
            if cur and cur["mom"] > 0:
                return best["code"] if best["mom"] > cur["mom"] + thr else holding
            return best["code"]
        return best["code"]

    return select


def main() -> None:
    data = rq.load_data()
    factors = [
        ("V3基线", make_selector_v2(None, 0.0)),
        ("+短期动量", make_selector_v2(f_mom_short, WEIGHT)),
        ("+成交量趋势", make_selector_v2(f_vol_trend, WEIGHT)),
        ("+突破新高", make_selector_v2(f_breakout, WEIGHT)),
        ("+风险调整", make_selector_v2(f_risk_adj, WEIGHT)),
        ("+低波动", make_selector_v2(f_low_vol, WEIGHT)),
        ("+趋势强度", make_selector_v2(f_strength, WEIGHT)),
    ]
    years = list(range(2020, 2027))

    results = {}
    for name, sel in factors:
        yearly = {}
        for y in years:
            r = backtest(data, sel, {}, 5, start_date=date(y, 1, 1), end_date=date(y, 12, 31))
            yearly[y] = r["total_return"] * 100
        rf = backtest(data, sel, {}, 5)
        results[name] = {
            "yearly": yearly,
            "full": rf["total_return"] * 100,
            "sharpe": rf["sharpe"],
            "mdd": rf["max_drawdown"] * 100,
        }

    base = results["V3基线"]["yearly"]
    print("=" * 92)
    print("  滚动OOS + 修复基线 | 逐年独立回测(每年从10万 fresh 起步) | 因子 vs 真实V3")
    print("=" * 92)
    hdr = (
        f"  {'因子':<13}"
        + "".join(f"{y:>8}" for y in years)
        + f"{'全周期':>9}{'夏普':>7}{'跑赢':>6}"
    )
    print(hdr)
    print("  " + "-" * 88)
    for name, _ in factors:
        row = results[name]
        cells = "".join(f"{row['yearly'][y]:>+7.0f}%" for y in years)
        if name == "V3基线":
            wins = "-"
        else:
            wins = sum(1 for y in years if row["yearly"][y] > base[y])
            wins = f"{wins}/7"
        print(f"  {name:<13}{cells}{row['full']:>+8.0f}%{row['sharpe']:>7.2f}{wins:>6}")
    print("=" * 92)
    print("  判读: '跑赢'=因子在该年收益超过V3基线。须>=4/7年跑赢才算稳健alpha;")
    print("        若只在1-2年(尤其2025牛市)跑赢 = 过拟合那段行情, 不可用。")


if __name__ == "__main__":
    main()
