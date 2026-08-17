"""因子大筛查: 系统测试多种因子及组合能否增强V3, 样本内+样本外双重检验.

方法: 在V3正动量候选中, 用 截面z-score(动量) + 权重×z-score(因子) 综合排名选Top1。
每个因子跑 IS(2020-2023) 与 OOS(2024-2026), 重点看 OOS 是否真站得住。
警示: 测N个因子, 样本内最优者多半是运气(多重检验); 只有OOS也优才是真alpha。
用法: uv run python scripts/exp_factor_zoo.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from strategy_lab.engine import backtest

IS_END = date(2024, 1, 1)
WEIGHT = 0.5  # 因子权重(z-score单位)


# --------------------------------------------------------------------------- #
# 候选因子: 输入(close, volume)数组, 输出原始因子值(后续截面标准化)
# --------------------------------------------------------------------------- #
def f_risk_adj(close, volume):  # 风险调整动量 = 动量/波动率
    if len(close) < 21:
        return 0.0
    vol = np.std(np.diff(close[-21:]) / close[-21:-1])
    return rq.calc_momentum_score(close) / vol if vol > 0 else 0.0


def f_low_vol(close, volume):  # 低波动偏好(取负)
    if len(close) < 21:
        return 0.0
    return -np.std(np.diff(close[-21:]) / close[-21:-1])


def f_vol_trend(close, volume):  # 成交量趋势(近5日vs前5日)
    if len(volume) < 11:
        return 0.0
    older = volume[-10:-5].mean()
    return (volume[-5:].mean() - older) / older if older > 0 else 0.0


def f_reversal(close, volume):  # 短期反转(近5日涨幅取负)
    if len(close) < 6:
        return 0.0
    return -(close[-1] - close[-6]) / close[-6]


def f_mom_short(close, volume):  # 短期动量(10日)
    if len(close) < 11:
        return 0.0
    return (close[-1] - close[-11]) / close[-11]


def f_mom_long(close, volume):  # 长期动量(60日)
    if len(close) < 61:
        return 0.0
    return (close[-1] - close[-61]) / close[-61]


def f_breakout(close, volume):  # 突破(接近20日高点)
    if len(close) < 21:
        return 0.0
    return close[-1] / close[-21:].max()


def f_strength(close, volume):  # 趋势强度(收盘价相对均线偏离)
    if len(close) < 21:
        return 0.0
    return close[-1] / close[-20:].mean() - 1


def make_selector(factor_fn, weight):
    """构造因子增强选股器(截面z-score综合排名)."""

    def select(data, idx_map, holding, params):
        a_share_weak = (
            rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
            if rq.USE_A_SHARE_FILTER
            else False
        )
        raw = []
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
            raw.append({"code": code, "mom": mom, "f": f})
        if not raw:
            return rq.DEFENSE

        def zscore(a):
            a = np.array(a, dtype=float)
            s = a.std()
            return (a - a.mean()) / s if s > 0 else a - a.mean()

        mz = zscore([r["mom"] for r in raw])
        fz = zscore([r["f"] for r in raw])
        for i, r in enumerate(raw):
            r["score"] = mz[i] + weight * fz[i]
        raw.sort(key=lambda r: -r["score"])
        best = raw[0]
        thr = 0.0 if best["mom"] > 0.10 else 0.05
        if holding and holding != rq.DEFENSE:
            cur = next((r for r in raw if r["code"] == holding), None)
            if cur and cur["mom"] > 0:
                return best["code"] if best["score"] > cur["score"] + thr else holding
            return best["code"]
        return best["code"]

    return select


def combo_selector(fns, weight):
    """多因子组合: 各因子z-score平均后加权."""

    def combo_fn(close, volume):
        return np.mean([fn(close, volume) for fn in fns])

    return make_selector(combo_fn, weight)


def main() -> None:
    data = rq.load_data()
    factors = [
        ("V3基线(纯动量)", make_selector(None, 0.0)),
        ("+风险调整动量", make_selector(f_risk_adj, WEIGHT)),
        ("+低波动", make_selector(f_low_vol, WEIGHT)),
        ("+成交量趋势", make_selector(f_vol_trend, WEIGHT)),
        ("+短期反转", make_selector(f_reversal, WEIGHT)),
        ("+短期动量", make_selector(f_mom_short, WEIGHT)),
        ("+长期动量", make_selector(f_mom_long, WEIGHT)),
        ("+突破新高", make_selector(f_breakout, WEIGHT)),
        ("+趋势强度", make_selector(f_strength, WEIGHT)),
        ("+组合(风调+低波)", combo_selector([f_risk_adj, f_low_vol], WEIGHT)),
        ("+组合(动量+反转)", combo_selector([f_mom_short, f_reversal], WEIGHT)),
    ]

    print("=" * 82)
    print(f"  因子大筛查 | 权重{WEIGHT} | IS=2020-2023, OOS=2024-2026 | 重点看OOS")
    print("=" * 82)
    print(
        f"  {'因子':<20}{'IS年化':>9}{'IS夏普':>8}{'IS回撤':>9}{'OOS年化':>10}{'OOS夏普':>9}{'OOS回撤':>9}{'OOS胜?':>8}"
    )
    print("  " + "-" * 76)
    base_oos_sharpe = None
    for name, sel in factors:
        is_r = backtest(data, sel, {}, 5, start_date=None, end_date=IS_END)
        oos_r = backtest(data, sel, {}, 5, start_date=IS_END, end_date=None)
        if base_oos_sharpe is None:
            base_oos_sharpe = oos_r["sharpe"]
        win = "✓" if oos_r["sharpe"] > base_oos_sharpe else "✗"
        if name.startswith("V3"):
            win = "-"
        print(
            f"  {name:<20}{is_r['ann_return'] * 100:>+8.1f}%{is_r['sharpe']:>8.2f}"
            f"{is_r['max_drawdown'] * 100:>8.1f}%{oos_r['ann_return'] * 100:>+9.1f}%"
            f"{oos_r['sharpe']:>9.2f}{oos_r['max_drawdown'] * 100:>8.1f}%{win:>8}"
        )
    print("=" * 82)
    print("  判读: 'OOS胜?'=该因子OOS夏普是否超过V3基线。测11个因子, 即使全无效,")
    print("        样本内也会有2-3个'看起来好'(运气); 只有OOS也胜的才是真alpha。")


if __name__ == "__main__":
    main()
