"""因子归因 + Regime分析 + OOS验证.

范式: 样本内(2020-2024)找区分涨跌期的因子 → 形成假设 → 样本外(2024-2026)验证。
候选因子: 动量离散度/最强动量/平均动量/市场宽度/平均波动率。
用法: uv run python scripts/exp_factor_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import WARMUP, backtest, build_idx_map, get_common_dates  # noqa: E402
from strategy_lab.strategies import v3_select  # noqa: E402

PARAMS = {"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5}
IS_END = pd.Timestamp("2024-01-01")   # 样本内/外分界


def compute_factors_at(data: dict, idx_map: dict) -> dict | None:
    """在某调仓日计算 regime 因子 (只用截至当日数据, 无前瞻)."""
    scores, vols = [], []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        scores.append(rq.calc_momentum_score(close))
        if len(close) >= 21:
            rets = np.diff(close[-21:]) / close[-21:-1]
            vols.append(np.std(rets))
    scores = np.array(scores)
    if len(scores) == 0:
        return None
    return {
        "dispersion": float(np.std(scores)),
        "best_mom": float(np.max(scores)),
        "avg_mom": float(np.mean(scores)),
        "breadth": float((scores > 0).sum()),
        "avg_vol": float(np.mean(vols)) if vols else 0.0,
    }


def build_panel(data: dict) -> pd.DataFrame:
    """构建面板: 每个调仓日的因子 + 下一期策略收益."""
    res = backtest(data, v3_select, PARAMS, 5)
    eq = res["equity_curve"].reset_index(drop=True)
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    rows = []
    for i in range(len(eq) - 1):
        td = eq["trade_date"].iloc[i]
        idx_map = build_idx_map(data, td.date())
        f = compute_factors_at(data, idx_map)
        if f is None:
            continue
        fwd_ret = eq["equity"].iloc[i + 1] / eq["equity"].iloc[i] - 1
        rows.append({"date": td, **f, "fwd_ret": fwd_ret})
    return pd.DataFrame(rows)


def main() -> None:
    data = rq.load_data()
    panel = build_panel(data)
    is_df = panel[panel["date"] < IS_END]
    oos_df = panel[panel["date"] >= IS_END]
    factors = ["dispersion", "best_mom", "avg_mom", "breadth", "avg_vol"]

    print("=" * 68)
    print("  因子归因分析 | 样本内 2020-2024 (找规律) → 样本外 2024-2026 (验证)")
    print("=" * 68)

    # === 样本内: 各因子与下期收益的相关性 + 涨跌期因子均值对比 ===
    print(f"\n  【样本内 {len(is_df)} 个调仓期】 因子与下期收益相关性:")
    print(f"  {'因子':<12}{'相关系数':>10}{'赚钱期均值':>14}{'亏钱期均值':>14}")
    print("  " + "-" * 52)
    up = is_df[is_df["fwd_ret"] > 0]
    dn = is_df[is_df["fwd_ret"] <= 0]
    is_corr = {}
    for f in factors:
        c = is_df[f].corr(is_df["fwd_ret"])
        is_corr[f] = c
        print(f"  {f:<12}{c:>10.3f}{up[f].mean():>14.4f}{dn[f].mean():>14.4f}")

    # 选相关性最强的因子
    best_factor = max(is_corr, key=lambda k: abs(is_corr[k]))
    print(f"\n  → 样本内最强预测因子: {best_factor} (相关 {is_corr[best_factor]:+.3f})")

    # === 形成假设: 用该因子中位数分高低组, 看收益差异 ===
    thr = is_df[best_factor].median()
    print(f"  → 假设: {best_factor} > {thr:.4f}(高) 时做多, 否则防御")
    is_high = is_df[is_df[best_factor] > thr]
    is_low = is_df[is_df[best_factor] <= thr]
    print(f"  【样本内验证】 高组平均下期收益 {is_high['fwd_ret'].mean()*100:+.2f}% "
          f"vs 低组 {is_low['fwd_ret'].mean()*100:+.2f}%")

    # === 样本外: 该因子是否依然有效 ===
    print(f"\n  【样本外 2024-2026, {len(oos_df)} 期】 同一因子是否仍有效?")
    oos_corr = oos_df[best_factor].corr(oos_df["fwd_ret"])
    oos_high = oos_df[oos_df[best_factor] > thr]
    oos_low = oos_df[oos_df[best_factor] <= thr]
    print(f"  样本外相关系数: {oos_corr:+.3f} (样本内 {is_corr[best_factor]:+.3f})")
    print(f"  高组平均下期收益 {oos_high['fwd_ret'].mean()*100:+.2f}% "
          f"vs 低组 {oos_low['fwd_ret'].mean()*100:+.2f}%")

    # === OOS 规则模拟: 低因子值时持币防御 vs 纯V3 ===
    # 低组若持币(收益≈0), 高组正常 → 估算增强效果
    plain_oos = (1 + oos_df["fwd_ret"]).prod() - 1
    # 增强: 低组收益替换为0(持货币), 高组保留
    enh_rets = oos_df.apply(
        lambda r: r["fwd_ret"] if r[best_factor] > thr else 0.0, axis=1)
    enh_oos = (1 + enh_rets).prod() - 1
    print(f"\n  【OOS 策略对比 2024-2026】")
    print(f"    纯V3:        {plain_oos*100:+.1f}%")
    print(f"    增强(低{best_factor}持币): {enh_oos*100:+.1f}%")
    verdict = "✅ 样本外依然有效(可考虑采纳)" if (
        abs(oos_corr) > 0.1 and np.sign(oos_corr) == np.sign(is_corr[best_factor])
    ) else "⚠️ 样本外失效(疑似过拟合, 否决)"
    print(f"\n  【结论】 {verdict}")
    print("=" * 68)


if __name__ == "__main__":
    main()
