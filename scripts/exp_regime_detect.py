"""周期识别分析: 分辨 上涨/震荡/下跌 周期, 验证 V3 在不同周期的表现差异.

核心前提检验: 动量轮动是否"上涨期赚钱、震荡期吃亏"? 这个关系样本外是否稳定?
周期指标(因果, 无前瞻): 等权池60日趋势 + 动量离散度。
用法: uv run python scripts/exp_regime_detect.py
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
IS_END = pd.Timestamp("2024-01-01")
TREND_WIN = 60  # 周期趋势窗口(日)
TREND_THR = 0.05  # 涨跌周期阈值(±5%)


def pool_trend_and_dispersion(data: dict, idx_map: dict) -> tuple[float, float] | None:
    """等权池60日趋势 + 动量离散度 (因果指标)."""
    rets, scores = [], []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < TREND_WIN + 1:
            continue
        rets.append((close[-1] - close[-TREND_WIN - 1]) / close[-TREND_WIN - 1])
        if len(close) >= 121:
            scores.append(rq.calc_momentum_score(close))
    if len(rets) < 3:
        return None
    return float(np.mean(rets)), float(np.std(scores)) if len(scores) > 1 else 0.0


def classify_regime(trend: float, disp: float) -> str:
    if trend > TREND_THR:
        return "上涨"
    if trend < -TREND_THR:
        return "下跌"
    return "震荡"


def build_panel(data: dict) -> pd.DataFrame:
    res = backtest(data, v3_select, PARAMS, 5)
    eq = res["equity_curve"].reset_index(drop=True)
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    rows = []
    for i in range(len(eq) - 1):
        td = eq["trade_date"].iloc[i]
        idx_map = build_idx_map(data, td.date())
        td_result = pool_trend_and_dispersion(data, idx_map)
        if td_result is None:
            continue
        trend, disp = td_result
        fwd_ret = eq["equity"].iloc[i + 1] / eq["equity"].iloc[i] - 1
        rows.append(
            {
                "date": td,
                "trend": trend,
                "dispersion": disp,
                "regime": classify_regime(trend, disp),
                "fwd_ret": fwd_ret,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_regime(df: pd.DataFrame, label: str) -> None:
    print(f"\n  【{label}】 V3在不同周期的表现:")
    print(f"  {'周期':<6}{'期数':>6}{'平均下期收益':>14}{'胜率':>10}{'累计':>12}")
    print("  " + "-" * 50)
    for reg in ["上涨", "震荡", "下跌"]:
        sub = df[df["regime"] == reg]
        if sub.empty:
            print(f"  {reg:<6}{'0':>6}{'-':>14}{'-':>10}{'-':>12}")
            continue
        avg = sub["fwd_ret"].mean() * 100
        winrate = (sub["fwd_ret"] > 0).mean() * 100
        cum = ((1 + sub["fwd_ret"]).prod() - 1) * 100
        print(f"  {reg:<6}{len(sub):>6}{avg:>+13.2f}%{winrate:>9.0f}%{cum:>+11.1f}%")


def main() -> None:
    data = rq.load_data()
    panel = build_panel(data)
    is_df = panel[panel["date"] < IS_END]
    oos_df = panel[panel["date"] >= IS_END]

    print("=" * 64)
    print(f"  周期识别分析 | 趋势窗口{TREND_WIN}日 阈值±{TREND_THR * 100:.0f}%")
    print("=" * 64)
    print(
        f"\n  周期分布(全段): "
        + "  ".join(f"{r}:{(panel['regime'] == r).sum()}期" for r in ["上涨", "震荡", "下跌"])
    )

    summarize_by_regime(is_df, "样本内 2020-2024")
    summarize_by_regime(oos_df, "样本外 2024-2026")

    # 核心前提检验: 上涨期收益 > 震荡期收益, 且样本内外一致?
    def regime_avg(df, reg):
        s = df[df["regime"] == reg]["fwd_ret"]
        return s.mean() * 100 if not s.empty else float("nan")

    is_up, is_side = regime_avg(is_df, "上涨"), regime_avg(is_df, "震荡")
    oos_up, oos_side = regime_avg(oos_df, "上涨"), regime_avg(oos_df, "震荡")
    print("\n  【核心前提检验】 上涨期 vs 震荡期 平均收益:")
    print(
        f"    样本内: 上涨 {is_up:+.2f}% vs 震荡 {is_side:+.2f}%  "
        f"({'上涨>震荡 ✓' if is_up > is_side else '上涨<震荡 ✗'})"
    )
    print(
        f"    样本外: 上涨 {oos_up:+.2f}% vs 震荡 {oos_side:+.2f}%  "
        f"({'上涨>震荡 ✓' if oos_up > oos_side else '上涨<震荡 ✗'})"
    )
    stable = (is_up > is_side) == (oos_up > oos_side)
    print(
        f"\n  【结论】 周期-表现关系样本内外{'一致 ✓ (前提成立, 可继续)' if stable else '不一致 ✗ (前提存疑, 警惕过拟合)'}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
