"""周期识别(动量离散度版): 用动量离散度分辨 趋势期/震荡期, 验证V3表现差异.

理论: 高分散=资产分化明显=有领涨者=动量轮动好使; 低分散=同涨同跌=震荡=动量失效。
检验: 离散度能否稳定预测V3表现 (样本内2020-2024 → 样本外2024-2026)。
用法: uv run python scripts/exp_regime_dispersion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import WARMUP, backtest, build_idx_map  # noqa: E402
from strategy_lab.strategies import v3_select  # noqa: E402

PARAMS = {"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5}
IS_END = pd.Timestamp("2024-01-01")


def compute_dispersion(data: dict, idx_map: dict) -> float | None:
    """动量离散度 = 各ETF动量分的标准差 (因果, 无前瞻)."""
    scores = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) >= 121:
            scores.append(rq.calc_momentum_score(close))
    if len(scores) < 3:
        return None
    return float(np.std(scores))


def build_panel(data: dict) -> pd.DataFrame:
    res = backtest(data, v3_select, PARAMS, 5)
    eq = res["equity_curve"].reset_index(drop=True)
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    rows = []
    for i in range(len(eq) - 1):
        td = eq["trade_date"].iloc[i]
        idx_map = build_idx_map(data, td.date())
        disp = compute_dispersion(data, idx_map)
        if disp is None:
            continue
        fwd_ret = eq["equity"].iloc[i + 1] / eq["equity"].iloc[i] - 1
        rows.append({"date": td, "dispersion": disp, "fwd_ret": fwd_ret})
    return pd.DataFrame(rows)


def main() -> None:
    data = rq.load_data()
    panel = build_panel(data)
    is_df = panel[panel["date"] < IS_END].copy()
    oos_df = panel[panel["date"] >= IS_END].copy()

    print("=" * 64)
    print("  周期识别 · 动量离散度 | 高分散=趋势期, 低分散=震荡期")
    print("=" * 64)

    is_corr = is_df["dispersion"].corr(is_df["fwd_ret"])
    oos_corr = oos_df["dispersion"].corr(oos_df["fwd_ret"])
    print(f"\n  离散度 与 V3下期收益 相关性:")
    print(f"    样本内 2020-2024: {is_corr:+.3f}")
    print(f"    样本外 2024-2026: {oos_corr:+.3f}")

    for label, df in [("样本内 2020-2024", is_df), ("样本外 2024-2026", oos_df)]:
        df = df.copy()
        df["tercile"] = pd.qcut(df["dispersion"], 3,
                                labels=["低分散(震荡)", "中分散", "高分散(趋势)"])
        print(f"\n  【{label}】 按离散度三分位:")
        print(f"  {'分组':<14}{'期数':>6}{'平均收益':>12}{'胜率':>10}{'累计':>12}")
        print("  " + "-" * 52)
        for t in ["低分散(震荡)", "中分散", "高分散(趋势)"]:
            sub = df[df["tercile"] == t]
            if sub.empty:
                continue
            avg = sub["fwd_ret"].mean() * 100
            wr = (sub["fwd_ret"] > 0).mean() * 100
            cum = ((1 + sub["fwd_ret"]).prod() - 1) * 100
            print(f"  {t:<14}{len(sub):>6}{avg:>+11.2f}%{wr:>9.0f}%{cum:>+11.1f}%")

    # 结论: 高分散是否稳定地带来更好表现
    def hi_lo(df):
        df = df.copy()
        df["t"] = pd.qcut(df["dispersion"], 3, labels=["lo", "mid", "hi"])
        return df[df["t"] == "hi"]["fwd_ret"].mean(), df[df["t"] == "lo"]["fwd_ret"].mean()
    is_hi, is_lo = hi_lo(is_df)
    oos_hi, oos_lo = hi_lo(oos_df)
    print(f"\n  【核心检验】 高分散 vs 低分散:")
    print(f"    样本内: 高 {is_hi*100:+.2f}% vs 低 {is_lo*100:+.2f}%  "
          f"({'高>低 ✓' if is_hi > is_lo else '高<低 ✗'})")
    print(f"    样本外: 高 {oos_hi*100:+.2f}% vs 低 {oos_lo*100:+.2f}%  "
          f"({'高>低 ✓' if oos_hi > oos_lo else '高<低 ✗'})")
    stable = (is_hi > is_lo) == (oos_hi > oos_lo)
    print(f"\n  【结论】 离散度周期信号样本内外"
          f"{'一致 ✓ (可作为周期判据)' if stable else '不一致 ✗ (仍不稳定)'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
