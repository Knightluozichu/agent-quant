"""二阶导(动量加速度)增强检验: 在V3动量(一阶导)上叠加加速度因子, IS/OOS验证.

综合分 = 动量(一阶导) + λ × 加速度(二阶导)
加速度 = 10日收益 - 30日收益 (短期动量 - 长期动量, >0=上涨加速)
λ=0 即纯V3基准。检验: 是否存在λ>0在样本外(2024-2026)优于λ=0。
用法: uv run python scripts/exp_acceleration.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import backtest  # noqa: E402

IS_END = date(2024, 1, 1)


def v3_accel_select(data: dict, idx_map: dict, holding, params: dict):
    """V3 + 二阶导加速度倾斜 (λ=0 时等价于V3)."""
    lam = params.get("accel_weight", 0.0)
    a_share_weak = (
        rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
        if rq.USE_A_SHARE_FILTER
        else False
    )
    candidates = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        df = data[code]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        mom = rq.calc_momentum_score(close)  # 一阶导
        ret10 = (close[-1] - close[-11]) / close[-11]
        ret30 = (close[-1] - close[-31]) / close[-31]
        accel = ret10 - ret30  # 二阶导(加速度)
        score = mom + lam * accel
        if mom > 0:  # 仍要求正动量
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0
    threshold = 0.0 if best_score > 0.10 else 0.05
    if holding and holding != rq.DEFENSE:
        cur = dict(candidates).get(holding, -999)
        if cur > 0:
            target = best_target if best_score > cur + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target


def main() -> None:
    data = rq.load_data()
    print("=" * 70)
    print("  二阶导(动量加速度)增强检验 | 综合分 = 动量 + λ×加速度")
    print("  加速度 = 10日收益 - 30日收益  (λ=0 即纯V3)")
    print("=" * 70)
    print(f"  {'λ':<6}{'全段年化':>10}{'夏普':>8}{'回撤':>9}{'IS年化':>10}{'OOS年化':>10}")
    print("  " + "-" * 56)

    results = []
    for lam in [0.0, 0.5, 1.0, 2.0]:
        params = {
            "mom_periods": (10, 20),
            "mom_weights": (0.5, 0.5),
            "rebalance_days": 5,
            "accel_weight": lam,
        }
        full = backtest(data, v3_accel_select, params, 5)
        is_r = backtest(data, v3_accel_select, params, 5, end_date=IS_END)
        oos_r = backtest(data, v3_accel_select, params, 5, start_date=IS_END)
        results.append((lam, full, is_r, oos_r))
        print(
            f"  {lam:<6}{full['ann_return'] * 100:>+9.1f}%{full['sharpe']:>8.2f}"
            f"{full['max_drawdown'] * 100:>8.1f}%{is_r['ann_return'] * 100:>+9.1f}%"
            f"{oos_r['ann_return'] * 100:>+9.1f}%"
        )

    base_oos = results[0][3]["ann_return"]
    best = max(results, key=lambda x: x[3]["ann_return"])
    print("  " + "-" * 56)
    print(f"  基准(λ=0, 纯一阶导) OOS年化: {base_oos * 100:+.1f}%")
    print(f"  OOS最优: λ={best[0]} (OOS年化 {best[3]['ann_return'] * 100:+.1f}%)")
    if best[0] == 0.0:
        print("\n  【结论】 λ=0(纯一阶导)样本外最优 → 二阶导无增强, 加了是过拟合。保持V3原样 ✓")
    else:
        imp = (best[3]["ann_return"] - base_oos) * 100
        print(f"\n  【结论】 λ={best[0]} 样本外略优 ({imp:+.1f}百分点)。")
        print("         但需警惕OOS侥幸, 建议模拟盘长期观察, 不直接上实盘。")
    print("=" * 70)


if __name__ == "__main__":
    main()
