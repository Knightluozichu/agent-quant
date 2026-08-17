"""实验: 七星V3 参数优化 (基于5日跌幅窗口).

锁定 DROP_LOOKBACK=5, 对以下关键参数做网格搜索:
  1. DROP_THRESHOLD: 单日跌幅阈值 [-0.02, -0.03, -0.04, -0.05]
  2. MOM_PERIODS + MOM_WEIGHTS: 动量周期与权重组合
  3. REBALANCE_DAYS: 调仓频率 [3, 5, 7, 10]
  4. A_SHARE_MA: A股走弱判断均线 [10, 20, 30]
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as base

OUTPUT_DIR = base.OUTPUT_DIR


def run_with_params(data: dict, params: dict) -> dict:
    """临时修改参数跑回测, 返回结果."""
    # 保存原始值
    originals = {}
    for key, val in params.items():
        originals[key] = getattr(base, key)
        setattr(base, key, val)
    try:
        result = base.run_qixing_v3(data)
    finally:
        # 恢复
        for key, val in originals.items():
            setattr(base, key, val)
    return result


def score_result(result: dict) -> float:
    """综合评分: 年化收益 × 夏普 / (1 + |最大回撤|)."""
    if "error" in result:
        return -999
    ann = result["ann_return"]
    sharpe = result["sharpe"]
    mdd = abs(result["max_drawdown"])
    # 兼顾收益、风险调整、回撤控制
    return ann * sharpe / (1 + mdd)


def main():
    print("=" * 70)
    print("  七星V3 参数优化 (DROP_LOOKBACK=5 锁定)")
    print("=" * 70)

    data = base.load_data()
    print(f"  数据: {len(data)}只ETF\n")

    # === 参数空间 ===
    drop_thresholds = [-0.02, -0.025, -0.03, -0.04, -0.05]
    mom_configs = [
        ((10, 20), (0.5, 0.5), "10+20 等权"),
        ((5, 20), (0.5, 0.5), "5+20 等权"),
        ((10, 30), (0.5, 0.5), "10+30 等权"),
        ((10, 20), (0.6, 0.4), "10+20 短重"),
        ((10, 20), (0.4, 0.6), "10+20 长重"),
        ((5, 10, 20), (0.4, 0.3, 0.3), "5+10+20"),
        ((10, 20, 60), (0.4, 0.3, 0.3), "10+20+60"),
        ((20, 60), (0.5, 0.5), "20+60 等权"),
    ]
    rebalance_days_list = [3, 5, 7, 10]
    a_share_mas = [10, 20, 30]

    # === 第一轮: 粗扫 DROP_THRESHOLD × REBALANCE_DAYS ===
    print("  [第1轮] DROP_THRESHOLD × REBALANCE_DAYS")
    print(f"  {'参数':<30} {'年化':>8} {'夏普':>6} {'回撤':>8} {'评分':>8}")
    print(f"  {'-' * 64}")

    best_score_r1 = -999
    best_params_r1 = {}
    for dt in drop_thresholds:
        for rd in rebalance_days_list:
            params = {"DROP_LOOKBACK": 5, "DROP_THRESHOLD": dt, "REBALANCE_DAYS": rd}
            r = run_with_params(data, params)
            s = score_result(r)
            if s > best_score_r1:
                best_score_r1 = s
                best_params_r1 = params.copy()
            if rd == 5:  # 只打印部分
                print(
                    f"  DT={dt:<6} RD={rd:<3}              "
                    f"{r['ann_return']:>+8.1%} {r['sharpe']:>6.2f} "
                    f"{r['max_drawdown']:>8.1%} {s:>8.3f}"
                )

    print(
        f"\n  ★ 第1轮最优: DT={best_params_r1['DROP_THRESHOLD']}, "
        f"RD={best_params_r1['REBALANCE_DAYS']} (评分={best_score_r1:.3f})"
    )

    # === 第二轮: 动量周期配置 (锁定第1轮最优) ===
    print(
        f"\n  [第2轮] 动量周期配置 (DT={best_params_r1['DROP_THRESHOLD']}, RD={best_params_r1['REBALANCE_DAYS']})"
    )
    print(f"  {'配置':<20} {'年化':>8} {'夏普':>6} {'回撤':>8} {'评分':>8}")
    print(f"  {'-' * 54}")

    best_score_r2 = -999
    best_mom = mom_configs[0]
    for periods, weights, label in mom_configs:
        params = {
            **best_params_r1,
            "MOM_PERIODS": periods,
            "MOM_WEIGHTS": weights,
        }
        r = run_with_params(data, params)
        s = score_result(r)
        print(
            f"  {label:<20} {r['ann_return']:>+8.1%} {r['sharpe']:>6.2f} "
            f"{r['max_drawdown']:>8.1%} {s:>8.3f}"
        )
        if s > best_score_r2:
            best_score_r2 = s
            best_mom = (periods, weights, label)

    print(f"\n  ★ 第2轮最优: {best_mom[2]} (评分={best_score_r2:.3f})")

    # === 第三轮: A股均线 (锁定前两轮) ===
    print(f"\n  [第3轮] A_SHARE_MA 均线周期")
    print(f"  {'MA':<10} {'年化':>8} {'夏普':>6} {'回撤':>8} {'评分':>8}")
    print(f"  {'-' * 44}")

    best_score_r3 = -999
    best_ma = 20
    for ma in a_share_mas:
        params = {
            **best_params_r1,
            "MOM_PERIODS": best_mom[0],
            "MOM_WEIGHTS": best_mom[1],
            "A_SHARE_MA": ma,
        }
        r = run_with_params(data, params)
        s = score_result(r)
        print(
            f"  MA={ma:<6} {r['ann_return']:>+8.1%} {r['sharpe']:>6.2f} "
            f"{r['max_drawdown']:>8.1%} {s:>8.3f}"
        )
        if s > best_score_r3:
            best_score_r3 = s
            best_ma = ma

    print(f"\n  ★ 第3轮最优: A_SHARE_MA={best_ma} (评分={best_score_r3:.3f})")

    # === 最终最优参数组合 ===
    final_params = {
        "DROP_LOOKBACK": 5,
        "DROP_THRESHOLD": best_params_r1["DROP_THRESHOLD"],
        "REBALANCE_DAYS": best_params_r1["REBALANCE_DAYS"],
        "MOM_PERIODS": best_mom[0],
        "MOM_WEIGHTS": best_mom[1],
        "A_SHARE_MA": best_ma,
    }

    print(f"\n{'=' * 70}")
    print(f"  最终最优参数组合")
    print(f"{'=' * 70}")
    for k, v in final_params.items():
        print(f"    {k} = {v}")

    # 跑最终结果
    final_result = run_with_params(data, final_params)
    eq = final_result["equity_curve"]

    print(f"\n  总收益: {final_result['total_return']:+.1%}")
    print(f"  年化:   {final_result['ann_return']:+.1%}")
    print(f"  夏普:   {final_result['sharpe']:.2f}")
    print(f"  回撤:   {final_result['max_drawdown']:.1%}")
    print(f"  交易:   {final_result['n_trades']}次")

    # 年度明细
    print(f"\n  {'年份':<6} {'收益':>10} {'回撤':>10}")
    print(f"  {'-' * 28}")
    for year in sorted(final_result["yearly"].keys()):
        yr_data = final_result["yearly"][year]
        print(f"  {year:<6} {yr_data['return']:>+10.2%} {yr_data['max_dd']:>10.2%}")

    # === 与当前生产参数对比 ===
    print(f"\n{'=' * 70}")
    print(f"  对比: 当前生产参数 vs 优化后参数")
    print(f"{'=' * 70}")
    baseline = run_with_params(data, {"DROP_LOOKBACK": 3})  # 当前生产
    print(f"  {'指标':<10} {'当前(3日)':>12} {'优化后':>12} {'提升':>12}")
    print(f"  {'-' * 48}")
    for name, key, is_pct in [
        ("年化收益", "ann_return", True),
        ("夏普比率", "sharpe", False),
        ("最大回撤", "max_drawdown", True),
        ("交易次数", "n_trades", False),
    ]:
        v_old = baseline[key]
        v_new = final_result[key]
        if is_pct:
            print(f"  {name:<10} {v_old:>+12.2%} {v_new:>+12.2%} {v_new - v_old:>+12.2%}")
        else:
            print(f"  {name:<10} {v_old:>12.1f} {v_new:>12.1f} {v_new - v_old:>+12.1f}")

    # 年度对比
    print(f"\n  {'年份':<6} {'当前':>10} {'优化后':>10} {'差异':>10}")
    print(f"  {'-' * 38}")
    all_years = sorted(set(baseline["yearly"].keys()) | set(final_result["yearly"].keys()))
    for year in all_years:
        r_old = baseline["yearly"].get(year, {}).get("return", 0)
        r_new = final_result["yearly"].get(year, {}).get("return", 0)
        print(f"  {year:<6} {r_old:>+10.2%} {r_new:>+10.2%} {r_new - r_old:>+10.2%}")

    # 保存结果
    summary = {
        "experiment": "V3参数优化(DROP_LOOKBACK=5)",
        "best_params": {k: str(v) for k, v in final_params.items()},
        "total_return": final_result["total_return"],
        "ann_return": final_result["ann_return"],
        "sharpe": final_result["sharpe"],
        "max_drawdown": final_result["max_drawdown"],
        "n_trades": final_result["n_trades"],
        "yearly": {str(k): v for k, v in final_result["yearly"].items()},
        "baseline_ann_return": baseline["ann_return"],
        "baseline_sharpe": baseline["sharpe"],
        "baseline_max_dd": baseline["max_drawdown"],
    }
    out_path = OUTPUT_DIR / "param_optimize_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
