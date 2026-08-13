"""动量周期双网格扫描 (V3-G 引擎): 短周期 × 长周期.

短周期 S ∈ {3,5,8,10,13,15,20} (覆盖3-20日)
长周期 L ∈ {13,15,20,25,30} (覆盖13-30日)
约束: L > S; 权重固定 (0.5,0.5); 调仓5日; V3-G (门控0.01+豁免+无降仓)

用法: uv run python scripts/exp_gate_mom_period_scan.py
输出: data/v9_results/gate_mom_period_scan.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import exp_drop_gate_h3 as h3  # noqa: E402
import run_qixing_v3 as rq  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
SHORT = [3, 5, 8, 10, 13, 15, 20]
LONG = [13, 15, 20, 25, 30]


def run_with_periods(data, short_period, long_period):
    orig = rq.MOM_PERIODS
    rq.MOM_PERIODS = (short_period, long_period)
    rq.MOM_WEIGHTS = (0.5, 0.5)
    rq.check_single_day_drop = h3.make_gated(0.01, True)
    h3.select_target = rq.select_target
    rep = h3.run_v3_risk_h3(data)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    rq.MOM_PERIODS = orig
    return rep


def main() -> None:
    print("=" * 78)
    print("  动量周期双网格 | 短周期×长周期 | 权重0.5/0.5 | V3-G")
    print("=" * 78)

    data = rq.load_data()
    results = {}
    combos = [(short, long) for short in SHORT for long in LONG if long > short]
    print(f"\n  组合数: {len(combos)} (短{SHORT} × 长{LONG}, 约束长>短)")
    print(f"\n  {'短/长':<8}" + "".join(f"{long:>10}" for long in LONG))
    final_mat = {}
    for s in SHORT:
        final_mat[s] = {}
    for short, long in combos:
        rep = run_with_periods(data, short, long)
        results[f"{short}/{long}"] = {
            "short": short,
            "long": long,
            "final": rep["final_value"],
            "ann": rep["ann_return"],
            "sharpe": rep["sharpe"],
            "dd": rep["max_drawdown"],
            "n_trades": rep["n_trades"],
        }
        final_mat[short][long] = rep["final_value"]
        print(
            f"  {short:>2}/{long:<3} 期末{rep['final_value']:>10,.0f} "
            f"年化{rep['ann_return'] * 100:>+6.1f}% 夏普{rep['sharpe']:>5.2f} "
            f"回撤{rep['max_drawdown'] * 100:>6.1f}%"
        )

    # 矩阵输出
    print("\n" + "=" * 78)
    print("  期末收益矩阵 (万):")
    print(f"  {'短\\长':<6}" + "".join(f"{long:>8}" for long in LONG))
    for short in SHORT:
        row = "".join(f"{final_mat[short].get(long, 0) / 10000:>8.0f}" for long in LONG)
        print(f"  {short:<6}" + row)

    # 最优
    best_key = max(results, key=lambda k: results[k]["final"])
    prod = results.get("10/20")
    print(
        f"\n  最优: {best_key} 期末{results[best_key]['final']:,.0f} "
        f"年化{results[best_key]['ann'] * 100:+.1f}% 夏普{results[best_key]['sharpe']:.2f}"
    )
    if prod:
        d = results[best_key]["final"] / prod["final"] - 1
        print(f"  生产(10/20): 期末{prod['final']:,.0f} | 最优相对生产 {d:+.1%}")

    out = OUTPUT_DIR / "gate_mom_period_scan.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()
