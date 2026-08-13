"""V3-G 引擎下动量参数网格扫描: 周期 × 权重 (调仓固定5日).

背景: 历史动量扫描 (7/31 param_optimize) 在旧引擎 (T+1/无门控/有降仓) 上做,
      结论 MOM_WEIGHTS=(0.4,0.6) 略优于生产 (0.5,0.5). V3-G (14:50+门控+豁免+无降仓)
      下是否仍成立? 本脚本在 V3-G 引擎下重扫.

网格: MOM_PERIODS × MOM_WEIGHTS (REBALANCE_DAYS=5 固定)
  周期: (10,20)当前 / (10,30) / (20,60)
  权重: (0.5,0.5)当前 / (0.4,0.6) / (0.6,0.4)

用法: uv run python scripts/exp_gate_mom_scan.py
输出: data/v9_results/gate_mom_scan.json
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


def run_with_mom(data, periods, weights):
    """patch 动量参数后跑 V3-G 全量回测."""
    orig_p = rq.MOM_PERIODS
    orig_w = rq.MOM_WEIGHTS
    rq.MOM_PERIODS = periods
    rq.MOM_WEIGHTS = weights
    rq.check_single_day_drop = h3.make_gated(0.01, True)
    h3.select_target = rq.select_target
    rep = h3.run_v3_risk_h3(data)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    rq.MOM_PERIODS = orig_p
    rq.MOM_WEIGHTS = orig_w
    return rep


def main() -> None:
    print("=" * 76)
    print("  V3-G 引擎下动量参数扫描 | 周期 × 权重 (调仓5日)")
    print("=" * 76)

    data = rq.load_data()
    grid = [
        ("生产(10,20,0.5/0.5)", (10, 20), (0.5, 0.5)),
        ("10/20 权重0.4/0.6", (10, 20), (0.4, 0.6)),
        ("10/20 权重0.6/0.4", (10, 20), (0.6, 0.4)),
        ("10/30 权重0.5/0.5", (10, 30), (0.5, 0.5)),
        ("10/30 权重0.4/0.6", (10, 30), (0.4, 0.6)),
        ("10/30 权重0.6/0.4", (10, 30), (0.6, 0.4)),
        ("20/60 权重0.5/0.5", (20, 60), (0.5, 0.5)),
        ("20/60 权重0.4/0.6", (20, 60), (0.4, 0.6)),
        ("20/60 权重0.6/0.4", (20, 60), (0.6, 0.4)),
    ]

    results = []
    for name, periods, weights in grid:
        rep = run_with_mom(data, periods, weights)
        results.append({"name": name, "periods": list(periods),
                        "weights": list(weights),
                        "final": rep["final_value"],
                        "ann_return": rep["ann_return"],
                        "sharpe": rep["sharpe"],
                        "max_dd": rep["max_drawdown"],
                        "n_trades": rep["n_trades"]})
        print(f"  {name:<22} 期末{rep['final_value']:>10,.0f} "
              f"年化{rep['ann_return']*100:>+6.1f}% 夏普{rep['sharpe']:>5.2f} "
              f"回撤{rep['max_drawdown']*100:>6.1f}% 交易{rep['n_trades']:>3}")

    # 对比生产
    base = results[0]
    print("\n" + "=" * 76)
    print("  相对生产 (10,20,0.5/0.5):")
    for r in results[1:]:
        diff = r["final"] / base["final"] - 1
        print(f"  {r['name']:<22} {diff:+.1%} {'✅超生产' if diff > 0 else ''}")

    out = OUTPUT_DIR / "gate_mom_scan.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()
