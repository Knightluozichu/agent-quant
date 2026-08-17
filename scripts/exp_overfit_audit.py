"""反过拟合审计 ① 收益集中度: 门控+豁免+H3 超额收益的时段/资产分布.

对标历史教训: R4提前换手 "OOS 3/4通过" 但超额95%集中在W4白银牛市,
W3震荡市真实亏损-12.3% → 被判定不可上线. 本脚本检查新方案是否同样
"牛市放大器": 超额(门控+豁免+H3 - 基线) 集中在哪些月份/年份/资产.

判定标准:
  - 超额按年分解, 若单一年份贡献 >70% 总超额 → 集中度过高, 过拟合风险大
  - 超额逐月累计曲线, 若增长集中在少数时段 → 同上
  - 同时输出各年超额符号 (正/负) 与幅度, 看是否有稳定正超额
用法: uv run python scripts/exp_overfit_audit.py
输出: data/v9_results/overfit_audit.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
import exp_drop_gate_h3 as h3  # noqa: E402
from exp_drop_gate_exempt import select_target_exempt  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"


def run_variant(data, gate: bool, exempt: bool, h3on: bool) -> dict:
    """跑一个变体, 返回 equity_curve 与指标."""
    rq.check_single_day_drop = h3.make_gated(0.00, True) if gate else h3.ORIG_CHECK
    h3.select_target = select_target_exempt if exempt else rq.select_target
    h3.H3_ENABLED = h3on
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False
    return rep


def main() -> None:
    print("=" * 76)
    print("  反过拟合审计① 收益集中度 | 门控+豁免+H3 vs 基线")
    print("=" * 76)

    data = rq.load_data()
    rep_a = run_variant(data, gate=False, exempt=False, h3on=False)
    rep_b = run_variant(data, gate=True, exempt=True, h3on=True)

    ea = rep_a["equity_curve"].copy()
    eb = rep_b["equity_curve"].copy()
    ea["d"] = ea["trade_date"]
    eb["d"] = eb["trade_date"]
    ea = ea.set_index("d")["equity"].sort_index()
    eb = eb.set_index("d")["equity"].sort_index()
    common = ea.index.intersection(eb.index)
    diff = (eb[common] - ea[common]).resample("ME").last()  # 月末超额净值差

    print(
        f"\n  基线期末 {rep_a['final_value']:,.0f} | "
        f"门控+豁免+H3 期末 {rep_b['final_value']:,.0f} | "
        f"超额 {rep_b['final_value'] - rep_a['final_value']:,.0f}"
    )

    # === 按年分解超额 (用月末净值差增量) ===
    monthly = diff.resample("ME").last()
    yearly_excess = {}
    prev = 0.0
    for d, v in monthly.items():
        y = str(d)[:4]
        delta = float(v) - prev
        yearly_excess[y] = yearly_excess.get(y, 0.0) + delta
        prev = float(v)
    total_excess = rep_b["final_value"] - rep_a["final_value"]
    print("\n  【按年超额贡献】")
    for y in sorted(yearly_excess):
        v = yearly_excess[y]
        print(f"    {y}: 超额 {v:>+12,.0f} ({v / total_excess * 100:>+6.1f}% of 总超额)")
    # 集中度判定
    sorted_years = sorted(yearly_excess, key=lambda k: -abs(yearly_excess[k]))
    top1 = yearly_excess[sorted_years[0]] / total_excess if total_excess else 0
    top2 = (
        (yearly_excess[sorted_years[0]] + yearly_excess[sorted_years[1]]) / total_excess
        if total_excess
        else 0
    )
    print(f"    最大单年贡献: {abs(top1):.1%} | 最大两年合计: {abs(top2):.1%}")

    # === 负超额月份统计 ===
    monthly_delta = diff.diff().dropna()
    neg_months = (monthly_delta < 0).sum()
    n_months = len(monthly_delta)
    print(f"\n  【月度超额】 负超额月份 {neg_months}/{n_months} ({neg_months / n_months:.0%})")

    # === 按资产统计: 放行事件后5日收益分布 (机制信息源) ===
    print("\n  【机制事件审计】")
    gated = h3.make_gated(0.00, True)
    GATE_STATS = {"passed": 0, "excluded": 0, "passed_detail": []}

    def gated_trace(close):
        ok = gated(close)
        if not ok:
            GATE_STATS["excluded"] += 1
        else:
            GATE_STATS["passed"] += 1
        return ok

    # 用 trace 脚本数据 (已有 JSON) 审计放行质量
    import pandas as pd

    trace_path = OUTPUT_DIR / "drop_gate_trace.json"
    if trace_path.exists():
        t = json.loads(trace_path.read_text())
        detail = pd.DataFrame(t["detail"])
        passed = detail[detail["gated_pass"]]
        print(f"  放行事件 {len(passed)} 次 (调仓日口径, ret60<0):")
        for code, grp in passed.groupby("code"):
            fwd = grp["fwd5"].dropna()
            if len(fwd) < 3:
                print(f"    {rq.ETF_POOL[code]}: {len(grp)}次 (后5日样本少)")
                continue
            print(
                f"    {rq.ETF_POOL[code]}: {len(grp)}次 后5日均 {fwd.mean():+.2%} "
                f"继续跌 {(fwd < 0).mean() * 100:.0f}%"
            )

    result = {
        "base_final": rep_a["final_value"],
        "new_final": rep_b["final_value"],
        "total_excess": total_excess,
        "yearly_excess": {k: round(v, 0) for k, v in yearly_excess.items()},
        "top1_year_share": round(abs(top1), 4),
        "top2_year_share": round(abs(top2), 4),
        "neg_months_ratio": round(neg_months / n_months, 3),
    }
    out = OUTPUT_DIR / "overfit_audit_concentration.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 结果已保存: {out}")
    print("\n  判定: 若单年贡献>70% 或 负超额月份>50%, 集中度过高 → 过拟合风险")


if __name__ == "__main__":
    main()
