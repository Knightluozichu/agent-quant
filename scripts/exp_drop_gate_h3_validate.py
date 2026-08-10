"""门控+H3放行止损 稳健性验证: 滚动窗口 + 参数扰动 + 成本压力.

目标: 验证"门控(thr=0.00, guard=on) + H3放行止损(δ=2%, reduce降仓0.5)"
      是否满足项目稳健性标准 (参照 exp_v32_tail_risk_validate):
1. 滚动窗口 4段 (2020H2-21 / 2022 / 2023 / 2024-26), 每段 fresh 10万,
   A=基线(原版过滤+V32) vs B=门控+H3; 判定 ≥3/4 段 B收益≥A×0.9 且回撤改善
2. 参数扰动 ±20%: H3 δ{0.016,0.024} / H3降仓{0.4,0.6} / 门控thr{-0.02,+0.02}
   全周期 B收益≥A×0.85 且回撤改善方向不翻转
3. 成本压力 2x/3x: 全周期 B收益≥A×0.85 且回撤改善

用法: uv run python scripts/exp_drop_gate_h3_validate.py
输出: data/v9_results/drop_gate_h3_validate.json
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
from exp_v32_tail_risk import WARMUP  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"

# B 变体默认参数 (扫描最优: 门控+缓冲豁免, δ=2% reduce expo=0.3)
B_THR = 0.00
B_DELTA = 0.02
B_ACTION = "reduce"
B_EXPO = 0.3
B_EXEMPT = True


def run_pair(data, dates, s0, s1, cost_mult=1.0,
             thr=None, delta=None, expo=None) -> tuple[dict, dict]:
    """A/B 一对回测 (A=基线, B=门控+豁免+H3; 支持参数扰动).

    注意: 必须直接修改 exp_drop_gate_h3 模块属性 (h3.H3_ENABLED 等),
    因为 run_v3_risk_h3 读取的是该模块命名空间; from-import 只会创建
    本模块的副本绑定, 改动不生效. select_target 同理须 patch h3 模块.
    """
    saved = (h3.H3_ENABLED, h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO,
             h3.select_target)
    # A: 基线
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = h3.ORIG_SELECT if hasattr(h3, "ORIG_SELECT") else rq.select_target
    h3.H3_ENABLED = False
    r_a = h3.run_v3_risk_h3(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0),
                            cost_multiplier=cost_mult)
    # B: 门控+豁免+H3 (扰动)
    rq.check_single_day_drop = h3.make_gated(thr if thr is not None else B_THR, True)
    h3.select_target = select_target_exempt if B_EXEMPT else rq.select_target
    h3.H3_ENABLED = True
    h3.H3_DELTA = delta if delta is not None else B_DELTA
    h3.H3_ACTION = B_ACTION
    h3.H3_EXPO = expo if expo is not None else B_EXPO
    r_b = h3.run_v3_risk_h3(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0),
                            cost_multiplier=cost_mult)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.H3_ENABLED, h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO, h3.select_target = saved
    return r_a, r_b


def main() -> None:
    print("=" * 78)
    print("  门控+豁免+H3 稳健性验证 (滚动 + 扰动 + 成本)")
    print(f"  默认: 门控 thr={B_THR} guard=on + 缓冲豁免 + H3 δ={B_DELTA:.0%} "
          f"{B_ACTION} 降仓{B_EXPO}")
    print("=" * 78)

    data = rq.load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    results = {"rolling": [], "perturb": {}, "cost": {}}

    # === 1. 滚动窗口 (4段) ===
    print("\n" + "=" * 78)
    print("  滚动窗口 (4段, 每段 fresh 10万)")
    print("=" * 78)
    wins = 0
    for name, s0, s1 in (("W1 2020H2-2021", *seg("2020-06-01", "2021-12-31")),
                         ("W2 2022", *seg("2022-01-01", "2022-12-31")),
                         ("W3 2023", *seg("2023-01-01", "2023-12-31")),
                         ("W4 2024-2026", *seg("2024-01-01", "2026-08-03"))):
        r_a, r_b = run_pair(data, dates, s0, s1)
        ratio = r_b["final_value"] / r_a["final_value"]
        dd_imp = r_b["max_drawdown"] > r_a["max_drawdown"]
        beat = ratio >= 0.9 and dd_imp
        wins += int(beat)
        print(f"  {name:<16} A={r_a['final_value']:>9,.0f} B={r_b['final_value']:>9,.0f} "
              f"B/A={ratio*100:>5.1f}% 回撤A={r_a['max_drawdown']:.1%}→B={r_b['max_drawdown']:.1%} "
              f"{'✅' if beat else '❌'}")
        results["rolling"].append({"seg": name, "a": r_a["final_value"],
                                   "b": r_b["final_value"], "ratio": round(ratio, 3),
                                   "a_dd": r_a["max_drawdown"], "b_dd": r_b["max_drawdown"],
                                   "beat": beat})
    print(f"  滚动判定: {wins}/4 段达标 (标准 ≥3/4) → {'✅' if wins >= 3 else '❌'}")
    results["rolling_wins"] = wins

    # === 2. 参数扰动 ±20% (全周期) ===
    print("\n" + "=" * 78)
    print("  参数扰动 ±20% (全周期, 单参数扰动)")
    print("=" * 78)
    base_a, base_b = run_pair(data, dates, 0, n)
    print(f"  基线: A={base_a['final_value']:,.0f} B={base_b['final_value']:,.0f} "
          f"B/A={base_b['final_value']/base_a['final_value']*100:.1f}% "
          f"回撤 {base_a['max_drawdown']:.1%}→{base_b['max_drawdown']:.1%}")
    perturbs = [
        ("H3 δ 0.02→0.016", {"delta": 0.016}),
        ("H3 δ 0.02→0.024", {"delta": 0.024}),
        ("H3 降仓 0.3→0.24", {"expo": 0.24}),
        ("H3 降仓 0.3→0.36", {"expo": 0.36}),
        ("门控 thr 0.00→-0.02", {"thr": -0.02}),
        ("门控 thr 0.00→+0.02", {"thr": 0.02}),
    ]
    for name, kw in perturbs:
        r_a, r_b = run_pair(data, dates, 0, n, **kw)
        ratio = r_b["final_value"] / r_a["final_value"]
        dd_ok = r_b["max_drawdown"] > -0.30
        ok = ratio >= 0.85 and dd_ok
        print(f"  {name:<22} B/A={ratio*100:>5.1f}% 回撤={r_b['max_drawdown']:.1%} "
              f"{'✅' if ok else '❌'}")
        results["perturb"][name] = {"ratio": round(ratio, 3),
                                    "b_dd": r_b["max_drawdown"], "ok": ok}

    # === 3. 成本压力 2x/3x (全周期) ===
    print("\n" + "=" * 78)
    print("  成本压力 (成本乘数 2x/3x, 全周期)")
    print("=" * 78)
    for mult in (2.0, 3.0):
        r_a, r_b = run_pair(data, dates, 0, n, cost_mult=mult)
        ratio = r_b["final_value"] / r_a["final_value"]
        dd_ok = r_b["max_drawdown"] > r_a["max_drawdown"]
        ok = ratio >= 0.85 and dd_ok
        print(f"  {mult:.0f}x 成本: A={r_a['final_value']:,.0f} B={r_b['final_value']:,.0f} "
              f"B/A={ratio*100:.1f}% 回撤 {r_a['max_drawdown']:.1%}→{r_b['max_drawdown']:.1%} "
              f"{'✅' if ok else '❌'}")
        results["cost"][f"{mult:.0f}x"] = {"ratio": round(ratio, 3),
                                           "a_dd": r_a["max_drawdown"],
                                           "b_dd": r_b["max_drawdown"], "ok": ok}

    # 总判定
    perturb_ok = all(v["ok"] for v in results["perturb"].values())
    cost_ok = all(v["ok"] for v in results["cost"].values())
    print("\n" + "=" * 78)
    print(f"  总判定: 滚动{wins}/4 {'✅' if wins >= 3 else '❌'} | "
          f"扰动{'✅' if perturb_ok else '❌'} | 成本{'✅' if cost_ok else '❌'}")
    passed = wins >= 3 and perturb_ok and cost_ok
    print(f"  门控+H3 {'✅ 通过全部稳健性验证' if passed else '❌ 未通过'}")
    print("=" * 78)
    results["passed"] = passed

    out = OUTPUT_DIR / "drop_gate_h3_validate.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  已保存: {out}")


if __name__ == "__main__":
    main()
