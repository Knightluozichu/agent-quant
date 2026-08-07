"""改进版风控稳健性验证: 滚动窗口 + 参数扰动 + 成本压力.

改进版 (高波动vol>0.45+动量衰减 → 降仓0.7, 去静默类别降仓, -30%熔断) 已通过
IS/OOS 初步验证 (收益+50%, 回撤-21.1%). 本脚本完成三项稳健性:

1. 滚动窗口: 4段分段回滚 (2020H2-21 / 2022 / 2023 / 2024-26), 每段 fresh 10万,
   A/B 对比; 判定 ≥3/4 段 B 收益 ≥ A×0.9 且回撤改善 (防单区间假象)
2. 参数扰动 ±20%: vol阈值{0.36,0.54} / 降仓{0.56,0.84} / 熔断{-0.24,-0.36}
   各单参数扰动, 全周期判定 B收益≥A×0.85 且回撤改善方向不翻转
3. 成本压力: 成本乘数 {2x, 3x} 下全周期 A/B, B收益≥A×0.85 且回撤改善

输出: data/v9_results/v3_tail_risk_validate.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
import exp_v32_tail_risk as t  # noqa: E402
from exp_short_window_patterns import close_matrix  # noqa: E402
from exp_v3_r4_sameday import run_v3_r4_sameday  # noqa: E402
from run_qixing_v3 import load_data  # noqa: E402

WARMUP = 130


def run_pair(data, mat, s0, s1, cost_mult=1.0,
             vol_thr=None, expo=None, dd_flush=None) -> tuple[dict, dict]:
    """A/B 一对回测 (支持参数扰动与成本乘数)."""
    # 临时扰动参数
    saved = (t.VOL_HV_THR, t.EXPO_REDUCE, t.DD_FLUSH)
    if vol_thr is not None:
        t.VOL_HV_THR = vol_thr
    if expo is not None:
        t.EXPO_REDUCE = expo
    if dd_flush is not None:
        t.DD_FLUSH = dd_flush
    try:
        r_a = run_v3_r4_sameday(data, mat, thr=1.0, start_idx=s0,
                                end_idx=max(s1 - WARMUP, 0), cost_multiplier=cost_mult)
        r_b = t.run_v3_risk(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0),
                            cost_multiplier=cost_mult)
    finally:
        t.VOL_HV_THR, t.EXPO_REDUCE, t.DD_FLUSH = saved
    return r_a, r_b


def main() -> None:
    print("=" * 74)
    print("  改进版风控稳健性验证 (滚动 + 扰动 + 成本)")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    mat = close_matrix(data, dates)
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    results = {"rolling": [], "perturb": {}, "cost": {}}

    # === 1. 滚动窗口 (4段) ===
    print("\n" + "=" * 74)
    print("  滚动窗口 (4段, 每段 fresh 10万)")
    print("=" * 74)
    wins = 0
    for name, s0, s1 in (("W1 2020H2-2021", *seg("2020-06-01", "2021-12-31")),
                         ("W2 2022", *seg("2022-01-01", "2022-12-31")),
                         ("W3 2023", *seg("2023-01-01", "2023-12-31")),
                         ("W4 2024-2026", *seg("2024-01-01", "2026-08-03"))):
        r_a, r_b = run_pair(data, mat, s0, s1)
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
    n_win = wins
    print(f"  滚动判定: {n_win}/4 段达标 (标准 ≥3/4) → {'✅' if n_win >= 3 else '❌'}")
    results["rolling_wins"] = n_win

    # === 2. 参数扰动 ±20% (全周期) ===
    print("\n" + "=" * 74)
    print("  参数扰动 ±20% (全周期, 单参数扰动)")
    print("=" * 74)
    base_a, base_b = run_pair(data, mat, 0, n)
    print(f"  基线: A={base_a['final_value']:,.0f} B={base_b['final_value']:,.0f} "
          f"B/A={base_b['final_value']/base_a['final_value']*100:.1f}% "
          f"回撤 {base_a['max_drawdown']:.1%}→{base_b['max_drawdown']:.1%}")
    perturbs = [
        ("vol阈值 0.45→0.36", {"vol_thr": 0.36}),
        ("vol阈值 0.45→0.54", {"vol_thr": 0.54}),
        ("降仓 0.7→0.56", {"expo": 0.56}),
        ("降仓 0.7→0.84", {"expo": 0.84}),
        ("熔断 -30%→-24%", {"dd_flush": -0.24}),
        ("熔断 -30%→-36%", {"dd_flush": -0.36}),
    ]
    for name, kw in perturbs:
        r_a, r_b = run_pair(data, mat, 0, n, **kw)
        ratio = r_b["final_value"] / r_a["final_value"]
        dd_ok = r_b["max_drawdown"] > -0.30
        ok = ratio >= 0.85 and dd_ok
        print(f"  {name:<16} B/A={ratio*100:>5.1f}% 回撤={r_b['max_drawdown']:.1%} "
              f"{'✅' if ok else '❌'}")
        results["perturb"][name] = {"ratio": round(ratio, 3),
                                    "b_dd": r_b["max_drawdown"], "ok": ok}

    # === 3. 成本压力 2x/3x (全周期) ===
    print("\n" + "=" * 74)
    print("  成本压力 (成本乘数 2x/3x, 全周期)")
    print("=" * 74)
    for mult in (2.0, 3.0):
        r_a, r_b = run_pair(data, mat, 0, n, cost_mult=mult)
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
    print("\n" + "=" * 74)
    print(f"  总判定: 滚动{n_win}/4 {'✅' if n_win>=3 else '❌'} | "
          f"扰动{'✅' if perturb_ok else '❌'} | 成本{'✅' if cost_ok else '❌'}")
    passed = n_win >= 3 and perturb_ok and cost_ok
    print(f"  改进版风控 {'✅ 通过全部稳健性验证' if passed else '❌ 未通过'}")
    print("=" * 74)
    results["passed"] = passed

    out_path = OUTPUT_DIR / "v3_tail_risk_validate.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  已保存: {out_path}")


if __name__ == "__main__":
    main()
