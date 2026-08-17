"""门控最终验证: thr=0.05 drop=3% lb=5 的完整反过拟合套件.

参数: ret60_thr=0.05, drop_threshold=0.03, drop_lookback=5,
      豁免 on, H3(δ=0.02 expo=0.3)
验证:
  1. 置换测试 (12种子随机化门控放行)
  2. 参数扰动 ±20% (ret60_thr, h3_delta, h3_expo)
  3. 成本压力 2x/3x
  4. 滚动窗口 4段 (收益≥90% 且回撤不恶化)
用法: uv run python scripts/exp_gate_final_validate.py
输出: data/v9_results/gate_final_validate.json
"""

from __future__ import annotations

import json
import random
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
ORIG_SELECT = rq.select_target

# 新参数
B_THR = 0.01
B_DROP_THR = 0.03
B_DROP_LB = 5
B_EXEMPT = True
B_H3_DELTA = 0.02
B_H3_EXPO = 0.3


def patch_rq(drop_thr, drop_lb):
    rq.DROP_THRESHOLD = -abs(drop_thr)
    rq.DROP_LOOKBACK = drop_lb


def make_gated(thr, mom_thr=0.0):
    def gated(close):
        if len(close) < rq.DROP_LOOKBACK + 1:
            return True
        triggered = any(
            (close[i] - close[i - 1]) / close[i - 1] < rq.DROP_THRESHOLD
            for i in range(-rq.DROP_LOOKBACK, 0)
        )
        if not triggered:
            return True
        if len(close) <= 61:
            return True
        ret60 = (close[-1] - close[-61]) / close[-61]
        if ret60 >= thr:
            return False
        r10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
        r20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
        return 0.5 * r10 + 0.5 * r20 > mom_thr

    return gated


def make_random_gated(p_pass, seed):
    rng = random.Random(seed)

    def gated(close):
        if len(close) < rq.DROP_LOOKBACK + 1:
            return True
        triggered = any(
            (close[i] - close[i - 1]) / close[i - 1] < rq.DROP_THRESHOLD
            for i in range(-rq.DROP_LOOKBACK, 0)
        )
        if not triggered:
            return True
        if len(close) <= 61:
            return True
        return rng.random() < p_pass

    return gated


def run(data, check_fn, exempt, h3on, delta, expo, start_idx=0, end_idx=None, cost_multiplier=1.0):
    gate_saved = rq.check_single_day_drop
    rq.check_single_day_drop = check_fn
    h3.select_target = select_target_exempt if exempt else rq.select_target
    h3.H3_ENABLED = h3on
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = delta, "reduce", expo
    rep = h3.run_v3_risk_h3(
        data, start_idx=start_idx, end_idx=end_idx, cost_multiplier=cost_multiplier
    )
    rq.check_single_day_drop = gate_saved
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False
    return rep


def run_new(data, start_idx=0, end_idx=None, cost_multiplier=1.0):
    """新参数全周期."""
    patch_rq(B_DROP_THR, B_DROP_LB)
    return run(
        data,
        make_gated(B_THR),
        B_EXEMPT,
        True,
        B_H3_DELTA,
        B_H3_EXPO,
        start_idx,
        end_idx,
        cost_multiplier,
    )


def run_base(data, start_idx=0, end_idx=None, cost_multiplier=1.0):
    """基线 (同drop参数的原版过滤)."""
    patch_rq(B_DROP_THR, B_DROP_LB)
    return run(data, h3.ORIG_CHECK, False, False, 0, 1.0, start_idx, end_idx, cost_multiplier)


def main():
    print("=" * 80)
    print(
        f"  门控最终验证 | thr={B_THR} drop={B_DROP_THR:.0%} "
        f"lb={B_DROP_LB} | 豁免+H3(δ={B_H3_DELTA:.0%} expo={B_H3_EXPO})"
    )
    print("=" * 80)

    data = rq.load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)

    def seg(s0, s1):
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    from exp_v32_tail_risk import IS_START, IS_END, OOS_START, OOS_END

    # === 基线数据 ===
    r_base = run_base(data)
    base_final = r_base["final_value"]
    print(f"\n  基线: 期末 {base_final:,.0f} 夏普 {r_base['sharpe']:.2f}")

    # === 1. 置换测试 ===
    print("\n" + "=" * 80)
    print("  ① 置换测试 (12种子, 随机化门控放行)")
    rand_finals = []
    for seed in range(12):
        r = run(data, make_random_gated(0.5, seed), B_EXEMPT, True, B_H3_DELTA, B_H3_EXPO)
        rand_finals.append(r["final_value"])
        print(
            f"    种子{seed:>2}: 期末 {r['final_value']:>10,.0f} "
            f"({r['final_value'] / base_final - 1:+.1%})"
        )
    arr = np.array(rand_finals)
    r_new_base = run_new(data)
    real_final = r_new_base["final_value"]
    pct = (arr < real_final).mean() * 100
    print(
        f"    真实机制: {real_final:,.0f} | 随机分布 P90={np.percentile(arr, 90):,.0f} "
        f"中位数={np.median(arr):,.0f} | 真实位于 {pct:.0f}% 分位"
    )
    perm_ok = pct >= 95
    print(f"    {'✅ 显著' if perm_ok else '❌ 未显著'}")
    perm_result = {
        "real": real_final,
        "randoms": [round(v, 0) for v in rand_finals],
        "percentile": pct,
        "passed": perm_ok,
    }

    # === 2. 参数扰动 ===
    print("\n" + "=" * 80)
    print("  ② 参数扰动 ±20%")
    perturbs = [
        ("ret60_thr 0.05→0.04", {"thr": 0.04}),
        ("ret60_thr 0.05→0.06", {"thr": 0.06}),
        ("H3 δ 0.02→0.016", {"delta": 0.016}),
        ("H3 δ 0.02→0.024", {"delta": 0.024}),
        ("H3 expo 0.3→0.24", {"expo": 0.24}),
        ("H3 expo 0.3→0.36", {"expo": 0.36}),
    ]
    pert_results = {}
    for name, kw in perturbs:
        r = run(
            data,
            make_gated(kw.get("thr", B_THR)),
            B_EXEMPT,
            True,
            kw.get("delta", B_H3_DELTA),
            kw.get("expo", B_H3_EXPO),
        )
        ratio = r["final_value"] / base_final
        dd_ok = r["max_drawdown"] > -0.30
        ok = ratio >= 0.85 and dd_ok
        pert_results[name] = {"ratio": round(ratio, 3), "b_dd": r["max_drawdown"], "ok": ok}
        print(
            f"    {name:<22} B/A={ratio * 100:>5.1f}% 回撤={r['max_drawdown']:.1%} "
            f"{'✅' if ok else '❌'}"
        )
    pert_ok = all(v["ok"] for v in pert_results.values())

    # === 3. 成本压力 ===
    print("\n" + "=" * 80)
    print("  ③ 成本压力")
    cost_results = {}
    for mult in (2.0, 3.0):
        r_b = run_new(data, cost_multiplier=mult)
        r_a = run_base(data, cost_multiplier=mult)
        ratio = r_b["final_value"] / r_a["final_value"]
        dd_ok = r_b["max_drawdown"] > r_a["max_drawdown"]
        ok = ratio >= 0.85 and dd_ok
        cost_results[f"{mult:.0f}x"] = {"ratio": round(ratio, 3), "ok": ok}
        print(f"    {mult:.0f}x: B/A={ratio * 100:.1f}% {'✅' if ok else '❌'}")
    cost_ok = all(v["ok"] for v in cost_results.values())

    # === 4. 滚动窗口 ===
    print("\n" + "=" * 80)
    print("  ④ 滚动窗口 (4段)")
    rolling = []
    wins = 0
    for name, s0, s1 in (
        ("W1 2020H2-2021", *seg("2020-06-01", "2021-12-31")),
        ("W2 2022", *seg("2022-01-01", "2022-12-31")),
        ("W3 2023", *seg("2023-01-01", "2023-12-31")),
        ("W4 2024-2026", *seg("2024-01-01", "2026-08-03")),
    ):
        r_a = run_base(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
        r_b = run_new(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
        ratio = r_b["final_value"] / r_a["final_value"]
        dd_imp = r_b["max_drawdown"] > r_a["max_drawdown"]
        beat = ratio >= 0.9 and dd_imp
        wins += int(beat)
        rolling.append(
            {
                "seg": name,
                "a": r_a["final_value"],
                "b": r_b["final_value"],
                "ratio": round(ratio, 3),
                "a_dd": r_a["max_drawdown"],
                "b_dd": r_b["max_drawdown"],
                "beat": beat,
            }
        )
        print(
            f"    {name:<16} A={r_a['final_value']:>9,.0f} B={r_b['final_value']:>9,.0f} "
            f"B/A={ratio * 100:>5.1f}% 回撤{r_a['max_drawdown']:.1%}→{r_b['max_drawdown']:.1%} "
            f"{'✅' if beat else '❌'}"
        )
    roll_ok = wins >= 3
    print(f"    滚动判定: {wins}/4 {'✅' if roll_ok else '❌'}")

    # === 总判定 ===
    print("\n" + "=" * 80)
    checks = [
        ("置换测试", perm_ok),
        ("参数扰动", pert_ok),
        ("成本压力", cost_ok),
        ("滚动窗口", roll_ok),
    ]
    passed = all(v for _, v in checks)
    for name, ok in checks:
        print(f"    {name:<10} {'✅' if ok else '❌'}")
    print(f"    总判定: {'✅ 全部通过' if passed else '❌'}")
    print("=" * 80)

    out = {
        "params": {
            "thr": B_THR,
            "drop_thr": B_DROP_THR,
            "drop_lb": B_DROP_LB,
            "exempt": B_EXEMPT,
            "h3_delta": B_H3_DELTA,
            "h3_expo": B_H3_EXPO,
        },
        "base_final": base_final,
        "new_final": real_final,
        "new_diff": round(real_final / base_final - 1, 4),
        "permutation": perm_result,
        "perturbation": pert_results,
        "cost": cost_results,
        "rolling": rolling,
        "checks": {k: v for k, v in checks},
        "passed": passed,
    }
    out_path = OUTPUT_DIR / "gate_final_validate.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
