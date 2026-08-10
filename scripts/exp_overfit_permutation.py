"""反过拟合审计 ② 置换测试: ret60门控 vs 随机放行.

问题: 门控+豁免+H3 的超额 100% 来自 2026-07-15 单次换仓决策 (n=1),
      无法区分"ret60 判据有真实信息"与"牛市环境+运气".
方法: 保持豁免/H3 逻辑不变, 仅把"门控放行决策"随机化:
        - 真实版: 触发暴跌后 ret60<0 → 放行 (ret60>=0 → 排除)
        - 随机版: 触发暴跌后 random()<p → 放行 (p≈真实放行率 0.40)
      跑 N 个随机种子, 得到随机放行的期末净值分布;
      比较真实机制 3,120,703 在随机分布中的分位.
      若真实值 < 90% 分位 → ret60 门控无显著信息 (可能是运气/环境)
      若真实值 > 95% 分位 → 有统计显著信息
用法: uv run python scripts/exp_overfit_permutation.py [seeds=8]
输出: data/v9_results/overfit_permutation.json
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

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
P_PASS = 0.40  # 真实放行率 (ret60<0 在触发暴跌中的比例, 调仓日口径 85/213)


def make_random_gated(seed: int, p_pass: float = P_PASS):
    """随机版门控: 触发暴跌后按概率 p 放行 (不查 ret60)."""
    rng = random.Random(seed)

    def gated(close: np.ndarray) -> bool:
        if len(close) < rq.DROP_LOOKBACK + 1:
            return True
        triggered = False
        for i in range(-rq.DROP_LOOKBACK, 0):
            dr = (close[i] - close[i - 1]) / close[i - 1]
            if dr < rq.DROP_THRESHOLD:
                triggered = True
                break
        if not triggered:
            return True
        if len(close) <= 61:
            return True
        return rng.random() < p_pass
    return gated


def run_variant(data, check_fn, exempt: bool, h3on: bool) -> dict:
    rq.check_single_day_drop = check_fn
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
    print(f"  反过拟合审计② 置换测试 | ret60门控 vs 随机放行 ({N_SEEDS}种子)")
    print("=" * 76)

    data = rq.load_data()

    # 真实机制
    rep_real = run_variant(data, h3.make_gated(0.00, True), exempt=True, h3on=True)
    real_final = rep_real["final_value"]
    print(f"\n  真实机制 (ret60门控+豁免+H3): 期末 {real_final:,.0f}")

    # 基线 (无门控无豁免无H3)
    rep_base = run_variant(data, h3.ORIG_CHECK, exempt=False, h3on=False)
    base_final = rep_base["final_value"]
    print(f"  基线 (当前生产):           期末 {base_final:,.0f}")

    # 随机门控 × N 种子
    rand_finals = []
    for seed in range(N_SEEDS):
        rep_r = run_variant(data, make_random_gated(seed), exempt=True, h3on=True)
        rand_finals.append(rep_r["final_value"])
        print(f"  随机种子 {seed:>2}: 期末 {rep_r['final_value']:>12,.0f} "
              f"({rep_r['final_value'] / base_final - 1:+.1%} vs 基线)")

    arr = np.array(rand_finals)
    pct_rank = (arr < real_final).mean() * 100
    print("\n" + "=" * 76)
    print("  判定:")
    print(f"    随机分布: 中位数 {np.median(arr):,.0f} | "
          f"P10 {np.percentile(arr, 10):,.0f} | P90 {np.percentile(arr, 90):,.0f}")
    print(f"    真实机制期末 {real_final:,.0f} 位于随机分布 {pct_rank:.0f}% 分位")
    if pct_rank >= 95:
        verdict = "✅ ret60门控有统计显著信息 (真实 > 95%随机)"
    elif pct_rank >= 90:
        verdict = "⚠️ 边缘: 真实 > 90%随机, 信息弱但存在"
    else:
        verdict = "❌ ret60门控无显著信息 (随机放行也能达到/超过, 超额来自环境)"
    print(f"    {verdict}")

    result = {
        "real_final": real_final, "base_final": base_final,
        "random_finals": [round(v, 0) for v in rand_finals],
        "random_median": round(float(np.median(arr)), 0),
        "real_percentile": round(float(pct_rank), 1),
        "verdict": verdict,
        "n_seeds": N_SEEDS, "p_pass": P_PASS,
    }
    out = OUTPUT_DIR / "overfit_permutation.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()
