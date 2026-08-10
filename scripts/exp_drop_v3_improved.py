"""V3 改进版: 结合被甩分析 + 门控+豁免+H3 — 全量6.5年回测.

被甩分析发现:
  - 动量≤0 的被甩持仓 73% 是误杀 (后5日反弹) → 加入"动量豁免"
  - 门控放行 (ret60<0 且动量>0, 候选池平淡假摔) → 已有方案
两者正交互补: 一个管候选池, 一个管持仓被甩.

对比 4 变体:
  1. 基线 (生产原版)
  2. 门控+豁免+H3 (上次方案)
  3. 动量豁免 (新, 仅持仓被甩场景)
  4. 全叠加 (门控+豁免+H3+动量豁免)

用法: uv run python scripts/exp_drop_v3_improved.py
输出: data/v9_results/drop_v3_improved.json
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
from exp_v32_tail_risk import IS_END, IS_START, OOS_END, OOS_START, WARMUP  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"


def make_mom_exempt_check():
    """动量豁免: 触发暴跌时, 若动量≤0 则放行 (不排除).
    基于被甩分析: 动量≤0 的被甩持仓 73% 是误杀 (后5日反弹).
    """
    def check(close: np.ndarray) -> bool:
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
        if len(close) <= 11:
            return False
        r10 = (close[-1] - close[-11]) / close[-11]
        r20 = (close[-1] - close[-21]) / close[-21]
        return 0.5 * r10 + 0.5 * r20 <= 0  # 动量≤0 → 放行
    return check


def make_combined_check(ret60_gate: bool, mom_exempt: bool,
                        ret60_thr: float = 0.00, mom_thr: float = 0.00):
    """组合 check: 门控放行 + 动量豁免 (两者正交)."""
    gated_fn = h3.make_gated(ret60_thr, True) if ret60_gate else None

    def check(close: np.ndarray) -> bool:
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
            return False
        # ① 动量豁免: 动量≤0 假摔嫌疑大 → 放行
        if mom_exempt:
            r10 = (close[-1] - close[-11]) / close[-11]
            r20 = (close[-1] - close[-21]) / close[-21]
            if 0.5 * r10 + 0.5 * r20 <= 0:
                return True
        # ② 门控放行: ret60<0 且 动量>0 → 放行
        if gated_fn is not None:
            return gated_fn(close)
        return False  # 排除
    return check


def run_variant(data, check_fn, exempt: bool, h3on: bool,
               start_idx: int = 0, end_idx: int | None = None) -> dict:
    rq.check_single_day_drop = check_fn
    h3.select_target = select_target_exempt if exempt else rq.select_target
    h3.H3_ENABLED = h3on
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data, start_idx=start_idx, end_idx=end_idx)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False
    return rep


KEYS = ("final_value", "ann_return", "sharpe", "max_drawdown", "n_trades")


def main() -> None:
    print("=" * 80)
    print("  V3 改进版: 被甩分析 + 门控+豁免+H3 | 全量6.5年回测")
    print("=" * 80)

    data = rq.load_data()
    dates = sorted(set.intersection(
        *[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    segs = [("全周期", 0, n), ("IS", *seg(IS_START, IS_END)),
            ("OOS", *seg(OOS_START, OOS_END))]

    variants = [
        ("基线 (原版)", h3.ORIG_CHECK, False, False),
        ("门控+豁免+H3", h3.make_gated(0.00, True), True, True),
        ("动量豁免", make_mom_exempt_check(), False, False),
        ("全部叠加", make_combined_check(True, True, 0.00, 0.00), True, True),
    ]

    results = {}
    for vname, check_fn, exempt, h3on in variants:
        results[vname] = {}
        for name, s0, s1 in segs:
            r = run_variant(data, check_fn, exempt, h3on,
                           start_idx=s0, end_idx=max(s1 - WARMUP, 0))
            results[vname][name] = {k: r[k] for k in KEYS}
        r = results[vname]["全周期"]
        is_r = results[vname]["IS"]
        oos_r = results[vname]["OOS"]
        print(f"\n  [{vname}]")
        print(f"    全周期 期末{r['final_value']:>12,.0f} 年化{r['ann_return']*100:>+6.1f}% "
              f"夏普{r['sharpe']:>5.2f} 回撤{r['max_drawdown']*100:>6.1f}% "
              f"交易{r['n_trades']:>4}")
        print(f"    IS     {is_r['final_value']:>12,.0f} 年化{is_r['ann_return']*100:>+6.1f}% "
              f"回撤{is_r['max_drawdown']*100:>6.1f}% | "
              f"OOS    {oos_r['final_value']:>12,.0f} 年化{oos_r['ann_return']*100:>+6.1f}% "
              f"回撤{oos_r['max_drawdown']*100:>6.1f}%")

    # 判定
    base = results["基线 (原版)"]
    print("\n" + "=" * 80)
    print("  判定 (vs 基线):")
    for vname in results:
        if vname == "基线 (原版)":
            continue
        v = results[vname]
        checks = {
            "全周期>=基线-1%": v["全周期"]["final_value"] >= base["全周期"]["final_value"] * 0.99,
            "IS>=基线-1%": v["IS"]["final_value"] >= base["IS"]["final_value"] * 0.99,
            "OOS>=基线-1%": v["OOS"]["final_value"] >= base["OOS"]["final_value"] * 0.99,
            "回撤不劣化": v["全周期"]["max_drawdown"] >= base["全周期"]["max_drawdown"],
        }
        passed = all(checks.values())
        print(f"  {vname:<22} {'✅全部通过' if passed else '❌'} "
              f"({'/'.join('✓' if c else '✗' for c in checks.values())})")

    out = OUTPUT_DIR / "drop_v3_improved.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()