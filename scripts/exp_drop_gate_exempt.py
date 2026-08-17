"""门控 + 换仓缓冲豁免 — A/B 验证.

背景: 归因发现门控在2022-10-31的亏损根因不是"放行纳指"本身, 而是:
      - 原油(最强+6.69%) vs 纳指(持仓+5.55%) 差距仅1.13pp
      - 被 0.05 自适应换仓缓冲吃掉 → 继续持纳指 → 11月初纳指阴跌
      - 基线靠暴跌过滤强制排除纳指 → 换原油 → 躲过
缓冲豁免: 放行类型持仓 (ret60<0 且近5日单日跌>3%) 不享 0.05 缓冲,
          必须保持候选第一名才保留, 否则立即换最强.

对比: 基线 / 门控+H3(当前最优) / 门控+豁免 / 门控+豁免+H3
验证: 全周期 + IS/OOS; 目标: 豁免让IS段恢复接近基线, 且OOS保持改善
用法: uv run python scripts/exp_drop_gate_exempt.py
输出: data/v9_results/drop_gate_exempt.json
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
import exp_drop_gate_h3 as h3mod  # noqa: E402
import exp_v32_tail_risk as ev32  # noqa: E402
from exp_v32_tail_risk import (  # noqa: E402
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    OUTPUT_DIR,
    WARMUP,
)

ORIG_CHECK = rq.check_single_day_drop
ORIG_SELECT = rq.select_target


def select_target_exempt(data: dict, etf_data_at_date: dict, holding: str | None):
    """复制 run_qixing_v3.select_target + 缓冲豁免.

    放行类型持仓 (ret60<0 且 近5日单日跌>3%) 不享 0.05 换仓缓冲:
    必须保持候选第一名才保留, 否则换最强 (防止下跌趋势反弹品种白嫖续命).
    """
    a_share_weak = (
        rq.check_a_share_weak(data, etf_data_at_date.get(rq.A_SHARE_ETF, 0))
        if rq.USE_A_SHARE_FILTER
        else False
    )
    candidates = []
    for code in rq.ETF_POOL:
        if code not in etf_data_at_date:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = etf_data_at_date[code]
        df = data[code]
        close = df["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_SHORT_MOM_FILTER and not rq.check_short_momentum(close):
            continue
        if rq.USE_VOL_SPIKE_FILTER and not rq.check_volume_spike(
            df["volume"].values[: idx + 1].astype(float), close
        ):
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        if (
            rq.USE_LONG_MOM_FILTER
            and code in ("513100", "159915")
            and len(close) > rq.LONG_MOM_PERIOD
        ):
            long_mom = (close[-1] - close[-rq.LONG_MOM_PERIOD - 1]) / close[-rq.LONG_MOM_PERIOD - 1]
            if long_mom < 0:
                continue
        score = rq.calc_momentum_score(close)
        if score > 0:
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])
    if rq.USE_CATEGORY_SWITCH and candidates:
        score_map = dict(candidates)
        cat_scores = {}
        for cat_name, cat_codes in rq.CATEGORIES.items():
            cat_moms = [score_map[c] for c in cat_codes if c in score_map]
            if cat_moms:
                cat_scores[cat_name] = np.mean(cat_moms)
        if cat_scores:
            best_cat = max(cat_scores, key=lambda k: cat_scores[k])
            best_cat_codes = set(rq.CATEGORIES[best_cat])
            cat_candidates = [(c, s) for c, s in candidates if c in best_cat_codes]
            if cat_candidates:
                candidates = cat_candidates
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0

    threshold = 0.0 if best_score > 0.10 else 0.05
    # === 缓冲豁免: 放行类型持仓 (ret60<0 且近5日暴跌) 不享缓冲 ===
    if holding and holding in etf_data_at_date and holding != rq.DEFENSE:
        hclose = data[holding]["close"].values[: etf_data_at_date[holding] + 1].astype(float)
        if len(hclose) > rq.DROP_LOOKBACK + 1 and len(hclose) > 61:
            drop = any(
                (hclose[i] - hclose[i - 1]) / hclose[i - 1] < rq.DROP_THRESHOLD
                for i in range(-rq.DROP_LOOKBACK, 0)
            )
            ret60 = (hclose[-1] - hclose[-61]) / hclose[-61]
            if drop and ret60 < 0.0:
                threshold = 0.0

    if holding and holding != rq.DEFENSE:
        cur_score = dict(candidates).get(holding, -999)
        if cur_score > 0:
            target = best_target if best_score > cur_score + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target, candidates, best_score, a_share_weak


def run_pair(
    data, dates, s0, s1, use_gate: bool, use_exempt: bool, use_h3: bool, cost_mult: float = 1.0
) -> tuple[dict, dict]:
    """A/B: A=基线, B=门控[+豁免][+H3]."""
    # A: 基线
    rq.check_single_day_drop = ORIG_CHECK
    h3mod.select_target = ORIG_SELECT
    h3mod.H3_ENABLED = False
    r_a = h3mod.run_v3_risk_h3(
        data, start_idx=s0, end_idx=max(s1 - WARMUP, 0), cost_multiplier=cost_mult
    )
    # B: 门控[+豁免][+H3]
    rq.check_single_day_drop = h3mod.make_gated(0.00, True) if use_gate else ORIG_CHECK
    # 注意: run_v3_risk_h3 内部调用的是 exp_drop_gate_h3 模块绑定的 select_target
    h3mod.select_target = select_target_exempt if use_exempt else ORIG_SELECT
    h3mod.H3_ENABLED = use_h3
    h3mod.H3_DELTA, h3mod.H3_ACTION, h3mod.H3_EXPO = 0.02, "reduce", 0.3
    r_b = h3mod.run_v3_risk_h3(
        data, start_idx=s0, end_idx=max(s1 - WARMUP, 0), cost_multiplier=cost_mult
    )
    rq.check_single_day_drop = ORIG_CHECK
    h3mod.select_target = ORIG_SELECT
    h3mod.H3_ENABLED = False
    return r_a, r_b


KEYS = ("final_value", "ann_return", "sharpe", "max_drawdown", "n_trades")


def main() -> None:
    print("=" * 78)
    print("  门控 + 换仓缓冲豁免 A/B | 全周期 + IS/OOS")
    print("=" * 78)

    data = rq.load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    segs = [("全周期", 0, n), ("IS", *seg(IS_START, IS_END)), ("OOS", *seg(OOS_START, OOS_END))]

    variants = [
        ("基线", False, False, False),
        ("门控+H3 (当前最优)", True, False, True),
        ("门控+豁免 (无H3)", True, True, False),
        ("门控+豁免+H3", True, True, True),
    ]

    results: dict = {"variants": {}}
    for vname, gate, exempt, h3on in variants:
        seg_results = {}
        for name, s0, s1 in segs:
            _, r_b = run_pair(data, dates, s0, s1, use_gate=gate, use_exempt=exempt, use_h3=h3on)
            seg_results[name] = {k: r_b[k] for k in KEYS}
        results["variants"][vname] = seg_results
        row = seg_results["全周期"]
        is_row, oos_row = seg_results["IS"], seg_results["OOS"]
        print(f"\n  [{vname}]")
        print(
            f"    全周期 {row['final_value']:>12,.0f} 年化{row['ann_return'] * 100:>+6.1f}% "
            f"夏普{row['sharpe']:>5.2f} 回撤{row['max_drawdown'] * 100:>6.1f}%"
        )
        print(
            f"    IS     {is_row['final_value']:>12,.0f} 年化{is_row['ann_return'] * 100:>+6.1f}% | "
            f"OOS    {oos_row['final_value']:>12,.0f} 年化{oos_row['ann_return'] * 100:>+6.1f}%"
        )

    # 判定 (各变体 vs 基线)
    base = results["variants"]["基线"]
    print("\n" + "=" * 78)
    print("  判定 (变体 vs 基线):")
    verdict = {}
    for vname in results["variants"]:
        if vname == "基线":
            continue
        v = results["variants"][vname]
        checks = {
            "全周期>=基线-1%": v["全周期"]["final_value"] >= base["全周期"]["final_value"] * 0.99,
            "IS>=基线-1%": v["IS"]["final_value"] >= base["IS"]["final_value"] * 0.99,
            "OOS>=基线-1%": v["OOS"]["final_value"] >= base["OOS"]["final_value"] * 0.99,
            "回撤不劣化": v["全周期"]["max_drawdown"] >= base["全周期"]["max_drawdown"],
        }
        verdict[vname] = {k: bool(vv) for k, vv in checks.items()}
        mark = "✅ 全部通过" if all(checks.values()) else "❌"
        print(f"  {vname:<22} {mark} ({'/'.join('✓' if c else '✗' for c in checks.values())})")
    results["verdict"] = verdict

    out = OUTPUT_DIR / "drop_gate_exempt.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
