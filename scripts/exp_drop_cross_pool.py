"""跨资产池验证: 门控+豁免+H3 在不同资产组合上的表现.

当前池子(8)的收益主要来自纳指+创业板 → 门控+9.3%可能是资产特定运气的产物.
新池子C: 去掉纳指/创业板/城投债, 替换为标普500/十年国债.
  商品: 518880/159985/501018/161226
  海外: 513500 (标普500, 替换513100纳指)
  债券: 511260 (十年国债, 替换511220城投债)
  防御: 511880
若新池子下门控仍+9.3% → 机制有普适性;
若失效/变差 → 原+9.3% 是特定资产组合的运气.

用法: uv run python scripts/exp_drop_cross_pool.py
输出: data/v9_results/drop_cross_pool.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
import exp_drop_gate_h3 as h3  # noqa: E402  # 含 run_v3_risk_h3
from exp_drop_gate_exempt import select_target_exempt  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
DATA_DIR = Path(rq.PROJECT_ROOT) / "data" / "cross_asset"

# 新池子C: 替换纳指/创业板为标普500/十年国债
POOL_D = {
    "513500": "标普500ETF",
    "513030": "德国ETF",
    "510900": "恒生ETF",
    "159981": "能源化工ETF",
    "511260": "十年国债ETF",
}
DEFENSE_D = "511880"
A_SHARE_ETF_D = "159915"  # 不在池中, check 返回 False
CATEGORIES_D = {
    "海外": ["513500", "513030", "510900"],
    "商品": ["159981"],
    "债券": ["511260"],
}


def load_pool_data(pool: dict, defense: str) -> dict:
    """手动加载指定池子的数据."""
    data = {}
    for code in [*list(pool.keys()), defense]:
        f = DATA_DIR / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            data[code] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def run_variant(
    data, check_fn, exempt: bool, h3on: bool, start_idx: int = 0, end_idx: int | None = None
) -> dict:
    rq.check_single_day_drop = check_fn
    h3.select_target = select_target_exempt if exempt else rq.select_target
    h3.H3_ENABLED = h3on
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data, start_idx=start_idx, end_idx=end_idx)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False
    return rep


def main() -> None:
    print("=" * 80)
    print("  跨资产池验证 | 门控+豁免+H3 在全新池子D上的表现")
    print("  池子D: 标普500/德国ETF/恒生ETF/能源化工/十年国债/货币")
    print("  (与当前8资产池零重叠)")
    print("=" * 80)

    # 保存原池子并替换为池子D
    orig_pool = dict(rq.ETF_POOL)
    orig_defense = rq.DEFENSE
    orig_a_share = rq.A_SHARE_ETF
    orig_cats = dict(rq.CATEGORIES)
    rq.ETF_POOL.clear()
    rq.ETF_POOL.update(POOL_D)
    rq.DEFENSE = DEFENSE_D
    rq.A_SHARE_ETF = A_SHARE_ETF_D
    rq.CATEGORIES.clear()
    rq.CATEGORIES.update(CATEGORIES_D)

    # 加载池子D数据
    data = load_pool_data(POOL_D, DEFENSE_D)
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)
    WARMUP = 130
    print(f"\n  公共交易日: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    from exp_v32_tail_risk import IS_START, IS_END, OOS_START, OOS_END

    OOS_END_ACTUAL = min(OOS_END, str(dates[-1]))
    segs = [
        ("全周期", 0, n),
        ("IS", *seg(IS_START, IS_END)),
        ("OOS", *seg(OOS_START, OOS_END_ACTUAL)),
    ]

    variants = [
        ("基线 (原版过滤)", h3.ORIG_CHECK, False, False),
        ("门控+豁免+H3", h3.make_gated(0.00, True), True, True),
    ]

    KEYS = ("final_value", "ann_return", "sharpe", "max_drawdown", "n_trades")
    results = {}
    for vname, check_fn, exempt, h3on in variants:
        results[vname] = {}
        for name, s0, s1 in segs:
            r = run_variant(data, check_fn, exempt, h3on, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
            results[vname][name] = {k: r[k] for k in KEYS}
        r = results[vname]["全周期"]
        is_r = results[vname]["IS"]
        oos_r = results[vname]["OOS"]
        print(f"\n  [{vname}]")
        print(
            f"    全周期 期末{r['final_value']:>12,.0f} 年化{r['ann_return'] * 100:>+6.1f}% "
            f"夏普{r['sharpe']:>5.2f} 回撤{r['max_drawdown'] * 100:>6.1f}% "
            f"交易{r['n_trades']:>4}"
        )
        print(
            f"    IS     {is_r['final_value']:>12,.0f} 年化{is_r['ann_return'] * 100:>+6.1f}% | "
            f"OOS    {oos_r['final_value']:>12,.0f} 年化{oos_r['ann_return'] * 100:>+6.1f}%"
        )

    # 判定
    base = results["基线 (原版过滤)"]
    gate = results["门控+豁免+H3"]
    checks = {
        "全周期>=基线-1%": gate["全周期"]["final_value"] >= base["全周期"]["final_value"] * 0.99,
        "IS>=基线-1%": gate["IS"]["final_value"] >= base["IS"]["final_value"] * 0.99,
        "OOS>=基线-1%": gate["OOS"]["final_value"] >= base["OOS"]["final_value"] * 0.99,
        "回撤不劣化": gate["全周期"]["max_drawdown"] >= base["全周期"]["max_drawdown"],
    }
    print("\n" + "=" * 80)
    print("  判定 (门控 vs 基线, 池子C):")
    for k, v in checks.items():
        print(f"    {k:<28} {'✅' if v else '❌'}")
    print(f"  {'✅ 全部通过' if all(checks.values()) else '❌ 未通过'}")

    results["checks"] = {k: bool(v) for k, v in checks.items()}
    out = OUTPUT_DIR / "drop_cross_pool.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {out}")

    # 恢复原池子
    rq.ETF_POOL.clear()
    rq.ETF_POOL.update(orig_pool)
    rq.DEFENSE = orig_defense
    rq.A_SHARE_ETF = orig_a_share
    rq.CATEGORIES.clear()
    rq.CATEGORIES.update(orig_cats)


if __name__ == "__main__":
    main()
