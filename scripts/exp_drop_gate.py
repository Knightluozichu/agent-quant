"""门控版暴跌过滤 A/B 回测 (基线 vs 门控, 叠加 V32 风控层).

背景: 生产 DROP_FILTER "近5日单日跌>3% → 排除候选" 不分真假跌;
      用户场景"震荡上涨中-6%大阴线"会被误杀 (诊断: 大跌后5日 47%真跌/53%假摔,
      ret60<0 平淡组继续跌率 42%, ret60>=0 上涨组 51%, 区分度 9pp).
门控版: 触发暴跌后检查趋势位置 ret60:
        - ret60 >= 阈值 (前期上涨) → 照旧排除 (真跌保护)
        - ret60 < 阈值 且 动量>0 (平淡+动量未转负) → 放行 (假摔不误杀)
        - 放行后由 V32 风控层兜底 (H1/H2降仓0.7 / 高波动衰减降仓 / -30%熔断)

验证 (项目规范): 全周期 + IS/OOS + ret60阈值/动量守卫扰动 + 成本2x/3x
用法: uv run python scripts/exp_drop_gate.py
输出: data/v9_results/drop_gate_ab.json
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
from exp_short_window_patterns import close_matrix  # noqa: E402
from exp_v32_tail_risk import (  # noqa: E402
    IS_END, IS_START, OOS_END, OOS_START, OUTPUT_DIR, WARMUP, run_v3_risk,
)

ORIG_CHECK = rq.check_single_day_drop
GATE_STATS: dict = {"passed": 0, "excluded": 0}


def make_gated(ret60_thr: float = 0.0, mom_guard: bool = True):
    """构造门控版 check_single_day_drop(close) -> bool (True=通过不过滤).

    Args:
        ret60_thr: 趋势位置阈值; ret60>=阈值视为前期上涨(排除), <阈值视为平淡(可放行)
        mom_guard: True 时要求生产动量评分>0 才放行 (平淡但动量转负仍排除)
    """
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
        # 触发暴跌 → 趋势位置门控 (历史不足61日按放行, 与生产 len 检查一致)
        if len(close) <= 61:
            return True
        ret60 = (close[-1] - close[-61]) / close[-61]
        if ret60 >= ret60_thr:
            GATE_STATS["excluded"] += 1
            return False
        if mom_guard:
            ret10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
            ret20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
            if 0.5 * ret10 + 0.5 * ret20 <= 0:
                GATE_STATS["excluded"] += 1
                return False
        GATE_STATS["passed"] += 1
        return True
    return gated


KEYS = ("final_value", "total_return", "ann_return", "sharpe",
        "max_drawdown", "calmar", "n_trades")


def run_segment(name: str, s0: int, s1: int) -> dict:
    """跑一段 (对齐 exp_v32_tail_risk 段索引平移)."""
    r = run_v3_risk(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
    out = {k: r[k] for k in KEYS}
    out["n_risk_events"] = r.get("n_risk_events", 0)
    out["deep_dd_time"] = r.get("deep_dd_time", 0.0)
    out["cvar95"] = r.get("cvar95", 0.0)
    return out


def main() -> None:
    global data
    print("=" * 76)
    print("  门控版暴跌过滤 A/B | 基线(原版) vs 门控(ret60+动量守卫) | V32风控叠加")
    print("=" * 76)

    data = rq.load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    segs = [("全周期", 0, n), ("IS", *seg(IS_START, IS_END)),
            ("OOS", *seg(OOS_START, OOS_END))]

    # 变体: (名称, ret60阈值, 动量守卫)
    variants = [
        ("基线(原版)", None, None),
        ("门控 thr=0.00 guard=on", 0.00, True),
        ("门控 thr=0.00 guard=off", 0.00, False),
        ("门控 thr=0.10 guard=on", 0.10, True),
    ]

    results: dict = {"variants": {}}
    for vname, thr, guard in variants:
        rq.check_single_day_drop = (
            make_gated(thr, guard) if thr is not None else ORIG_CHECK)
        GATE_STATS["passed"] = GATE_STATS["excluded"] = 0
        seg_results = {}
        for name, s0, s1 in segs:
            seg_results[name] = run_segment(name, s0, s1)
        results["variants"][vname] = {"segs": seg_results,
                                      "gate_stats": dict(GATE_STATS)}
        print(f"\n  [{vname}]  (放行{results['variants'][vname]['gate_stats'].get('passed', 0)}次 / "
              f"排除{results['variants'][vname]['gate_stats'].get('excluded', 0)}次)")
        for name in ("全周期", "IS", "OOS"):
            r = seg_results[name]
            print(f"    {name:<6} 期末{r['final_value']:>12,.0f} "
                  f"年化{r['ann_return'] * 100:>+7.1f}% 夏普{r['sharpe']:>6.2f} "
                  f"回撤{r['max_drawdown'] * 100:>7.1f}% 交易{r['n_trades']:>4}")
    rq.check_single_day_drop = ORIG_CHECK  # 恢复

    # === 成本压力 (推荐变体 thr=0.00 guard=on) ===
    print("\n" + "=" * 76)
    print("  成本压力测试 (推荐变体: 门控 thr=0.00 guard=on, 全周期):")
    base = results["variants"]["基线(原版)"]["segs"]["全周期"]
    cost_rows = {}
    for mult in (1.0, 2.0, 3.0):
        rq.check_single_day_drop = make_gated(0.00, True)
        r = run_v3_risk(data, cost_multiplier=mult)
        rq.check_single_day_drop = ORIG_CHECK
        cost_rows[f"{mult:g}x"] = {k: r[k] for k in KEYS}
        ratio = r["final_value"] / base["final_value"] if base["final_value"] else 0
        print(f"    {mult:g}x 期末{r['final_value']:>12,.0f} "
              f"(基线比 {ratio - 1:+.2%}) 夏普{r['sharpe']:>6.2f} "
              f"回撤{r['max_drawdown'] * 100:>7.1f}%")
    results["cost_stress"] = {"base_final": base["final_value"], "rows": cost_rows}

    # === 判定 ===
    b = results["variants"]["基线(原版)"]["segs"]
    g = results["variants"]["门控 thr=0.00 guard=on"]["segs"]
    checks = {
        "全周期收益不劣化(>=基线-1%)": g["全周期"]["final_value"] >= b["全周期"]["final_value"] * 0.99,
        "全周期回撤不劣化": g["全周期"]["max_drawdown"] >= b["全周期"]["max_drawdown"],
        "OOS收益不劣化(>=基线-1%)": g["OOS"]["final_value"] >= b["OOS"]["final_value"] * 0.99,
        "OOS夏普不劣化0.15": g["OOS"]["sharpe"] >= b["OOS"]["sharpe"] - 0.15,
        "成本3x仍有正收益": cost_rows.get("3x", {}).get("total_return", -1) > 0,
    }
    print("\n" + "=" * 76)
    print("  判定 (推荐变体 vs 基线):")
    for k, v in checks.items():
        print(f"    {k:<28} {'✅' if v else '❌'}")

    results["checks"] = {k: bool(v) for k, v in checks.items()}
    out_path = OUTPUT_DIR / "drop_gate_ab.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
