"""最轻版风控收益损失归因 + 波动率环境分析.

1. 逐事件审计: 每次 H1/H2 硬触发/层2退出/类别约束的机会成本
   (触发日原持仓后续5/20日收益 vs 防御收益) → 定位最大损失决策
2. 波动率环境: V3 每日按持仓 vol20 分桶 → 各桶收益/回撤贡献;
   高波动期(>0.45) 动量失效模式; 高波动+动量衰减的组合信号有效性

输出: data/v9_results/v3_loss_attribution.json
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
from exp_short_window_patterns import close_matrix  # noqa: E402
from run_qixing_v3 import ETF_POOL, load_data  # noqa: E402


def main() -> None:
    print("=" * 74)
    print("  最轻版风控收益损失归因 + 波动率环境分析")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    mat = close_matrix(data, dates)
    n = len(dates)

    # === 1. 复现最轻版 B 的事件 (import run_v3_risk) ===
    import exp_v32_tail_risk as t
    r_b = t.run_v3_risk(data)
    events = r_b["risk_events"]
    print(f"\n最轻版 B 全量期末: {r_b['final_value']:,.0f} | 风控事件: {len(events)}")

    # 事件机会成本审计: 触发日原持仓 fwd5/fwd20 vs 防御
    idx_of = {str(d): i for i, d in enumerate(dates)}
    total_opp_cost = 0.0
    events_audit = []
    for e in events:
        td_i = idx_of.get(e["date"])
        if td_i is None:
            continue
        frm = e.get("from") or e.get("target")
        if frm not in mat or frm not in ETF_POOL:
            continue
        # 原持仓后续收益
        fwd5 = mat[frm][td_i + 5] / mat[frm][td_i] - 1.0 if td_i + 5 < n and mat[frm][td_i] > 0 else 0.0
        fwd20 = mat[frm][td_i + 20] / mat[frm][td_i] - 1.0 if td_i + 20 < n and mat[frm][td_i] > 0 else 0.0
        # 防御 (货币基金 511880) 同期收益 ≈0, 债券可能正
        e["fwd5_orig"] = round(float(fwd5), 4)
        e["fwd20_orig"] = round(float(fwd20), 4)
        e["opp_cost20"] = round(float(fwd20), 4)  # 防御≈0, 机会成本≈原持仓涨幅
        total_opp_cost += float(fwd20)
        events_audit.append(e)
    print(f"\n全部风控事件累计机会成本 (触发后20日原持仓累计涨幅): {total_opp_cost:.1%}")
    worst = sorted(events_audit, key=lambda e: -e.get("opp_cost20", 0))[:6]
    print("机会成本最大的6次触发:")
    for e in worst:
        print(f"  {e['date']} {e['type']} {ETF_POOL.get(e.get('from') or e.get('target',''),'?')} "
              f"fwd5={e.get('fwd5_orig',0):+.1%} fwd20={e.get('fwd20_orig',0):+.1%}")

    # === 2. 波动率环境分析: V3 基线持仓按 vol20 分桶 ===
    print("\n" + "=" * 74)
    print("  波动率环境分析 (V3基线持仓, 按持仓资产 vol20 分桶)")
    print("=" * 74)
    from run_qixing_v3 import run_qixing_v3_same_day
    r_a = run_qixing_v3_same_day(data)
    eq = r_a["equity_curve"]
    hold = eq["holding"].tolist()
    eqv = eq["equity"].values
    eq_dates = eq["trade_date"].tolist()
    daily_ret = np.diff(eqv) / eqv[:-1]

    # 每交易日持仓资产的 vol20
    vol20_series = []
    for i, td in enumerate(eq_dates[:-1]):
        h = hold[i]
        if h not in ETF_POOL:
            vol20_series.append(0.05)  # 防御≈0波动
            continue
        idx = idx_of.get(str(td.date()))
        if idx is None or idx < 20:
            vol20_series.append(0.3)
            continue
        seg = mat[h][idx - 19:idx + 1].astype(float)
        if np.all(np.isfinite(seg)) and len(seg) > 1:
            dr = np.diff(seg) / seg[:-1]
            vol20_series.append(float(np.std(dr) * np.sqrt(252)))
        else:
            vol20_series.append(0.3)
    vol20_arr = np.array(vol20_series)
    ret_arr = daily_ret

    buckets = [(0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 10.0)]
    print(f"  {'vol20桶':<16} {'天数':>6} {'日均收益':>10} {'年化贡献':>10} {'深回撤期':>8}")
    for lo, hi in buckets:
        mask = (vol20_arr >= lo) & (vol20_arr < hi)
        if mask.sum() == 0:
            continue
        mean_d = ret_arr[mask].mean()
        # 年化贡献 = 日均 × 252 × 天数占比
        ann_contrib = mean_d * 252 * mask.mean()
        deep = (vol20_arr[mask] > 0).sum()  # 占位
        print(f"  [{lo:.2f}, {hi:.2f})          {mask.sum():>6} {mean_d:>+10.4%} "
              f"{ann_contrib:>+10.1%}")

    # 高波动期动量失效: vol20>0.45 时, 持仓日均收益 vs 低波动
    hv = vol20_arr > 0.45
    lv = vol20_arr <= 0.45
    print(f"\n  高波动 (vol20>0.45) 持仓日均收益: {ret_arr[hv].mean():+.4%} "
          f"({hv.sum()}天, 占{len(ret_arr):.0%}天)")
    print(f"  低波动 (vol20≤0.45) 持仓日均收益: {ret_arr[lv].mean():+.4%} ({lv.sum()}天)")

    # 高波动+高动量 (追高) 的后续表现: 持仓 vol20>0.5 且 mom10>0.15
    from run_qixing_v3 import load_data as _ld
    hv_mom = 0.0
    hv_mom_n = 0
    for i, td in enumerate(eq_dates[:-1]):
        h = hold[i]
        if h not in ETF_POOL:
            continue
        idx = idx_of.get(str(td.date()))
        if idx is None or idx < 20:
            continue
        seg = mat[h][idx - 19:idx + 1].astype(float)
        if not np.all(np.isfinite(seg)):
            continue
        dr = np.diff(seg) / seg[:-1]
        vol = np.std(dr) * np.sqrt(252)
        mom10 = mat[h][idx] / mat[h][idx - 10] - 1.0 if mat[h][idx - 10] > 0 else 0.0
        if vol > 0.5 and mom10 > 0.15:
            hv_mom += ret_arr[i]
            hv_mom_n += 1
    if hv_mom_n:
        print(f"  高波动(vol>0.5)+高动量(mom10>0.15) 追高持仓: {hv_mom_n}天, "
              f"日均 {hv_mom/hv_mom_n:+.4%}")

    out = {"events_audit": events_audit,
           "total_opp_cost": round(float(total_opp_cost), 4),
           "vol_buckets": {"hv_days": int(hv.sum()), "lv_days": int(lv.sum()),
                           "hv_mean_daily": round(float(ret_arr[hv].mean()), 6),
                           "lv_mean_daily": round(float(ret_arr[lv].mean()), 6)}}
    with open(OUTPUT_DIR / "v3_loss_attribution.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  已保存: {OUTPUT_DIR}/v3_loss_attribution.json")


if __name__ == "__main__":
    main()
