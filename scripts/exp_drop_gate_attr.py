"""门控版暴跌过滤 · IS段决策归因分析.

回答: 门控版在IS段(2020-06~2023-12)比基线少赚 3.6万(-12.1%) 的根源?
      对比基线/门控在每个调仓日的选股决策, 定位分歧点及其后续影响;
      并用净值曲线对比定位差距扩大的时段.
方法: 包装 select_target 记录决策日志 (注意 run_v3_risk 调用的是
      exp_v32_tail_risk 命名空间的 select_target, 须 patch ev32 模块),
      门控过滤 patch run_qixing_v3.check_single_day_drop (select_target
      函数体自由变量解析到 rq 模块).
用法: uv run python scripts/exp_drop_gate_attr.py
输出: data/v9_results/drop_gate_attr.json
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
import exp_v32_tail_risk as ev32  # noqa: E402
from exp_v32_tail_risk import IS_END, IS_START, OUTPUT_DIR, WARMUP, run_v3_risk

ORIG_CHECK = rq.check_single_day_drop
ORIG_SELECT = rq.select_target
DECISIONS: list[dict] = []


def make_gated(ret60_thr: float = 0.0, mom_guard: bool = True):
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
        ret60 = (close[-1] - close[-61]) / close[-61]
        if ret60 >= ret60_thr:
            return False
        if mom_guard:
            ret10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
            ret20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
            if 0.5 * ret10 + 0.5 * ret20 <= 0:
                return False
        return True
    return gated


def wrapped_select(data, etf_data_at_date, holding):
    """记录每次调仓的目标/候选排名 (td 从 idx_map 反推)."""
    t, c, s, a = ORIG_SELECT(data, etf_data_at_date, holding)
    td = None
    for code in rq.ETF_POOL:
        if code in etf_data_at_date:
            df = data[code]
            td = df.iloc[etf_data_at_date[code]]["trade_date"]
            break
    DECISIONS.append({
        "date": str(td) if td is not None else "?",
        "holding": holding, "target": t,
        "top3": [x[0] for x in c[:3]],
        "n_cand": len(c),
    })
    return t, c, s, a


def run_is(gated: bool) -> tuple[dict, list[dict]]:
    """跑 IS 段, 返回 (report, 决策日志)."""
    global DECISIONS
    DECISIONS = []
    rq.check_single_day_drop = make_gated(0.00, True) if gated else ORIG_CHECK
    ev32.select_target = wrapped_select
    data = rq.load_data()
    dates = sorted(set.intersection(
        *[set(data[c]["trade_date"]) for c in list(data.keys())]))
    a = next(i for i, d in enumerate(dates) if str(d) >= IS_START)
    b = next(i for i, d in enumerate(dates) if str(d) >= IS_END)
    report = run_v3_risk(data, start_idx=max(a - WARMUP, 0),
                         end_idx=max(min(b, len(dates)) - WARMUP, 0))
    log = DECISIONS
    rq.check_single_day_drop = ORIG_CHECK
    ev32.select_target = ORIG_SELECT
    return report, log


def diff_equity_curve(eq_base, eq_gate, label: str) -> None:
    """对比两条净值曲线, 定位差距扩大的时段 (月末采样)."""
    import pandas as pd
    eb = pd.DataFrame(eq_base).copy()
    eg = pd.DataFrame(eq_gate).copy()
    eb["trade_date"] = pd.to_datetime(eb["trade_date"])
    eg["trade_date"] = pd.to_datetime(eg["trade_date"])
    eb = eb.set_index("trade_date").resample("ME").last()
    eg = eg.set_index("trade_date").resample("ME").last()
    merged = eb["equity"].rename("base").to_frame().join(
        eg["equity"].rename("gate"), how="outer")
    print(f"\n  【{label} 净值对比 (月末)】")
    prev_gap = 0.0
    for d, row in merged.iterrows():
        b, g = row.get("base"), row.get("gate")
        if b is None or g is None or b <= 0:
            continue
        gap = (g - b) / b
        marker = ""
        if abs(gap - prev_gap) > 0.03:
            marker = f"  <- 差距变化 {prev_gap:+.1%} -> {gap:+.1%}"
        print(f"    {d.strftime('%Y-%m')}  基线{b:>11,.0f} 门控{g:>11,.0f} "
              f"差 {gap:+.1%}{marker}")
        prev_gap = gap


def main() -> None:
    print("=" * 76)
    print("  门控版暴跌过滤 · IS段决策归因 (基线 vs 门控 thr=0.00 guard=on)")
    print("=" * 76)

    rep_base, log_base = run_is(gated=False)
    rep_gate, log_gate = run_is(gated=True)
    print(f"\n  基线 IS 期末 {rep_base['final_value']:>12,.0f} | "
          f"门控 IS 期末 {rep_gate['final_value']:>12,.0f} "
          f"({rep_gate['final_value'] / rep_base['final_value'] - 1:+.2%})")

    # 对齐决策: 按 date 对齐 (holding 可能已分叉, 不能作为匹配键)
    bmap = {d["date"]: d for d in log_base}
    div_points = []
    for dg in log_gate:
        db = bmap.get(dg["date"])
        if db is None:
            continue
        if db["target"] != dg["target"] or db["holding"] != dg["holding"]:
            div_points.append({
                "date": dg["date"], "holding": dg["holding"],
                "base_target": db["target"], "gate_target": dg["target"],
                "base_top3": db["top3"], "gate_top3": dg["top3"],
                "base_n": db["n_cand"], "gate_n": dg["n_cand"],
            })

    print(f"\n  决策分歧点: {len(div_points)} 个 (基线/门控 持仓或目标不同)")
    for p in div_points[:40]:
        print(f"    {p['date']} 持{p['holding']}: 基线→{p['base_target']} "
              f"(候选{p['base_top3']}) | 门控→{p['gate_target']} (候选{p['gate_top3']})")

    # 分歧点后20交易日收益差 (近似分歧影响)
    data = rq.load_data()
    common = sorted(set.intersection(
        *[set(data[c]["trade_date"]) for c in list(data.keys())]))
    ci = {d: i for i, d in enumerate(common)}

    def fwd(code: str, td, n: int) -> float | None:
        if code not in data:
            return None
        from datetime import date as _date
        td = _date.fromisoformat(td) if isinstance(td, str) else td
        df = data[code]
        row = df[df["trade_date"] == td]
        if row.empty:
            return None
        i = ci.get(td)
        if i is None or i + n >= len(common):
            return None
        fut = common[i + n]
        frow = df[df["trade_date"] == fut]
        if frow.empty:
            return None
        return float(frow.iloc[0]["close"]) / float(row.iloc[0]["close"]) - 1.0

    impacts = []
    for p in div_points:
        f_b = fwd(p["base_target"], p["date"], 20) if p["base_target"] else None
        f_g = fwd(p["gate_target"], p["date"], 20) if p["gate_target"] else None
        diff = None
        if f_b is not None and f_g is not None:
            diff = f_g - f_b
        impacts.append({**p, "fwd20_base": f_b, "fwd20_gate": f_g, "impact": diff})

    print("\n  分歧点 后20交易日 目标收益差 (门控-基线):")
    total_diff = 0.0
    for p in impacts:
        if p["impact"] is not None:
            total_diff += p["impact"]
            print(f"    {p['date']} 基{p['base_target']}({p['fwd20_base']:+.1%}) "
                  f"vs 门{p['gate_target']}({p['fwd20_gate']:+.1%}) "
                  f"差 {p['impact']:+.1%}")
    print(f"\n  分歧影响合计 (近似): {total_diff:+.1%}")

    diff_equity_curve(rep_base["equity_curve"], rep_gate["equity_curve"], "IS段")

    result = {
        "base_final": rep_base["final_value"], "gate_final": rep_gate["final_value"],
        "n_divergence": len(div_points), "divergence": impacts,
    }
    out = OUTPUT_DIR / "drop_gate_attr.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()
