"""ret60_thr 密集扫描: 0.00~0.10 步长0.01, 固定 drop=3% lb=5.

分析: 分歧次数与表现的关系, 找"分歧多且收益高"的阈值区间.
用法: uv run python scripts/exp_gate_thr_scan.py
输出: data/v9_results/gate_thr_scan.json
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

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
DECISIONS: list[tuple] = []


def run_with_thr(data, thr):
    global DECISIONS
    DECISIONS = []

    def gated(close):
        if len(close) < rq.DROP_LOOKBACK + 1:
            return True
        triggered = any(
            (close[i] - close[i - 1]) / close[i - 1] < rq.DROP_THRESHOLD
            for i in range(-rq.DROP_LOOKBACK, 0))
        if not triggered:
            return True
        if len(close) <= 61:
            return True
        ret60 = (close[-1] - close[-61]) / close[-61]
        if ret60 >= thr:
            return False
        r10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
        r20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
        return 0.5 * r10 + 0.5 * r20 > 0.0

    def wrapped(data_, etf_data_at_date, holding):
        global DECISIONS
        t, c, s, a = select_target_exempt(data_, etf_data_at_date, holding)
        td = None
        for code in rq.ETF_POOL:
            if code in etf_data_at_date:
                td = data_[code].iloc[etf_data_at_date[code]]["trade_date"]
                break
        DECISIONS.append((str(td), holding, t))
        return t, c, s, a

    rq.check_single_day_drop = gated
    h3.select_target = wrapped
    h3.H3_ENABLED = True
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data)
    log = list(DECISIONS)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False

    # 基线日志
    DECISIONS2 = []
    def wrapped2(d2, em, h):
        t, c, s, a = rq.select_target(d2, em, h)
        td = None
        for code in rq.ETF_POOL:
            if code in em:
                td = d2[code].iloc[em[code]]["trade_date"]
                break
        DECISIONS2.append((str(td), h, t))
        return t, c, s, a
    h3.select_target = wrapped2
    h3.H3_ENABLED = False
    rep_base = h3.run_v3_risk_h3(data)
    h3.select_target = rq.select_target
    rq.check_single_day_drop = h3.ORIG_CHECK

    amap = {d: t for d, h, t in DECISIONS2}
    div = sum(1 for d, h, t in log if amap.get(d) is not None and amap[d] != t)
    return {"final": rep["final_value"], "base": rep_base["final_value"],
            "sharpe": rep["sharpe"], "max_dd": rep["max_drawdown"],
            "divergences": div, "n_trades": rep["n_trades"]}


def main():
    print("=" * 80)
    print("  ret60_thr 密集扫描 | 0.00~0.10 步长0.01 | 固定drop=3% lb=5")
    print("=" * 80)

    data = rq.load_data()
    results = []
    for thr in [round(x * 0.01, 2) for x in range(0, 11)]:
        r = run_with_thr(data, thr)
        diff = r["final"] / r["base"] - 1
        results.append({"thr": thr, "final": r["final"], "base": r["base"],
                        "diff": diff, "sharpe": r["sharpe"],
                        "max_dd": r["max_dd"], "divergences": r["divergences"],
                        "n_trades": r["n_trades"]})
        print(f"  thr={thr:+.2f}: 期末{r['final']:>10,.0f} ({diff:+.1%}) "
              f"夏普{r['sharpe']:.2f} 回撤{r['max_dd']:.1%} 分歧{r['divergences']:>3}次")

    print("\n" + "=" * 80)
    print("  分歧次数 vs 表现:")
    for r in sorted(results, key=lambda x: -x["divergences"]):
        print(f"    thr={r['thr']:+.2f} 分歧{r['divergences']:>3}次 {r['diff']:+.1%}")

    # 找"分歧≥5且≥基线"的区间
    ok = [r for r in results if r["divergences"] >= 5 and r["diff"] >= 0]
    if ok:
        thr_min = min(r["thr"] for r in ok)
        thr_max = max(r["thr"] for r in ok)
        print(f"\n  分歧≥5 且 ≥基线的阈值区间: [{thr_min:.2f}, {thr_max:.2f}] "
              f"({len(ok)}个点)")
        for r in ok:
            print(f"    thr={r['thr']:+.2f}: 分歧{r['divergences']}次 {r['diff']:+.1%}")
    else:
        print("\n  ❌ 无分歧≥5且≥基线的阈值")

    out = OUTPUT_DIR / "gate_thr_scan.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()