"""门控参数全网格扫描: ret60_thr × drop_threshold × drop_lookback.

固定: 豁免 on, H3(δ=2% expo=0.3), mom_thr=0.00
目的: 找到分歧次数(有效样本量)与表现的关系; 验证"放宽后单调恶化"是否成立.
用法: uv run python scripts/exp_gate_full_grid.py
输出: data/v9_results/gate_full_grid.json
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
ORIG_SELECT = rq.select_target
DECISIONS: list[tuple] = []


def make_gated_par(ret60_thr: float, mom_thr: float):
    """门控放行: ret60>=thr 排除; 动量>mom_thr 才放行."""
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
        r10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
        r20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
        return 0.5 * r10 + 0.5 * r20 > mom_thr
    return gated


def run_with_params(data, ret60_thr, drop_thr, drop_lookback) -> dict:
    """组合参数跑全周期, 返回指标 + 分歧次数."""
    global DECISIONS
    DECISIONS = []

    # patch rq 模块 (使 select_target 和 run_v3_risk_h3 内部生效)
    orig_dt = rq.DROP_THRESHOLD
    orig_dl = rq.DROP_LOOKBACK
    rq.DROP_THRESHOLD = -abs(drop_thr)
    rq.DROP_LOOKBACK = drop_lookback

    gate_fn = make_gated_par(ret60_thr, 0.00)

    # wrapped select_target 记录决策日志
    base_select = select_target_exempt

    def wrapped(data_, etf_data_at_date, holding):
        global DECISIONS
        t, c, s, a = base_select(data_, etf_data_at_date, holding)
        td = None
        for code in rq.ETF_POOL:
            if code in etf_data_at_date:
                td = data_[code].iloc[etf_data_at_date[code]]["trade_date"]
                break
        DECISIONS.append((str(td), holding, t))
        return t, c, s, a

    rq.check_single_day_drop = gate_fn
    h3.select_target = wrapped
    h3.H3_ENABLED = True
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data)
    log = list(DECISIONS)

    # 恢复
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False
    rq.DROP_THRESHOLD = orig_dt
    rq.DROP_LOOKBACK = orig_dl

    # 基线决策日志 (原版过滤, 无豁免无H3)
    DECISIONS2 = []
    rq.check_single_day_drop = h3.ORIG_CHECK
    def wrapped2(data_, etf_data_at_date, holding):
        t, c, s, a = rq.select_target(data_, etf_data_at_date, holding)
        td = None
        for code in rq.ETF_POOL:
            if code in etf_data_at_date:
                td = data_[code].iloc[etf_data_at_date[code]]["trade_date"]
                break
        DECISIONS2.append((str(td), holding, t))
        return t, c, s, a
    h3.select_target = wrapped2
    h3.H3_ENABLED = False
    rep_base = h3.run_v3_risk_h3(data)
    h3.select_target = rq.select_target
    rq.check_single_day_drop = h3.ORIG_CHECK

    # 分歧次数
    amap = {d: t for d, h, t in DECISIONS2}
    div = sum(1 for d, h, t in log if amap.get(d) is not None and amap[d] != t)

    return {
        "final": rep["final_value"], "base_final": rep_base["final_value"],
        "sharpe": rep["sharpe"], "max_dd": rep["max_drawdown"],
        "n_trades": rep["n_trades"], "divergences": div,
        "base_final": rep_base["final_value"],
        "base_sharpe": rep_base["sharpe"],
    }


def main() -> None:
    print("=" * 80)
    print("  门控参数全网格 | ret60_thr × drop_threshold × drop_lookback")
    print("=" * 80)

    data = rq.load_data()
    results = []
    for ret60_thr in (0.00, 0.05, 0.10):
        for drop_thr in (0.02, 0.03, 0.05):  # 正值, 内部取负
            for dl in (3, 5, 7):
                r = run_with_params(data, ret60_thr, drop_thr, dl)
                diff = r["final"] / r["base_final"] - 1
                results.append({
                    "ret60_thr": ret60_thr, "drop_threshold": drop_thr,
                    "drop_lookback": dl, "final": r["final"],
                    "diff_vs_base": diff, "sharpe": r["sharpe"],
                    "max_dd": r["max_dd"], "divergences": r["divergences"],
                    "n_trades": r["n_trades"],
                    "base_final": r["base_final"],
                })
                print(f"  thr={ret60_thr:.2f} drop={drop_thr:.0%} lookback={dl}: "
                      f"期末{r['final']:>10,.0f} ({diff:+.1%}) "
                      f"夏普{r['sharpe']:.2f} 分歧{r['divergences']:>3}次")

    # 分析
    print("\n" + "=" * 80)
    print("  分歧次数 vs 表现:")
    for r in sorted(results, key=lambda x: -x["divergences"]):
        print(f"    分歧{r['divergences']:>3}次 thr={r['ret60_thr']:.2f} "
              f"drop={r['drop_threshold']:.0%} lb={r['drop_lookback']}: "
              f"{r['diff_vs_base']:+.1%}")

    # 找"分歧≥5 且 表现≥基线"的参数区
    robust = [r for r in results if r["divergences"] >= 5 and r["diff_vs_base"] >= 0]
    print(f"\n  分歧≥5 且 ≥基线: {len(robust)} 个")
    for r in robust:
        print(f"    thr={r['ret60_thr']:.2f} drop={r['drop_threshold']:.0%} "
              f"lb={r['drop_lookback']}: 分歧{r['divergences']}次 {r['diff_vs_base']:+.1%}")

    # 当前参数附近结果
    print(f"\n  当前参数 (thr=0.00 drop=3% lb=5): "
          f"{[r for r in results if r['ret60_thr']==0.00 and r['drop_threshold']==0.03 and r['drop_lookback']==5]}")

    out = OUTPUT_DIR / "gate_full_grid.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {out}")


if __name__ == "__main__":
    main()