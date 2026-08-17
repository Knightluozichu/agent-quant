"""门控版暴跌过滤 · 放行事件追踪分析.

回答: 门控放行的18次事件发生在哪些时段? 放行后的实际收益(假摔赚/真跌亏)?
      被排除的品种后续收益(排除保护价值)? 用于决定最终门控形态.

口径: 与 run_v3_risk 一致 (调仓日网格 = trading_dates[WARMUP:][::5]),
      逐品种检查"近5日单日跌>3%"触发 → ret60 门控判定 (thr=0.00, guard=on),
      统计后5日收益 (放行/排除两种命运的收益差).
用法: uv run python scripts/exp_drop_gate_trace.py
输出: data/v9_results/drop_gate_trace.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402

WARMUP = 130
RET60_THR = 0.00
HORIZON = 5
OUT = Path(rq.PROJECT_ROOT) / "data" / "v9_results" / "drop_gate_trace.json"


def mom_score(close: np.ndarray) -> float:
    """生产动量评分: 10日×0.5 + 20日×0.5."""
    r10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
    r20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
    return 0.5 * r10 + 0.5 * r20


def main() -> None:
    data = rq.load_data()
    common: set | None = None
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        ds = set(data[code]["trade_date"].tolist())
        common = ds if common is None else common & ds
    if rq.DEFENSE in data:
        common &= set(data[rq.DEFENSE]["trade_date"].tolist())
    all_dates = sorted(common)
    trading_dates = all_dates[WARMUP:]
    rebalance_set = set(trading_dates[:: rq.REBALANCE_DAYS])
    dates_idx = {d: i for i, d in enumerate(all_dates)}

    rows = []
    for td in rebalance_set:
        for code in rq.ETF_POOL:
            if code not in data:
                continue
            df = data[code]
            mask = df["trade_date"] <= td
            if mask.sum() < WARMUP:
                continue
            close = df["close"].values[: mask.sum()].astype(float)
            # 触发暴跌过滤?
            triggered = False
            for i in range(-rq.DROP_LOOKBACK, 0):
                dr = (close[i] - close[i - 1]) / close[i - 1]
                if dr < rq.DROP_THRESHOLD:
                    triggered = True
                    break
            if not triggered:
                continue
            ret60 = (close[-1] - close[-61]) / close[-61] if len(close) > 61 else None
            mom = mom_score(close)
            gated_pass = bool(ret60 is not None and ret60 < RET60_THR)
            # 后5日收益
            pos = dates_idx.get(td)
            fwd = None
            if pos is not None and pos + HORIZON < len(all_dates):
                fut = all_dates[pos + HORIZON]
                prow = df[df["trade_date"] == fut]
                if not prow.empty:
                    fwd = float(prow.iloc[0]["close"]) / close[-1] - 1.0
            rows.append(
                {
                    "code": code,
                    "name": rq.ETF_POOL[code],
                    "date": str(td),
                    "ret60": round(ret60, 4) if ret60 is not None else None,
                    "mom": round(float(mom), 4),
                    "gated_pass": bool(gated_pass),
                    "fwd5": round(fwd, 4) if fwd is not None else None,
                }
            )

    df_out = pd.DataFrame(rows)
    total = len(df_out)
    passed = df_out[df_out["gated_pass"]]
    excluded = df_out[~df_out["gated_pass"]]

    print("=" * 74)
    print(
        f"  门控追踪 | 触发暴跌过滤 {total}次 (调仓日口径) | "
        f"放行{len(passed)}次 / 排除{len(excluded)}次"
    )
    print("=" * 74)

    def report(d: pd.DataFrame, label: str) -> dict | None:
        if d.empty or d["fwd5"].isna().all():
            return None
        v = d["fwd5"].dropna()
        cont = (v < 0).mean() * 100
        avg = v.mean() * 100
        print(f"    {label:<24} {len(d):>3}次 | 后5日继续跌 {cont:>3.0f}% | 平均 {avg:+.2f}%")
        return {
            "n": int(len(d)),
            "cont_pct": round(float(cont), 1),
            "avg_fwd": round(float(avg), 2),
        }

    print("\n  【全部触发事件的后5日表现】")
    report(df_out, "全部触发")
    print("\n  【按门控结果分组】")
    g_passed = report(passed, "放行 (ret60<0)")
    g_excluded = report(excluded, "排除 (ret60>=0)")
    print("\n  【按段分组 (放行事件)】")
    segs = [
        ("IS(2020-06~2023-12)", "2020-06-01", "2023-12-31"),
        ("OOS(2024-01~)", "2024-01-01", "9999-12-31"),
    ]
    seg_stats = {}
    for name, s0, s1 in segs:
        sub = passed[(passed["date"] >= s0) & (passed["date"] <= s1)]
        seg_stats[name] = report(sub, name)
    print("\n  【放行事件明细 (含后5日)】")
    pd.set_option("display.width", 120)
    show = passed.copy()
    show["gain_loss"] = show["fwd5"].apply(lambda x: "赚" if (x or 0) > 0 else "亏")
    print(show[["date", "name", "ret60", "mom", "fwd5", "gain_loss"]].to_string(index=False))

    result = {
        "n_triggered": total,
        "n_passed": int(len(passed)),
        "n_excluded": int(len(excluded)),
        "all": report(df_out, "全部触发"),
        "passed": g_passed,
        "excluded": g_excluded,
        "seg_stats": seg_stats,
        "detail": df_out.to_dict(orient="records"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 结果已保存: {OUT}")


if __name__ == "__main__":
    main()
