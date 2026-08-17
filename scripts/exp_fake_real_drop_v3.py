"""真假跌识别实验 v3: 对齐生产 -3% 口径, 验证"趋势位置门控"的区分度.

为"门控版暴跌过滤"提供参数依据:
  生产 DROP_FILTER: 近5日有单日跌>3% → 排除候选 (不分真假跌, 一刀切)
  门控版候选: 仅当趋势位置高(前期大涨)才排除; 前期平淡中大跌放行(假摔概率高)

方法: 收集池内所有ETF"单日大跌>3%"事件, 按 ret60/MA20位置/动量分组,
      对比"触发后5日继续跌(真跌)"比例 —— 有效门控应显著降低平淡组的继续跌率,
      同时保留大涨组的排除保护。

输出: 控制台摘要 + data/v9_results/fake_real_drop_v3.json (供阈值扫描/归档)
用法: uv run python scripts/exp_fake_real_drop_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq

DROP_THR = -0.03  # 对齐生产 DROP_THRESHOLD
HORIZON = 5  # 观察后续5日
OUT = Path(rq.PROJECT_ROOT) / "data" / "v9_results" / "fake_real_drop_v3.json"


def collect_drop_events(data: dict) -> pd.DataFrame:
    """收集所有ETF的单日大跌>3%事件及其因子与后续走势 (无前瞻).

    因子只用当日及之前数据计算; fwd5 用大跌日后第5日收盘, 不引入未来数据.
    """
    events = []
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        df = data[code].sort_values("trade_date").reset_index(drop=True)
        close = df["close"].astype(float).values
        dates = df["trade_date"].tolist()
        for i in range(61, len(close) - HORIZON):
            day_ret = (close[i] - close[i - 1]) / close[i - 1]
            if day_ret < DROP_THR:
                ret60 = (close[i] - close[i - 61]) / close[i - 61]
                ret20 = (close[i] - close[i - 21]) / close[i - 21]
                ret10 = (close[i] - close[i - 11]) / close[i - 11]
                ma20 = float(np.mean(close[i - 20 : i]))
                mom_score = 0.5 * ret10 + 0.5 * ret20  # 生产动量口径
                # 箱体确认因子: 60日区间位置 + MA60斜率 (区分箱体震荡 vs 下跌趋势)
                w60 = close[i - 60 : i]
                hi60, lo60 = float(w60.max()), float(w60.min())
                pos60 = (close[i] - lo60) / (hi60 - lo60) if hi60 > lo60 else 0.5
                ma60_now = float(np.mean(close[i - 60 : i]))
                ma60_prev = float(np.mean(close[i - 65 : i - 5])) if i >= 65 else ma60_now
                ma60_slope = ma60_now - ma60_prev
                box_ok = bool(0.40 <= pos60 <= 0.80 and ma60_slope >= 0)
                fwd5 = (close[i + HORIZON] - close[i]) / close[i]
                events.append(
                    {
                        "code": code,
                        "date": str(dates[i]),
                        "ret60": float(ret60),
                        "ret20": float(ret20),
                        "mom_score": float(mom_score),
                        "above_ma20": bool(close[i] > ma20),
                        "pos60": float(pos60),
                        "ma60_slope": float(ma60_slope),
                        "box_ok": box_ok,
                        "fwd5": float(fwd5),
                    }
                )
    return pd.DataFrame(events)


def stats(d: pd.DataFrame, label: str) -> dict | None:
    """打印并返回一组事件的统计 (继续跌=真跌, 反弹=假摔被误杀)."""
    if d.empty:
        print(f"    {label:<32} 0次")
        return None
    cont = (d["fwd5"] < 0).mean() * 100
    avg = d["fwd5"].mean() * 100
    print(f"    {label:<32} {len(d):>3}次 | 继续跌 {cont:>3.0f}% | 后5日均 {avg:+.2f}%")
    return {"n": len(d), "cont_pct": float(cont), "avg_fwd": float(avg)}


def threshold_scan(df: pd.DataFrame) -> list[dict]:
    """扫描 ret60 阈值: 找到区分度最大的分割点.

    对每个阈值, 低组(ret60<thr, 平淡)与高组(ret60>=thr, 大涨)的继续跌率之差
    为区分度; 目标: 低组继续跌率尽量低(<40%), 高组尽量高(>=50%).
    """
    rows = []
    for thr in np.arange(-0.10, 0.31, 0.05):
        lo = df[df["ret60"] < thr]
        hi = df[df["ret60"] >= thr]
        if len(lo) < 20 or len(hi) < 20:
            continue
        lo_cont = (lo["fwd5"] < 0).mean() * 100
        hi_cont = (hi["fwd5"] < 0).mean() * 100
        rows.append(
            {
                "ret60_thr": round(float(thr), 2),
                "lo_n": len(lo),
                "lo_cont_pct": round(float(lo_cont), 1),
                "hi_n": len(hi),
                "hi_cont_pct": round(float(hi_cont), 1),
                "spread_pp": round(float(hi_cont - lo_cont), 1),
            }
        )
        print(
            f"    ret60<{thr:+.2f}: {len(lo):>3}次 继续跌{lo_cont:>3.0f}% | "
            f"ret60>={thr:+.2f}: {len(hi):>3}次 继续跌{hi_cont:>3.0f}% "
            f"| 区分度 {hi_cont - lo_cont:>+.0f}pp"
        )
    return rows


def main() -> None:
    data = rq.load_data()
    df = collect_drop_events(data)
    total = len(df)
    base_cont = (df["fwd5"] < 0).mean() * 100
    base_avg = df["fwd5"].mean() * 100

    print("=" * 70)
    print(
        f"  真假跌识别 v3 | 单日大跌>{abs(DROP_THR) * 100:.0f}%事件 共{total}次 "
        f"({len(df['code'].unique())}只ETF)"
    )
    print("=" * 70)
    print(
        f"\n  【基准】 大跌后5日: 继续跌(真跌) {base_cont:.0f}% | "
        f"反弹(假摔) {100 - base_cont:.0f}% | 平均 {base_avg:+.2f}%"
    )
    print("          注: 生产过滤-3%口径, 阈值比原实验(-5%)低, 事件更噪")

    print("\n  【因子: 趋势位置 ret60 中位数分组】")
    med = df["ret60"].median()
    print(f"    ret60中位数 = {med:+.1%}")
    stats(df[df["ret60"] <= med], "前期平淡组(ret60<=中位)")
    stats(df[df["ret60"] > med], "前期大涨组(ret60>中位)")

    print("\n  【因子: MA20 位置】")
    stats(df[df["above_ma20"]], "大跌日仍在MA20上方")
    stats(df[~df["above_ma20"]], "大跌日已跌破MA20")

    print("\n  【门控组合 (方案A核心统计)】")
    r = {}
    r["gate_low_ma20"] = stats(
        df[(df["ret60"] <= med) & df["above_ma20"]], "平淡 + 未破MA20 (放行)"
    )
    r["gate_low_mom"] = stats(
        df[(df["ret60"] <= med) & (df["mom_score"] > 0)], "平淡 + 动量>0 (放行)"
    )
    r["gate_low_ma20_mom"] = stats(
        df[(df["ret60"] <= med) & df["above_ma20"] & (df["mom_score"] > 0)],
        "平淡 + 未破MA20 + 动量>0 (放行)",
    )
    r["gate_hi"] = stats(df[df["ret60"] > med], "前期大涨 (照旧排除)")

    print("\n  【箱体确认因子 (区分箱体震荡 vs 下跌趋势)】")
    r["box_ok_all"] = stats(df[df["box_ok"]], "箱体确认 (pos60中段+MA60走平/上)")
    r["box_ng_all"] = stats(df[~df["box_ok"]], "非箱体 (贴近低点/MA60向下)")
    print("  → 仅看 ret60<0 组:")
    lo_ret60 = df[df["ret60"] < 0]
    r["box_ok_low"] = stats(lo_ret60[lo_ret60["box_ok"]], "ret60<0 + 箱体确认 (精化放行)")
    r["box_ng_low"] = stats(lo_ret60[~lo_ret60["box_ok"]], "ret60<0 + 非箱体 (下跌趋势, 不放行)")
    r["box_ok_low_mom"] = stats(
        lo_ret60[lo_ret60["box_ok"] & (lo_ret60["mom_score"] > 0)],
        "ret60<0 + 箱体 + 动量>0 (最终放行)",
    )
    r["box_ng_low_mom"] = stats(
        lo_ret60[~lo_ret60["box_ok"] & (lo_ret60["mom_score"] > 0)],
        "ret60<0 + 非箱体 + 动量>0 (下跌趋势误放行风险)",
    )

    print("\n  【ret60 阈值扫描 (找区分度最大分割点)】")
    scan = threshold_scan(df)

    result = {
        "drop_thr": DROP_THR,
        "horizon": HORIZON,
        "n_events": total,
        "base": {"cont_pct": round(float(base_cont), 1), "avg_fwd": round(float(base_avg), 2)},
        "ret60_median": round(float(med), 4),
        "gates": {k: v for k, v in r.items() if v is not None},
        "threshold_scan": scan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓ 结果已保存: {OUT}")
    print("  判读: 若'放行组'继续跌率明显低于基准且'排除组'接近/高于基准,")
    print("        则门控成立: 放行假摔不误杀, 排除真跌不挨打。")


if __name__ == "__main__":
    main()
